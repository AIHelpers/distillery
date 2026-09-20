package usecase

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"distillery/internal/domain"
)

type DeploymentUsecase struct {
	tasks       domain.TaskRepository
	jobs        domain.TrainingJobRepository
	examples    domain.ExampleRepository
	deployments domain.DeploymentRepository
	engine      domain.InferenceEngine
	exporter    domain.Exporter
	idGen       IDGenerator
}

func NewDeploymentUsecase(
	tasks domain.TaskRepository,
	jobs domain.TrainingJobRepository,
	examples domain.ExampleRepository,
	deployments domain.DeploymentRepository,
	engine domain.InferenceEngine,
	exporter domain.Exporter,
	idGen IDGenerator,
) *DeploymentUsecase {
	return &DeploymentUsecase{
		tasks:       tasks,
		jobs:        jobs,
		examples:    examples,
		deployments: deployments,
		engine:      engine,
		exporter:    exporter,
		idGen:       idGen,
	}
}

// Deploy exposes the latest completed training job for a task as an API
// endpoint with one-click deployment + autoscaling, as in the product spec.
// It returns the deployment plus the raw API key — the key is only ever
// available at this moment; only its hash is persisted.
func (u *DeploymentUsecase) Deploy(taskID string, autoscale bool) (*domain.Deployment, string, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.LatestCompleted(taskID)
	if err != nil {
		return nil, "", err
	}

	return u.deployJob(taskID, job, autoscale)
}

// DeployVersion deploys a specific (completed) training job version for the
// task, enabling model comparison and rollback to an earlier fine-tune.
func (u *DeploymentUsecase) DeployVersion(taskID, jobID string, autoscale bool) (*domain.Deployment, string, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.Get(jobID)
	if err != nil {
		return nil, "", err
	}

	if job.TaskID != taskID || job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	return u.deployJob(taskID, job, autoscale)
}

func (u *DeploymentUsecase) GetActiveDeployment(taskID string) (*domain.Deployment, error) {
	return u.deployments.GetActiveForTask(taskID)
}

func (u *DeploymentUsecase) ListDeployments(taskID string) ([]*domain.Deployment, error) {
	return u.deployments.ListByTask(taskID)
}

func (u *DeploymentUsecase) StopDeployment(id string) error {
	d, err := u.deployments.Get(id)
	if err != nil {
		return err
	}

	d.Status = domain.DeploymentStopped

	return u.deployments.Update(d)
}

// Invoke calls the deployed endpoint by ID with a raw input string,
// authorizing the caller via their API key first.
func (u *DeploymentUsecase) Invoke(deploymentID, apiKey, input string) (output string, confidence float64, err error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return "", 0, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return "", 0, err
	}

	if d.Status != domain.DeploymentActive {
		return "", 0, domain.ErrNoDeployment
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return "", 0, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return "", 0, err
	}

	output, confidence = u.engine.Predict(job, usable, input)

	d.RequestCount++
	_ = u.deployments.Update(d)

	return output, confidence, nil
}

// BatchResult is one row of a batch inference run.
type BatchResult struct {
	Input      string
	Output     string
	Confidence float64
}

// InvokeBatch runs prediction over many inputs in one authorized call —
// the workload these narrow-task deployments exist to serve efficiently.
func (u *DeploymentUsecase) InvokeBatch(deploymentID, apiKey string, inputs []string) ([]BatchResult, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return nil, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return nil, err
	}

	if d.Status != domain.DeploymentActive {
		return nil, domain.ErrNoDeployment
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return nil, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return nil, err
	}

	results := make([]BatchResult, 0, len(inputs))
	for _, in := range inputs {
		in = strings.TrimSpace(in)
		if in == "" {
			continue
		}

		out, conf := u.engine.Predict(job, usable, in)
		results = append(results, BatchResult{Input: in, Output: out, Confidence: conf})
	}

	d.RequestCount += len(results)
	_ = u.deployments.Update(d)

	return results, nil
}

// Export builds the portable "run anywhere" export package for the task's
// latest completed model.
func (u *DeploymentUsecase) Export(taskID string) (data []byte, filename string, err error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.LatestCompleted(taskID)
	if err != nil {
		return nil, "", err
	}

	return u.exporter.BuildExport(task, job)
}

// ExportGGUF converts the task's latest completed model to GGUF format
// (HomeBred-LLM / llama.cpp compatible) and returns the file bytes + download name.
func (u *DeploymentUsecase) ExportGGUF(taskID string, opts domain.GGUFExportOptions) ([]byte, string, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.LatestCompleted(taskID)
	if err != nil {
		return nil, "", err
	}

	return u.exportGGUF(task, job, opts)
}

// ExportGGUFVersion converts a specific completed job version to GGUF.
func (u *DeploymentUsecase) ExportGGUFVersion(taskID, jobID string, opts domain.GGUFExportOptions) ([]byte, string, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.Get(jobID)
	if err != nil {
		return nil, "", err
	}

	if job.TaskID != taskID {
		return nil, "", domain.ErrNotFound
	}

	return u.exportGGUF(task, job, opts)
}

// exportGGUF routes to the configured GGUF exporter, guarding for the case
// where the configured exporter does not implement domain.GGUFExporter.
func (u *DeploymentUsecase) exportGGUF(task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) ([]byte, string, error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	ggufExporter, ok := u.exporter.(domain.GGUFExporter)
	if !ok || ggufExporter == nil {
		return nil, "", errors.New("configured exporter does not support GGUF conversion")
	}

	return ggufExporter.BuildGGUF(task, job, opts)
}

// StartGGUFAsync starts an async GGUF conversion and returns a session ID
// for polling progress. If the exporter doesn't support async conversion,
// it returns an error.
func (u *DeploymentUsecase) StartGGUFAsync(taskID, jobID, sessionID string, opts domain.GGUFExportOptions) error {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return err
	}

	var job *domain.TrainingJob
	if jobID == "" {
		job, err = u.jobs.LatestCompleted(taskID)
	} else {
		job, err = u.jobs.Get(jobID)
		if err == nil && job.TaskID != taskID {
			err = domain.ErrNotFound
		}
	}

	if err != nil {
		return err
	}

	if job.Status != domain.TrainingCompleted {
		return domain.ErrNoModel
	}

	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return errors.New("configured exporter does not support async GGUF conversion")
	}

	asyncExporter.StartGGUFAsync(sessionID, taskID, jobID, task, job, opts)

	return nil
}

// GetGGUFProgress returns the progress for an async GGUF session.
func (u *DeploymentUsecase) GetGGUFProgress(sessionID string) (*domain.GGUFProgressInfo, error) {
	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return nil, errors.New("async GGUF not available")
	}

	p := asyncExporter.GetGGUFProgress(sessionID)
	if p == nil {
		return nil, domain.ErrNotFound
	}

	return p, nil
}

// GetGGUFResult returns the file path and filename for a ready session.
// The caller streams the file directly from disk — it is never loaded
// into memory.
func (u *DeploymentUsecase) GetGGUFResult(sessionID string) (string, string, error) {
	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return "", "", errors.New("async GGUF not available")
	}

	return asyncExporter.GetGGUFResult(sessionID)
}

// CleanupGGUFSession deletes the converted GGUF file from disk and removes
// the session from the progress store. Called after the file has been
// streamed to the client so large GGUF files don't accumulate.
func (u *DeploymentUsecase) CleanupGGUFSession(sessionID string) {
	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return
	}

	asyncExporter.CleanupGGUFSession(sessionID)
}

// --- unexported helpers ---.

func (u *DeploymentUsecase) deployJob(
	taskID string,
	job *domain.TrainingJob,
	autoscale bool,
) (*domain.Deployment, string, error) {
	// Stop any currently active deployment for this task before deploying anew.
	active, err := u.deployments.GetActiveForTask(taskID)
	if err == nil {
		active.Status = domain.DeploymentStopped
		_ = u.deployments.Update(active)
	}

	id := u.idGen.NewID("dep")

	rawKey, keyHash, err := generateAPIKey()
	if err != nil {
		return nil, "", err
	}

	d := &domain.Deployment{
		ID:            id,
		TaskID:        taskID,
		TrainingJobID: job.ID,
		Endpoint:      fmt.Sprintf("/api/v1/inference/%s/predict", id),
		Autoscale:     autoscale,
		Status:        domain.DeploymentActive,
		APIKeyHash:    keyHash,
		CreatedAt:     time.Now().UTC(), RequestCount: 0,
	}

	err = u.deployments.Create(d)
	if err != nil {
		return nil, "", err
	}

	return d, rawKey, nil
}

// authorize checks the presented raw API key against the deployment's
// stored hash using a constant-time comparison.
func (u *DeploymentUsecase) authorize(d *domain.Deployment, apiKey string) error {
	if apiKey == "" || d.APIKeyHash == "" {
		return domain.ErrUnauthorized
	}

	if subtle.ConstantTimeCompare([]byte(hashAPIKey(apiKey)), []byte(d.APIKeyHash)) != 1 {
		return domain.ErrUnauthorized
	}

	return nil
}

func (u *DeploymentUsecase) usableExamples(taskID string) ([]*domain.Example, error) {
	examples, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	var usable []*domain.Example

	for _, e := range examples {
		if !e.Duplicate && !e.Flagged {
			usable = append(usable, e)
		}
	}

	return usable, nil
}

func generateAPIKey() (raw, hash string, err error) {
	b := make([]byte, 24)

	_, err = rand.Read(b)
	if err != nil {
		return "", "", err
	}

	raw = "sk_" + hex.EncodeToString(b)

	return raw, hashAPIKey(raw), nil
}

func hashAPIKey(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}
