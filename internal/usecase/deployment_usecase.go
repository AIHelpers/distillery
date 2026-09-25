package usecase

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
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

// DeploymentKind returns the architectural model kind of a deployment so the
// HTTP layer can shape the inference response per kind. Pre-kind records
// default to causal_lm.
func (u *DeploymentUsecase) DeploymentKind(id string) (domain.ModelKind, error) {
	d, err := u.deployments.Get(id)
	if err != nil {
		return "", err
	}

	if d.Kind == "" {
		return domain.DefaultModelKind, nil
	}

	return d.Kind, nil
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

// InvokeStructured runs the deployed endpoint with schema-constrained decoding
// for Track B extraction tasks. When the task declares a JSON schema and the
// configured engine supports grammar-constrained generation, a GBNF grammar is
// generated from the schema and passed to the backend so the output is
// guaranteed to be schema-valid JSON. For tasks without a schema (or engines
// without support) it degrades to the plain Invoke path.
func (u *DeploymentUsecase) InvokeStructured(deploymentID, apiKey, input string) (output string, confidence float64, constrained bool, err error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return "", 0, false, err
	}

	task, err := u.tasks.Get(d.TaskID)
	if err != nil {
		return "", 0, false, err
	}

	grammar := domain.GenerateGBNF(task.JSONSchema)

	engine, ok := u.engine.(domain.ConstrainedInferenceEngine)
	if grammar == "" || !ok || engine == nil {
		out, conf, err := u.Invoke(deploymentID, apiKey, input)

		return out, conf, false, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return "", 0, false, err
	}

	if d.Status != domain.DeploymentActive {
		return "", 0, false, domain.ErrNoDeployment
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return "", 0, false, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return "", 0, false, err
	}

	output, confidence = engine.PredictConstrained(job, usable, input, grammar)

	d.RequestCount++
	_ = u.deployments.Update(d)

	return output, confidence, true, nil
}

// ValidateOutput reports whether an output string satisfies the deployment's
// task JSON schema. Deployments without a schema always return true (nothing
// to validate against).
func (u *DeploymentUsecase) ValidateOutput(deploymentID, output string) bool {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return false
	}

	task, err := u.tasks.Get(d.TaskID)
	if err != nil {
		return false
	}

	if strings.TrimSpace(task.JSONSchema) == "" {
		return true
	}

	return domain.ValidateJSONAgainstSchema(task.JSONSchema, output) == nil
}

// InvokeClassify runs a seq_classifier deployment and returns the structured
// label + per-class scores. It only works for deployments whose kind is
// KindSeqClassifier (or an unset kind for pre-kind records).
func (u *DeploymentUsecase) InvokeClassify(deploymentID, apiKey, input string) (domain.ClassifierPrediction, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.ClassifierPrediction{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.ClassifierPrediction{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.ClassifierPrediction{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.ClassifierInferenceEngine)
	if !ok || engine == nil {
		return domain.ClassifierPrediction{}, errors.New("deployment is not a sequence classifier model")
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.ClassifierPrediction{}, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return domain.ClassifierPrediction{}, err
	}

	threshold := d.ConfidenceThreshold

	pred, err := engine.PredictClassification(job, usable, input, d.LabelMap, threshold)
	if err != nil {
		return domain.ClassifierPrediction{}, err
	}

	d.RequestCount++
	_ = u.deployments.Update(d)

	return pred, nil
}

// InvokeNER runs a token_classifier deployment and returns the labeled spans.
// It only works for deployments whose kind is KindTokenClassifier.
func (u *DeploymentUsecase) InvokeNER(deploymentID, apiKey, input string) (domain.NERPrediction, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.NERPrediction{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.NERPrediction{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.NERPrediction{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.NERInferenceEngine)
	if !ok || engine == nil {
		return domain.NERPrediction{}, errors.New("deployment is not a token classifier model")
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.NERPrediction{}, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return domain.NERPrediction{}, err
	}

	pred, err := engine.PredictSpans(job, usable, input, d.LabelMap)
	if err != nil {
		return domain.NERPrediction{}, err
	}

	d.RequestCount++
	_ = u.deployments.Update(d)

	return pred, nil
}

// BatchResult is one row of a batch inference run.
type BatchResult struct {
	Input      string
	Output     string
	Confidence float64
}

// Embed encodes a list of texts with the deployed embedding model. It only
// works for deployments whose kind is KindEmbedding.
func (u *DeploymentUsecase) Embed(deploymentID, apiKey string, inputs []string, embedType string) (domain.EmbeddingResult, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.EmbeddingResult{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.EmbeddingResult{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.EmbeddingResult{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.EmbeddingEngine)
	if !ok || engine == nil {
		return domain.EmbeddingResult{}, errors.New("deployment is not an embedding model")
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.EmbeddingResult{}, err
	}

	// Normalize the type to one of the two supported prefixes.
	if embedType != "document" {
		embedType = "query"
	}

	result := engine.Embed(job, inputs, embedType)

	d.RequestCount += len(inputs)
	_ = u.deployments.Update(d)

	return result, nil
}

// EmbedCorpus is the "embed my corpus" batch job: it collects every usable
// docs-only example for the deployment's task, embeds each document with the
// deployed embedding model, and returns the vectors as a CSV so users can load
// them into any vector DB (Distillery does not become a vector database).
// The returned job records progress/state; the bytes are the CSV payload.
func (u *DeploymentUsecase) EmbedCorpus(deploymentID, apiKey string) (*domain.CorpusEmbedJob, []byte, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return nil, nil, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return nil, nil, err
	}

	if d.Status != domain.DeploymentActive {
		return nil, nil, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.EmbeddingEngine)
	if !ok || engine == nil {
		return nil, nil, errors.New("deployment is not an embedding model")
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return nil, nil, err
	}

	examples, err := u.usableExamples(d.TaskID)
	if err != nil {
		return nil, nil, err
	}

	// Keep only docs-only rows; skip pair/triplet/graded rows (they aren't
	// standalone corpus documents).
	docTexts := make([]string, 0, len(examples))
	for _, e := range examples {
		if !isDocsOnlyPayload(e.Payload) {
			continue
		}

		var p struct {
			Document string `json:"document"`
		}

		_ = json.Unmarshal(e.Payload, &p)

		if strings.TrimSpace(p.Document) != "" {
			docTexts = append(docTexts, strings.TrimSpace(p.Document))
		}
	}

	if len(docTexts) == 0 {
		return nil, nil, errors.New("no docs-only examples found; import a corpus first (e.g. {\"document\": \"...\"})")
	}

	now := time.Now().UTC()

	embedJob := &domain.CorpusEmbedJob{
		ID:            u.idGen.NewID("ce"),
		DeploymentID:  deploymentID,
		TrainingJobID: job.ID,
		TaskID:        d.TaskID,
		Status:        domain.TrainingRunning,
		TotalDocs:     len(docTexts),
		DoneDocs:      0,
		CreatedAt:     now,
		StartedAt:     &now,
	}

	// Embed every document at once (batching handled by the engine).
	result := engine.Embed(job, docTexts, "document")

	if len(result.Vectors) != len(docTexts) {
		return nil, nil, errors.New("embedding engine returned a mismatched vector count")
	}

	// Write CSV: text, then one column per vector component.
	var buf bytes.Buffer

	cw := csv.NewWriter(&buf)

	header := make([]string, 0, result.Dim+1)
	header = append(header, "text")

	for i := range result.Dim {
		header = append(header, "dim_"+strconv.Itoa(i))
	}

	_ = cw.Write(header)

	for i, text := range docTexts {
		row := make([]string, 0, result.Dim+1)
		row = append(row, text)

		for _, v := range result.Vectors[i] {
			row = append(row, strconv.FormatFloat(float64(v), 'g', -1, 32))
		}

		_ = cw.Write(row)
	}

	cw.Flush()

	if err := cw.Error(); err != nil {
		return nil, nil, err
	}

	done := time.Now().UTC()
	embedJob.Status = domain.TrainingCompleted
	embedJob.DoneDocs = len(docTexts)
	embedJob.CompletedAt = &done

	d.RequestCount += len(docTexts)
	_ = u.deployments.Update(d)

	return embedJob, buf.Bytes(), nil
}

// Rerank scores a query against a list of documents with the deployed
// reranker model. It only works for deployments whose kind is KindReranker.
func (u *DeploymentUsecase) Rerank(deploymentID, apiKey, query string, documents []string, topK int) (domain.RerankResult, error) {
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.RerankResult{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.RerankResult{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.RerankResult{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.RerankerInferenceEngine)
	if !ok || engine == nil {
		return domain.RerankResult{}, errors.New("deployment is not a reranker model")
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.RerankResult{}, err
	}

	result := engine.Rerank(job, query, documents)

	// Apply top_k truncation when requested.
	if topK > 0 && len(result.Ranking) > topK {
		result.Ranking = result.Ranking[:topK]
	}

	d.RequestCount++
	_ = u.deployments.Update(d)

	return result, nil
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

	// Carry the job's kind and the model's label map / threshold so the
	// inference server can decode the head and the API can shape the result
	// per kind (seq_classifier -> label+scores, token_classifier -> entities).
	d := &domain.Deployment{
		ID:            id,
		TaskID:        taskID,
		TrainingJobID: job.ID,
		Kind:          job.Kind,
		Endpoint:      fmt.Sprintf("/api/v1/inference/%s/predict", id),
		Autoscale:     autoscale,
		Status:        domain.DeploymentActive,
		APIKeyHash:    keyHash,
		CreatedAt:     time.Now().UTC(), RequestCount: 0,
	}

	if job.Metrics != nil {
		if len(job.Metrics.LabelMap) > 0 {
			d.LabelMap = job.Metrics.LabelMap
		}

		if job.Metrics.DefaultThreshold > 0 {
			d.ConfidenceThreshold = job.Metrics.DefaultThreshold
		}
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
