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
	"sort"
	"strconv"
	"strings"
	"time"

	"distillery/internal/domain"
)

// Sentinel errors for deployment/inference-engine mismatches and exporter
// capability gaps.
var (
	ErrNotSequenceClassifier      = errors.New("deployment is not a sequence classifier model")
	ErrNotTokenClassifier         = errors.New("deployment is not a token classifier model")
	ErrNotVisionLanguageModel     = errors.New("deployment is not a vision-language model")
	ErrNotEmbeddingModel          = errors.New("deployment is not an embedding model")
	ErrNoDocsOnlyExamples         = errors.New("no docs-only examples found; import a corpus first (e.g. {\"document\": \"...\"})")
	ErrMismatchedVectorCount      = errors.New("embedding engine returned a mismatched vector count")
	ErrNotReranker                = errors.New("deployment is not a reranker model")
	ErrNotASRModel                = errors.New("deployment is not a speech-to-text model")
	ErrNotTabularModel            = errors.New("deployment is not a tabular model")
	ErrNotTimeSeriesModel         = errors.New("deployment is not a time-series forecasting model")
	ErrGGUFExportUnsupported      = errors.New("configured exporter does not support GGUF conversion")
	ErrAsyncGGUFUnsupported       = errors.New("configured exporter does not support async GGUF conversion")
	ErrAsyncGGUFResultUnavailable = errors.New("async GGUF not available")
)

type DeploymentUsecase struct {
	tasks       domain.TaskRepository
	jobs        domain.TrainingJobRepository
	examples    domain.ExampleRepository
	deployments domain.DeploymentRepository
	engine      domain.InferenceEngine
	exporter    domain.Exporter
	// tabular/forecaster serve the plan-08 tabular and time_series kinds.
	// Either may be nil (backend without tabular support); the invoke
	// methods then return the corresponding sentinel error.
	tabular    domain.TabularInferenceEngine
	forecaster domain.ForecastInferenceEngine
	tableRows  domain.TableRowSource
	idGen      IDGenerator
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
	return NewDeploymentUsecaseFull(tasks, jobs, examples, deployments, engine, exporter, nil, nil, nil, idGen)
}

// NewDeploymentUsecaseFull is NewDeploymentUsecase plus the tabular and
// forecast inference engines used by InvokeTable/InvokeForecast.
func NewDeploymentUsecaseFull(
	tasks domain.TaskRepository,
	jobs domain.TrainingJobRepository,
	examples domain.ExampleRepository,
	deployments domain.DeploymentRepository,
	engine domain.InferenceEngine,
	exporter domain.Exporter,
	tabular domain.TabularInferenceEngine,
	forecaster domain.ForecastInferenceEngine,
	tableRows domain.TableRowSource,
	idGen IDGenerator,
) *DeploymentUsecase {
	return &DeploymentUsecase{
		tasks:       tasks,
		jobs:        jobs,
		examples:    examples,
		deployments: deployments,
		engine:      engine,
		exporter:    exporter,
		tabular:     tabular,
		forecaster:  forecaster,
		tableRows:   tableRows,
		idGen:       idGen,
	}
}

// Deploy exposes the latest completed training job for a task as an API
// endpoint with one-click deployment + autoscaling, as in the product spec.
// It returns the deployment plus the raw API key — the key is only ever
// available at this moment; only its hash is persisted.
func (u *DeploymentUsecase) Deploy(taskID string, autoscale bool, force ...bool) (*domain.Deployment, string, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	job, err := u.jobs.LatestCompleted(taskID)
	if err != nil {
		return nil, "", err
	}

	err = checkRegressionGate(job, len(force) > 0 && force[0])
	if err != nil {
		return nil, "", err
	}

	return u.deployJob(taskID, job, autoscale)
}

// DeployVersion deploys a specific (completed) training job version for the
// task, enabling model comparison and rollback to an earlier fine-tune.
//
// Definition of done's regression gate: deploying a preference-tuned
// (DPO/ORPO) job that regressed on its parent SFT job's eval metric is
// blocked unless force is passed, so a drifted model doesn't silently
// replace a good one.
func (u *DeploymentUsecase) DeployVersion(taskID, jobID string, autoscale bool, force ...bool) (*domain.Deployment, string, error) {
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

	err = checkRegressionGate(job, len(force) > 0 && force[0])
	if err != nil {
		return nil, "", err
	}

	return u.deployJob(taskID, job, autoscale)
}

// checkRegressionGate blocks deploying a preference-tuned job whose
// regression check (computed by the trainer against the parent SFT job's
// eval split — see PreferenceConfig/dpo.py) failed, unless the caller
// forces it through.
func checkRegressionGate(job *domain.TrainingJob, force bool) error {
	if force || job == nil || job.Kind != domain.KindPreferenceLM || job.Metrics == nil {
		return nil
	}

	if job.Metrics.RegressionChecked && !job.Metrics.RegressionPassed {
		return fmt.Errorf(
			"%w: %s dropped from %.4f to %.4f (delta %.4f)",
			domain.ErrRegressionFailed,
			job.Metrics.RegressionMetric,
			job.Metrics.RegressionBase,
			job.Metrics.RegressionValue,
			job.Metrics.RegressionDelta,
		)
	}

	return nil
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
		return domain.ClassifierPrediction{}, ErrNotSequenceClassifier
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
		return domain.NERPrediction{}, ErrNotTokenClassifier
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

// InvokeVision serves a deployed vision_lm model: an image plus a prompt in,
// generated text (+ parsed JSON when it validates) out. It only works for
// deployments whose kind is KindVisionLM.
func (u *DeploymentUsecase) InvokeVision(deploymentID, apiKey string, imageBytes []byte, prompt string) (domain.VisionPrediction, error) { //nolint:dupl,lll // parallel tabular/forecast paths share shape but differ in types
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.VisionPrediction{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.VisionPrediction{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.VisionPrediction{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.VisionInferenceEngine)
	if !ok || engine == nil {
		return domain.VisionPrediction{}, ErrNotVisionLanguageModel
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.VisionPrediction{}, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return domain.VisionPrediction{}, err
	}

	pred, err := engine.PredictImage(job, usable, imageBytes, prompt)
	if err != nil {
		return domain.VisionPrediction{}, err
	}

	d.RequestCount++
	_ = u.deployments.Update(d)

	return pred, nil
}

// InvokeTranscribe serves a deployed asr model: raw audio bytes in, a
// transcript (plus timestamped segments for long files) out. It only works
// for deployments whose kind is KindASR.
func (u *DeploymentUsecase) InvokeTranscribe(deploymentID, apiKey string, audioBytes []byte, language string) (domain.ASRPrediction, error) { //nolint:dupl,lll // parallel tabular/forecast paths share shape but differ in types
	d, err := u.deployments.Get(deploymentID)
	if err != nil {
		return domain.ASRPrediction{}, err
	}

	err = u.authorize(d, apiKey)
	if err != nil {
		return domain.ASRPrediction{}, err
	}

	if d.Status != domain.DeploymentActive {
		return domain.ASRPrediction{}, domain.ErrNoDeployment
	}

	engine, ok := u.engine.(domain.ASRInferenceEngine)
	if !ok || engine == nil {
		return domain.ASRPrediction{}, ErrNotASRModel
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.ASRPrediction{}, err
	}

	usable, err := u.usableExamples(d.TaskID)
	if err != nil {
		return domain.ASRPrediction{}, err
	}

	pred, err := engine.Transcribe(job, usable, audioBytes, language)
	if err != nil {
		return domain.ASRPrediction{}, err
	}

	d.RequestCount++
	_ = u.deployments.Update(d)

	return pred, nil
}

// InvokeTable serves a deployed tabular model: a feature row in (validated
// against the deployment's stored feature schema with clear errors for missing
// fields and unknown categories), prediction + top factors out. It only works
// for deployments whose kind is KindTabular.
func (u *DeploymentUsecase) InvokeTable(deploymentID, apiKey string, input map[string]interface{}) (domain.TablePrediction, error) {
	d, job, err := u.activeTableDeployment(deploymentID, apiKey, domain.KindTabular)
	if err != nil {
		return domain.TablePrediction{}, err
	}

	if u.tabular == nil {
		return domain.TablePrediction{}, ErrNotTabularModel
	}

	row, err := d.FeatureSchema.ValidateRow(input)
	if err != nil {
		return domain.TablePrediction{}, err
	}

	pred, err := u.tabular.PredictTable(job, u.tableRows, d.FeatureSchema, row)
	if err != nil {
		return domain.TablePrediction{}, err
	}

	u.countRequests(d, 1)

	return pred, nil
}

// TableBatchResult is one scored row of a tabular batch run.
type TableBatchResult struct {
	Prediction domain.TablePrediction
	Error      string
}

// InvokeTableBatch scores many rows with a single authorization; rows that
// fail schema validation are reported per row instead of failing the batch.
func (u *DeploymentUsecase) InvokeTableBatch(
	deploymentID, apiKey string, rows []map[string]interface{},
) ([]TableBatchResult, error) {
	d, job, err := u.activeTableDeployment(deploymentID, apiKey, domain.KindTabular)
	if err != nil {
		return nil, err
	}

	if u.tabular == nil {
		return nil, ErrNotTabularModel
	}

	out := make([]TableBatchResult, len(rows))
	scored := 0

	for i, in := range rows {
		row, verr := d.FeatureSchema.ValidateRow(in)
		if verr != nil {
			out[i].Error = verr.Error()

			continue
		}

		pred, perr := u.tabular.PredictTable(job, u.tableRows, d.FeatureSchema, row)
		if perr != nil {
			out[i].Error = perr.Error()

			continue
		}

		out[i].Prediction = pred
		scored++
	}

	u.countRequests(d, scored)

	return out, nil
}

// InvokeForecast serves a deployed time-series model: optional caller-supplied
// history rows (falling back to the training series) in, forecast points with
// prediction intervals out. It only works for KindTimeSeries deployments.
func (u *DeploymentUsecase) InvokeForecast(deploymentID, apiKey string, req domain.ForecastRequest) (domain.ForecastResult, error) {
	d, job, err := u.activeTableDeployment(deploymentID, apiKey, domain.KindTimeSeries)
	if err != nil {
		return domain.ForecastResult{}, err
	}

	if u.forecaster == nil {
		return domain.ForecastResult{}, ErrNotTimeSeriesModel
	}

	// Fall back to the job's configured horizon when the caller omits one.
	if req.Horizon <= 0 && job.Forecast != nil {
		req.Horizon = job.Forecast.Horizon
	}

	if req.Horizon <= 0 || req.Horizon > maxForecastHorizon {
		return domain.ForecastResult{}, fmt.Errorf("%w: horizon must be between 1 and %d", domain.ErrInvalidInput, maxForecastHorizon)
	}

	result, err := u.forecaster.Forecast(job, u.tableRows, req)
	if err != nil {
		return domain.ForecastResult{}, err
	}

	u.countRequests(d, 1)

	return result, nil
}

// InvokeForecastBatch forecasts every series found in a history CSV (one
// forecast per distinct item_id, or one for single-series models).
func (u *DeploymentUsecase) InvokeForecastBatch(
	deploymentID, apiKey string, history []map[string]interface{}, horizon int,
) ([]domain.ForecastResult, error) {
	d, job, err := u.activeTableDeployment(deploymentID, apiKey, domain.KindTimeSeries)
	if err != nil {
		return nil, err
	}

	if u.forecaster == nil {
		return nil, ErrNotTimeSeriesModel
	}

	if horizon <= 0 && job.Forecast != nil {
		horizon = job.Forecast.Horizon
	}

	if horizon <= 0 || horizon > maxForecastHorizon {
		return nil, fmt.Errorf("%w: horizon must be between 1 and %d", domain.ErrInvalidInput, maxForecastHorizon)
	}

	ids := []string{""}

	if job.Forecast != nil && job.Forecast.ItemID != "" && len(history) > 0 {
		seen := map[string]bool{}
		ids = nil

		for _, r := range history {
			id := strings.TrimSpace(fmt.Sprintf("%v", r[job.Forecast.ItemID]))
			if id != "" && id != "<nil>" && !seen[id] {
				seen[id] = true
				ids = append(ids, id)
			}
		}

		sort.Strings(ids)
	}

	const maxBatchSeries = 500

	if len(ids) > maxBatchSeries {
		return nil, fmt.Errorf("%w: at most %d series per batch", domain.ErrInvalidInput, maxBatchSeries)
	}

	out := make([]domain.ForecastResult, 0, len(ids))

	for _, id := range ids {
		res, ferr := u.forecaster.Forecast(job, u.tableRows, domain.ForecastRequest{History: history, Horizon: horizon, ItemID: id})
		if ferr != nil {
			return nil, ferr
		}

		out = append(out, res)
	}

	u.countRequests(d, len(out))

	return out, nil
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
		return domain.EmbeddingResult{}, ErrNotEmbeddingModel
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return domain.EmbeddingResult{}, err
	}

	// Normalize the type to one of the two supported prefixes.
	if embedType != fieldDocument {
		embedType = fieldQuery
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
		return nil, nil, ErrNotEmbeddingModel
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
		return nil, nil, ErrNoDocsOnlyExamples
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
	result := engine.Embed(job, docTexts, fieldDocument)

	if len(result.Vectors) != len(docTexts) {
		return nil, nil, ErrMismatchedVectorCount
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

	err = cw.Error()
	if err != nil {
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
		return domain.RerankResult{}, ErrNotReranker
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
func (u *DeploymentUsecase) ExportGGUF(taskID string, opts domain.GGUFExportOptions) (data []byte, filename string, err error) {
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
func (u *DeploymentUsecase) ExportGGUFVersion(taskID, jobID string, opts domain.GGUFExportOptions) (data []byte, filename string, err error) {
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

// errGGUFUnsupportedVisionLM is returned when a GGUF export/conversion is
// requested for a vision_lm job. Per the plan, GGUF export is only offered
// "for verified architectures" — a multimodal projector's llama.cpp support
// is architecture-specific and hasn't been verified for the vision_lm
// catalog, so this is refused explicitly rather than attempting a
// conversion pipeline built for text-only causal LMs and producing a
// broken or misleading artifact.
var errGGUFUnsupportedVisionLM = fmt.Errorf(
	"%w: GGUF export is not supported for vision-language models "+
		"(unverified llama.cpp multimodal support) — use the portable safetensors export instead",
	domain.ErrInvalidInput,
)

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

	if job.Kind == domain.KindVisionLM {
		return errGGUFUnsupportedVisionLM
	}

	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return ErrAsyncGGUFUnsupported
	}

	asyncExporter.StartGGUFAsync(sessionID, taskID, jobID, task, job, opts)

	return nil
}

// GetGGUFProgress returns the progress for an async GGUF session.
func (u *DeploymentUsecase) GetGGUFProgress(sessionID string) (*domain.GGUFProgressInfo, error) {
	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return nil, ErrAsyncGGUFResultUnavailable
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
func (u *DeploymentUsecase) GetGGUFResult(sessionID string) (path, filename string, err error) {
	asyncExporter, ok := u.exporter.(domain.AsyncGGUFExporter)
	if !ok || asyncExporter == nil {
		return "", "", ErrAsyncGGUFResultUnavailable
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

// activeTableDeployment authorizes the caller and returns the deployment and
// its training job, checking the deployment serves the wanted kind.
func (u *DeploymentUsecase) activeTableDeployment(
	deploymentID, apiKey string, kind domain.ModelKind,
) (*domain.Deployment, *domain.TrainingJob, error) {
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

	if d.Kind != kind {
		if kind == domain.KindTabular {
			return nil, nil, ErrNotTabularModel
		}

		return nil, nil, ErrNotTimeSeriesModel
	}

	job, err := u.jobs.Get(d.TrainingJobID)
	if err != nil {
		return nil, nil, err
	}

	return d, job, nil
}

func (u *DeploymentUsecase) countRequests(d *domain.Deployment, n int) {
	d.RequestCount += n
	_ = u.deployments.Update(d)
}

// --- unexported helpers ---.

// exportGGUF routes to the configured GGUF exporter, guarding for the case
// where the configured exporter does not implement domain.GGUFExporter.
func (u *DeploymentUsecase) exportGGUF(task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) (data []byte, filename string, err error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	if job.Kind == domain.KindVisionLM {
		return nil, "", errGGUFUnsupportedVisionLM
	}

	ggufExporter, ok := u.exporter.(domain.GGUFExporter)
	if !ok || ggufExporter == nil {
		return nil, "", ErrGGUFExportUnsupported
	}

	return ggufExporter.BuildGGUF(task, job, opts)
}

// inferenceEndpoint is the URL clients call for a deployment of the given kind.
func inferenceEndpoint(id string, kind domain.ModelKind) string {
	if kind == domain.KindTimeSeries {
		return fmt.Sprintf("/api/v1/inference/%s/forecast", id)
	}

	return fmt.Sprintf("/api/v1/inference/%s/predict", id)
}

func (u *DeploymentUsecase) deployJob(
	taskID string,
	job *domain.TrainingJob,
	autoscale bool,
) (*domain.Deployment, string, error) {
	if job.Kind == domain.KindTabular && (job.Metrics == nil || job.Metrics.FeatureSchema == nil) {
		return nil, "", fmt.Errorf("%w: the trained model recorded no feature schema", domain.ErrNoModel)
	}

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
		Endpoint:      inferenceEndpoint(id, job.Kind),
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

		// Tabular deployments copy the trained feature schema for input
		// validation at predict time.
		d.FeatureSchema = job.Metrics.FeatureSchema
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
