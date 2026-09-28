// Command server runs Distillery: a no-code fine-tuning task-to-model
// platform. The same binary works two ways:
//
//   - Desktop: run it locally (`./distillery`) and it opens your browser
//     against http://localhost:8080 automatically. State persists to
//     ./data/distillery.json between runs.
//   - Server / container: run it in Docker (see Dockerfile) with PORT
//     and DATA_PATH set as needed, behind any reverse proxy.
package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"time"

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/exhaustruct"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/infra/blob"
	"distillery/internal/infra/modelstore"
	"distillery/internal/infra/pdfraster"
	"distillery/internal/infra/serving"
	"distillery/internal/infra/simulation"
	localtraining "distillery/internal/infra/training"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
	"distillery/web"
)

func main() {
	exhaustructFlag := flag.Bool("exhaustruct", false, "run the exhaustruct struct-rewrite tool and exit")

	flag.Parse()

	if *exhaustructFlag {
		runExhaustruct()
		return
	}

	dataPath := envOr("DATA_PATH", "./data/distillery.json")

	repos := newRepositories(dataPath)
	deps := newDependencies(&repos)
	handlers := newHandlers(deps)
	router := deliveryhttp.NewRouter(handlers, web.FS())

	// Vision-language image retention sweep: uploaded images/PDF pages often
	// carry PII/financial data, so they're deleted per-task after
	// RetentionDays (default domain.DefaultImageRetentionDays) rather than
	// kept forever. Runs once at startup, then on a fixed interval.
	go runRetentionSweep(deps.datasetUC, envDuration("RETENTION_SWEEP_INTERVAL", 24*time.Hour))

	port := envOr("PORT", "8080")
	addr := ":" + port
	url := "http://localhost:" + port

	if shouldOpenBrowser(envOr("OPEN_BROWSER", "auto")) {
		go tryOpenBrowser(url)
	}

	log.Printf("Distillery listening on %s (data: %s)", url, dataPath)

	err := http.ListenAndServe(addr, router)
	if err != nil {
		log.Fatal(err)
	}
}

// repositories holds every in-memory + JSON-snapshot repository the app
// wires usecases from.
type repositories struct {
	taskRepo       domain.TaskRepository
	exampleRepo    domain.ExampleRepository
	trainingRepo   domain.TrainingJobRepository
	deploymentRepo domain.DeploymentRepository
	feedbackRepo   domain.FeedbackRepository
	agentRepo      domain.AgentRepository
	ftReqRepo      domain.FineTuneRequestRepository
	ftJobRepo      domain.FineTuneJobRepository
	ftModelRepo    domain.TrainedModelRepository
	ftDatasetRepo  domain.DatasetRepository
	modelStoreRepo domain.ModelStoreRepository
}

// newRepositories builds every repository on top of a single shared,
// JSON-snapshot-backed in-memory store.
func newRepositories(dataPath string) repositories {
	store := memory.NewStore(dataPath)

	return repositories{
		taskRepo:       memory.NewTaskRepo(store),
		exampleRepo:    memory.NewExampleRepo(store),
		trainingRepo:   memory.NewTrainingRepo(store),
		deploymentRepo: memory.NewDeploymentRepo(store),
		feedbackRepo:   memory.NewFeedbackRepo(store),
		agentRepo:      memory.NewAgentRepo(store),
		ftReqRepo:      memory.NewFineTuneRequestRepo(store),
		ftJobRepo:      memory.NewFineTuneJobRepo(store),
		ftModelRepo:    memory.NewTrainedModelRepo(store),
		ftDatasetRepo:  memory.NewDatasetRepo(store),
		modelStoreRepo: memory.NewModelStoreRepo(store),
	}
}

// dependencies holds the infra adapters and usecases built from repositories.
type dependencies struct {
	datasetUC *usecase.DatasetUsecase

	taskUC       *usecase.TaskUsecase
	trainingUC   *usecase.TrainingUsecase
	deploymentUC *usecase.DeploymentUsecase
	feedbackUC   *usecase.FeedbackUsecase
	modelStoreUC *usecase.ModelStoreUsecase
	agentOrchUC  *usecase.AgentOrchestrationUsecase
	ftUC         *usecase.FineTuneUsecase
}

// newDependencies wires the infra adapters (training/inference backends,
// blob storage, PDF rasterization) and every usecase on top of repos.
func newDependencies(repos *repositories) dependencies {
	// --- Infra (GPU orchestration / model serving) ---.
	//
	// TRAINING_BACKEND selects which adapter implements domain.FineTuner:
	//   "simulation"  → fake progress/loss curves (default; no GPU needed)
	//   "local"       → real QLoRA fine-tuning via the Python trainer worker
	modelSelector := simulation.NewModelSelector()
	synthGen := simulation.NewSyntheticGenerator()

	// Shared progress store for async GGUF conversions (used by both backends).
	ggufProgress := localtraining.NewGGUFProgressStore()

	// BLOB_STORE_DIR is where vision_lm example images (and rasterized PDF
	// pages) are stored -- the "C12 blob storage" foundation the
	// vision-language plan depends on. Shared between the dataset importer
	// and the local trainer (which reads images straight off this
	// directory rather than round-tripping them into each job dir).
	blobRoot := envOr("BLOB_STORE_DIR", "./data/blobs")
	blobStore := blob.NewLocalStore(blobRoot)

	// PYTHON_BIN also selects the interpreter for the (optional) PDF
	// rasterization helper -- a separate, much lighter subprocess than the
	// training worker, so it's wired regardless of TRAINING_BACKEND.
	pdfRasterizer := pdfraster.NewRasterizer(envOr("PYTHON_BIN", "python"))

	tuner, exporter := newTrainingBackend(blobRoot, ggufProgress)
	inferenceEngine := newInferenceEngine()

	idGen := usecase.NewRandomIDGenerator()

	taskUC := usecase.NewTaskUsecase(repos.taskRepo, repos.exampleRepo, idGen)
	datasetUC := usecase.NewDatasetUsecase(repos.taskRepo, repos.exampleRepo, synthGen, idGen).
		WithBlobStore(blobStore).
		WithPDFRasterizer(pdfRasterizer)
	trainingUC := usecase.NewTrainingUsecase(repos.taskRepo, repos.exampleRepo, repos.trainingRepo, modelSelector, tuner, idGen)
	deploymentUC := usecase.NewDeploymentUsecase(
		repos.taskRepo, repos.trainingRepo, repos.exampleRepo, repos.deploymentRepo,
		inferenceEngine, exporter, idGen,
	)
	feedbackUC := usecase.NewFeedbackUsecase(repos.taskRepo, repos.feedbackRepo, repos.exampleRepo, idGen)

	// --- Model stores (choose / download / upload base + trained models) ---.
	modelTransfer := modelstore.New(envOr("MODEL_CACHE_DIR", ""))
	modelStoreUC := usecase.NewModelStoreUsecase(repos.modelStoreRepo, modelTransfer, idGen)

	// --- Agent orchestration ---.
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(infraagent.NewDatasetValidatorTool(repos.taskRepo))
	_ = toolReg.Register(infraagent.NewModelSelectorTool())
	_ = toolReg.Register(infraagent.NewTrainingControllerTool(repos.taskRepo))
	_ = toolReg.Register(infraagent.NewCodeEvaluatorTool())

	llmProvider := infraagent.NewSimulatedLLMProvider()
	agentOrchUC := usecase.NewAgentOrchestrationUsecase(repos.agentRepo, toolReg, llmProvider, repos.taskRepo, idGen)

	// --- Coding AI Agent (fine-tuning) ---.
	codingAgent := infraagent.NewCodingAgent(repos.ftReqRepo, repos.ftJobRepo, repos.ftDatasetRepo, repos.ftModelRepo)
	ftUC := usecase.NewFineTuneUsecase(codingAgent, repos.ftReqRepo, repos.ftJobRepo, repos.ftModelRepo, repos.ftDatasetRepo, idGen)

	return dependencies{
		datasetUC:    datasetUC,
		taskUC:       taskUC,
		trainingUC:   trainingUC,
		deploymentUC: deploymentUC,
		feedbackUC:   feedbackUC,
		modelStoreUC: modelStoreUC,
		agentOrchUC:  agentOrchUC,
		ftUC:         ftUC,
	}
}

// newTrainingBackend selects and wires the FineTuner/Exporter pair per
// TRAINING_BACKEND ("simulation" default, or "local" for real QLoRA training
// via the Python trainer worker).
func newTrainingBackend(blobRoot string, ggufProgress *localtraining.GGUFProgressStore) (domain.FineTuner, domain.Exporter) {
	if envOr("TRAINING_BACKEND", "simulation") == "local" {
		// MODEL_CACHE_DIR is shared by training and GGUF conversion so the
		// base model weights only need to be downloaded once.
		trainingCfg := localtraining.Config{
			CPUFallback:   envOr("TRAINING_CPU_FALLBACK", "false") == "true",
			MaxJobHistory: envInt("TRAINING_MAX_JOB_HISTORY", 0),
			ModelCacheDir: envOr("MODEL_CACHE_DIR", ""),
			BlobRoot:      blobRoot,
		}

		tuner := localtraining.NewLocalTrainer(&trainingCfg)
		localExporter := localtraining.NewLocalExporter(&trainingCfg)
		localExporter.Progress = ggufProgress

		return tuner, localExporter
	}

	// Even in simulation mode, GGUF export delegates to the real production
	// converter (training.LocalExporter) so a real, full-size GGUF is
	// produced from any trained weights on disk instead of a fake
	// pseudo-random file. When no weights exist, the converter fails fast
	// with an actionable "no trained adapter" error.
	ggufConverter := localtraining.NewLocalExporter(&localtraining.Config{
		ModelCacheDir: envOr("MODEL_CACHE_DIR", ""),
	})
	ggufConverter.Progress = ggufProgress

	return simulation.NewFineTuner(), simulation.NewExporterWithGGUF(ggufConverter)
}

// newInferenceEngine selects the deployment-serving backend per
// INFERENCE_BACKEND ("simulation" default nearest-neighbour demo engine, or
// "llamacpp" for a real llama-server process per trained GGUF).
func newInferenceEngine() domain.InferenceEngine {
	if envOr("INFERENCE_BACKEND", "simulation") == "llamacpp" {
		return serving.NewLlamacppEngine(&serving.LlamacppConfig{
			Bin:     envOr("LLAMACPP_BIN", "llama-server"),
			JobsDir: envOr("TRAINING_OUTPUT_DIR", "./data/training"),
			Backend: "llamacpp",
		})
	}

	return simulation.NewInferenceEngine()
}

// newHandlers builds the HTTP delivery layer's handler bundle from deps.
func newHandlers(deps dependencies) deliveryhttp.Handlers {
	return deliveryhttp.Handlers{
		Task:       deliveryhttp.NewTaskHandler(deps.taskUC),
		Dataset:    deliveryhttp.NewDatasetHandler(deps.datasetUC),
		Training:   deliveryhttp.NewTrainingHandler(deps.trainingUC),
		Deployment: deliveryhttp.NewDeploymentHandler(deps.deploymentUC),
		Feedback:   deliveryhttp.NewFeedbackHandler(deps.feedbackUC),
		Agent:      deliveryhttp.NewAgentHandler(deps.agentOrchUC),
		FineTune:   deliveryhttp.NewFineTuneHandler(deps.ftUC),
		ModelStore: deliveryhttp.NewModelStoreHandler(deps.modelStoreUC),
	}
}

// runExhaustruct rewrites composite literals of enforced struct types in the
// internal tree and exits.
func runExhaustruct() {
	total, err := exhaustruct.Run("internal")
	if err != nil {
		log.Fatalf("exhaustruct: %v", err)
	}

	log.Printf("exhaustruct: rewrote %d file(s)", total)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return def
}

// runRetentionSweep runs DatasetUsecase.SweepExpiredBlobs immediately, then
// again every interval, for the lifetime of the process.
func runRetentionSweep(datasetUC *usecase.DatasetUsecase, interval time.Duration) {
	sweep := func() {
		err := datasetUC.SweepExpiredBlobs()
		if err != nil {
			log.Printf("retention sweep: %v", err)
		}
	}

	sweep()

	if interval <= 0 {
		return
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		sweep()
	}
}

func envDuration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}

	d, err := time.ParseDuration(v)
	if err == nil {
		return d
	}

	return def
}

func envInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}

	n, err := strconv.Atoi(v)
	if err == nil {
		return n
	}

	return def
}

func shouldOpenBrowser(mode string) bool {
	switch mode {
	case "true":
		return true
	case "false":
		return false
	default: // "auto": don't try inside a container (no DISPLAY on Linux server images).
		if runtime.GOOS == "linux" && os.Getenv("DISPLAY") == "" && os.Getenv("DISTILLERY_DESKTOP") == "" {
			return false
		}

		return true
	}
}

func tryOpenBrowser(url string) {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}

	_ = cmd.Start()
}
