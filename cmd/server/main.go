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

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/exhaustruct"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/infra/modelstore"
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

	port := envOr("PORT", "8080")
	dataPath := envOr("DATA_PATH", "./data/distillery.json")
	openBrowser := envOr("OPEN_BROWSER", "auto") // auto|true|false.

	// --- Repositories (in-memory + JSON snapshot persistence) ---.
	store := memory.NewStore(dataPath)
	taskRepo := memory.NewTaskRepo(store)
	exampleRepo := memory.NewExampleRepo(store)
	trainingRepo := memory.NewTrainingRepo(store)
	deploymentRepo := memory.NewDeploymentRepo(store)
	feedbackRepo := memory.NewFeedbackRepo(store)
	agentRepo := memory.NewAgentRepo(store)
	ftReqRepo := memory.NewFineTuneRequestRepo(store)
	ftJobRepo := memory.NewFineTuneJobRepo(store)
	ftModelRepo := memory.NewTrainedModelRepo(store)
	ftDatasetRepo := memory.NewDatasetRepo(store)
	modelStoreRepo := memory.NewModelStoreRepo(store)

	// --- Infra (GPU orchestration / model serving) ---.
	//
	// TRAINING_BACKEND selects which adapter implements domain.FineTuner:
	//   "simulation"  → fake progress/loss curves (default; no GPU needed)
	//   "local"       → real QLoRA fine-tuning via the Python trainer worker
	modelSelector := simulation.NewModelSelector()
	synthGen := simulation.NewSyntheticGenerator()

	backend := envOr("TRAINING_BACKEND", "simulation")

	var (
		tuner    domain.FineTuner
		exporter domain.Exporter
	)

	if backend == "local" {
		trainingCfg := localtraining.Config{
			CPUFallback:   envOr("TRAINING_CPU_FALLBACK", "false") == "true",
			MaxJobHistory: envInt("TRAINING_MAX_JOB_HISTORY", 0),
		}
		tuner = localtraining.NewLocalTrainer(&trainingCfg)
		exporter = localtraining.NewLocalExporter(&trainingCfg)
	} else {
		tuner = simulation.NewFineTuner()
		exporter = simulation.NewExporter()
	}

	inferenceEngine := simulation.NewInferenceEngine()

	idGen := usecase.NewRandomIDGenerator()

	// --- Usecases ---.
	taskUC := usecase.NewTaskUsecase(taskRepo, exampleRepo, idGen)
	datasetUC := usecase.NewDatasetUsecase(taskRepo, exampleRepo, synthGen, idGen)
	trainingUC := usecase.NewTrainingUsecase(taskRepo, exampleRepo, trainingRepo, modelSelector, tuner, idGen)
	deploymentUC := usecase.NewDeploymentUsecase(
		taskRepo, trainingRepo, exampleRepo, deploymentRepo,
		inferenceEngine, exporter, idGen,
	)
	feedbackUC := usecase.NewFeedbackUsecase(taskRepo, feedbackRepo, exampleRepo, idGen)

	// --- Model stores (choose / download / upload base + trained models) ---.
	modelTransfer := modelstore.New(envOr("MODEL_CACHE_DIR", ""))
	modelStoreUC := usecase.NewModelStoreUsecase(modelStoreRepo, modelTransfer, idGen)

	// --- Agent orchestration ---.
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(infraagent.NewDatasetValidatorTool(taskRepo))
	_ = toolReg.Register(infraagent.NewModelSelectorTool())
	_ = toolReg.Register(infraagent.NewTrainingControllerTool(taskRepo))
	_ = toolReg.Register(infraagent.NewCodeEvaluatorTool())

	llmProvider := infraagent.NewSimulatedLLMProvider()
	agentOrchUC := usecase.NewAgentOrchestrationUsecase(agentRepo, toolReg, llmProvider, taskRepo, idGen)

	// --- Coding AI Agent (fine-tuning) ---.
	codingAgent := infraagent.NewCodingAgent(ftReqRepo, ftJobRepo, ftDatasetRepo, ftModelRepo)
	ftUC := usecase.NewFineTuneUsecase(codingAgent, ftReqRepo, ftJobRepo, ftModelRepo, ftDatasetRepo, idGen)

	// --- Delivery ---.
	handlers := deliveryhttp.Handlers{
		Task:       deliveryhttp.NewTaskHandler(taskUC),
		Dataset:    deliveryhttp.NewDatasetHandler(datasetUC),
		Training:   deliveryhttp.NewTrainingHandler(trainingUC),
		Deployment: deliveryhttp.NewDeploymentHandler(deploymentUC),
		Feedback:   deliveryhttp.NewFeedbackHandler(feedbackUC),
		Agent:      deliveryhttp.NewAgentHandler(agentOrchUC),
		FineTune:   deliveryhttp.NewFineTuneHandler(ftUC),
		ModelStore: deliveryhttp.NewModelStoreHandler(modelStoreUC),
	}
	router := deliveryhttp.NewRouter(handlers, web.FS())

	addr := ":" + port
	url := "http://localhost:" + port

	if shouldOpenBrowser(openBrowser) {
		go tryOpenBrowser(url)
	}

	log.Printf("Distillery listening on %s (data: %s)", url, dataPath)

	err := http.ListenAndServe(addr, router)
	if err != nil {
		log.Fatal(err)
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
