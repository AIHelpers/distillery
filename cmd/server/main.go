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
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime"

	deliveryhttp "distillery/internal/delivery/http"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/infra/simulation"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
	"distillery/web"
)

func main() {
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

	// --- Simulated infra (stands in for GPU orchestration / model serving) ---.
	modelSelector := simulation.NewModelSelector()
	synthGen := simulation.NewSyntheticGenerator()
	tuner := simulation.NewFineTuner()
	inferenceEngine := simulation.NewInferenceEngine()
	exporter := simulation.NewExporter()

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

	// --- Agent orchestration ---.
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(infraagent.NewDatasetValidatorTool(taskRepo))
	_ = toolReg.Register(infraagent.NewModelSelectorTool())
	_ = toolReg.Register(infraagent.NewTrainingControllerTool(taskRepo))

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

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
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
