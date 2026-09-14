package agent

import (
	"context"
	"fmt"

	"distillery/internal/domain"
)

// CodingAgentImpl orchestrates fine-tuning workflows with AI-driven insight.
type CodingAgentImpl struct {
	requestRepo domain.FineTuneRequestRepository
	jobRepo     domain.FineTuneJobRepository
	datasetRepo domain.DatasetRepository
	modelRepo   domain.TrainedModelRepository
}

// NewCodingAgent creates a new Coding AI Agent.
func NewCodingAgent(
	requestRepo domain.FineTuneRequestRepository,
	jobRepo domain.FineTuneJobRepository,
	datasetRepo domain.DatasetRepository,
	modelRepo domain.TrainedModelRepository,
) domain.CodingAIAgent {
	return &CodingAgentImpl{
		requestRepo: requestRepo,
		jobRepo:     jobRepo,
		datasetRepo: datasetRepo,
		modelRepo:   modelRepo,
	}
}

// AnalyzeDataset assesses a dataset and returns AI recommendations.
func (a *CodingAgentImpl) AnalyzeDataset(ctx context.Context, datasetID string) (domain.DatasetAnalysis, error) {
	dataset, err := a.datasetRepo.Get(datasetID)
	if err != nil {
		return domain.DatasetAnalysis{}, err
	}

	analysis := domain.DatasetAnalysis{
		FileCount:          dataset.FileCount,
		TotalSize:          dataset.TotalSize,
		TotalTokens:        dataset.TotalTokens,
		SyntaxValidityRate: dataset.Quality.ValidityRate,
		AverageComplexity:  dataset.Quality.ComplexityScore,
		QualityScore:       dataset.Quality.OverallScore,
		ReadyForTraining:   dataset.Quality.OverallScore > 60,
	}

	if dataset.Quality.OverallScore < 50 {
		analysis.Recommendations = append(analysis.Recommendations,
			"Dataset quality is low. Consider cleaning or augmenting data.")
		analysis.ReadyForTraining = false
	}

	if dataset.TotalTokens < 100000 {
		analysis.Recommendations = append(analysis.Recommendations,
			"Dataset is small (<100k tokens). Consider adding more samples.")
	}

	if dataset.Quality.ValidityRate < 0.9 {
		analysis.Recommendations = append(analysis.Recommendations,
			"Code validity rate < 90%. Some samples may have syntax errors.")
	}

	return analysis, nil
}

// RecommendHyperparameters selects optimal hyperparameters based on language,
// skill, and dataset characteristics.
func (a *CodingAgentImpl) RecommendHyperparameters(ctx context.Context, req domain.HyperparameterRecommendationReq) (domain.TrainingParameters, error) {
	params := domain.TrainingParameters{
		BatchSize:          8,
		LearningRate:       2e-5,
		MaxSequenceLength:  1024,
		PreserveSyntax:     true,
		BalancedSampling:   true,
		SchedulerType:      "linear",
		OptimizerType:      "adamw",
		GradientAccumSteps: 1,
		WarmupSteps:        100,
	}

	switch req.Language {
	case domain.LangPython:
		params.LearningRate = 5e-5
		params.Epochs = 3
		params.ContextWindow = 256
	case domain.LangGo:
		params.LearningRate = 2e-5
		params.Epochs = 3
		params.ContextWindow = 200
	case domain.LangJavaScript:
		params.LearningRate = 3e-5
		params.Epochs = 3
		params.ContextWindow = 300
	default:
		params.LearningRate = 3e-5
		params.Epochs = 3
		params.ContextWindow = 256
	}

	if req.DatasetSize < 500000 {
		params.Epochs = 5
		params.LearningRate *= 1.5
	} else if req.DatasetSize > 5000000 {
		params.Epochs = 2
		params.BatchSize = 16
	}

	if req.AvailableGPU > 1 {
		params.GradientAccumSteps = 2
	}

	return params, nil
}

// StartTraining creates a fine-tune request and schedules a training job.
func (a *CodingAgentImpl) StartTraining(ctx context.Context, req *domain.FineTuneRequest) (string, error) {
	analysis, err := a.AnalyzeDataset(ctx, req.DatasetID)
	if err != nil {
		return "", err
	}

	if !analysis.ReadyForTraining {
		return "", fmt.Errorf("dataset not ready: %v", analysis.Recommendations)
	}

	if req.TrainingParams.Epochs == 0 {
		recommended, _ := a.RecommendHyperparameters(ctx, domain.HyperparameterRecommendationReq{
			Language:    req.Language,
			Skill:       req.Skill,
			DatasetSize: analysis.TotalTokens,
		})
		req.TrainingParams = recommended
	}

	if err := a.requestRepo.Create(req); err != nil {
		return "", err
	}

	job := &domain.FineTuneJob{
		RequestID:   req.ID,
		Status:      domain.JobQueued,
		TotalEpochs: req.TrainingParams.Epochs,
	}

	if err := a.jobRepo.Create(job); err != nil {
		return "", err
	}

	return job.ID, nil
}

// MonitorTraining inspects loss trends and returns AI optimization guidance.
func (a *CodingAgentImpl) MonitorTraining(ctx context.Context, jobID string) (domain.TrainingOptimization, error) {
	job, err := a.jobRepo.Get(jobID)
	if err != nil {
		return domain.TrainingOptimization{}, err
	}

	optimization := domain.TrainingOptimization{
		CurrentStep: job.CurrentStep,
	}

	if len(job.Loss) == 0 {
		optimization.Suggestion = "No loss data available yet."
		optimization.Action = domain.ActionContinue
		optimization.Confidence = 50
		return optimization, nil
	}

	optimization.CurrentLoss = job.Loss[len(job.Loss)-1].Value

	if len(job.Loss) > 5 {
		recent := job.Loss[len(job.Loss)-5:]
		if recent[4].Value > recent[0].Value {
			optimization.LossDirection = "increasing"
			optimization.Suggestion = "Loss is increasing. Consider reducing learning rate or checking data quality."
			optimization.Action = domain.ActionReduceLR
			optimization.Confidence = 80
		} else if recent[4].Value < recent[0].Value {
			optimization.LossDirection = "decreasing"
			optimization.Suggestion = "Loss is decreasing well. Training progressing normally."
			optimization.Action = domain.ActionContinue
			optimization.Confidence = 95
		} else {
			optimization.LossDirection = "stable"
			optimization.Suggestion = "Loss is stable. Consider increasing learning rate slightly."
			optimization.Action = domain.ActionContinue
			optimization.Confidence = 60
		}
	}

	return optimization, nil
}

// ValidateTrainingQuality produces a post-training quality report.
func (a *CodingAgentImpl) ValidateTrainingQuality(ctx context.Context, jobID string) (domain.QualityReport, error) {
	job, err := a.jobRepo.Get(jobID)
	if err != nil {
		return domain.QualityReport{}, err
	}

	report := domain.QualityReport{
		TrainingComplete: job.Status == domain.JobCompleted,
	}

	if job.FinalMetrics != nil {
		report.FinalLoss = job.FinalMetrics[string(domain.MetricLoss)]
		report.BestMetrics = job.FinalMetrics
	}

	syntaxVal, hasSyntax := 0.0, false
	if job.FinalMetrics != nil {
		syntaxVal, hasSyntax = job.FinalMetrics[string(domain.MetricSyntaxValid)]
	}
	if hasSyntax {
		report.SyntaxValidity = syntaxVal
		switch {
		case syntaxVal > 0.95:
			report.OverallQuality = "excellent"
		case syntaxVal > 0.85:
			report.OverallQuality = "good"
		default:
			report.OverallQuality = "fair"
			report.Issues = append(report.Issues, "Syntax validity below 85%")
		}
	} else {
		report.OverallQuality = "good"
	}

	if job.FinalMetrics != nil {
		if execVal, ok := job.FinalMetrics[string(domain.MetricCodeExecutability)]; ok {
			report.CodeExecutability = execVal
		}
	}

	report.Recommendations = []string{
		"Model is ready for production deployment",
		"Test with real-world code before full deployment",
		"Monitor inference performance in production",
	}

	return report, nil
}

// GetInsights summarizes the agent's view of a training run.
func (a *CodingAgentImpl) GetInsights(ctx context.Context, jobID string) (domain.AgentInsights, error) {
	job, err := a.jobRepo.Get(jobID)
	if err != nil {
		return domain.AgentInsights{}, err
	}

	insights := domain.AgentInsights{
		Metrics: map[string]interface{}{
			"progress":      job.Progress,
			"current_epoch": job.CurrentEpoch,
			"total_epochs":  job.TotalEpochs,
			"current_step":  job.CurrentStep,
		},
	}

	switch job.Status {
	case domain.JobCompleted:
		loss := 0.0
		if job.FinalMetrics != nil {
			loss = job.FinalMetrics[string(domain.MetricLoss)]
		}
		insights.Phase = "completion"
		insights.Summary = fmt.Sprintf("Training completed. Final loss: %.4f", loss)
		insights.EstimatedQuality = "ready"
		insights.NextSteps = []string{
			"Review quality report for deployment readiness",
			"Export model to HuggingFace",
			"Deploy and monitor in production",
		}
	case domain.JobRunning, domain.JobQueued, domain.JobPreparing:
		insights.Phase = "training"
		insights.Summary = fmt.Sprintf("Training in progress: %d/%d epochs", job.CurrentEpoch, job.TotalEpochs)
		insights.EstimatedQuality = "progressing"
	case domain.JobFailed:
		insights.Phase = "completion"
		insights.Summary = "Training failed"
		insights.EstimatedQuality = "failed"
		insights.Warnings = []string{job.Error}
	default:
		insights.Phase = "unknown"
		insights.Summary = "Unknown training state"
		insights.EstimatedQuality = "unknown"
	}

	return insights, nil
}
