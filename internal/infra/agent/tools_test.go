package agent_test

import (
	"context"
	"testing"

	"distillery/internal/domain"
	infraagent "distillery/internal/infra/agent"
)

func TestDatasetValidatorTool_Name(t *testing.T) {
	tool := infraagent.NewDatasetValidatorTool(nil)
	if tool.Name() != "dataset_validator" {
		t.Errorf("expected name 'dataset_validator', got %q", tool.Name())
	}
}

func TestDatasetValidatorTool_Execute(t *testing.T) {
	tool := infraagent.NewDatasetValidatorTool(nil)

	out, err := tool.Execute(context.Background(), domain.ToolInput{
		Params: map[string]interface{}{"datasetID": "ds_1"},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if out.Error != "" {
		t.Fatalf("expected no error in output, got %q", out.Error)
	}
	result, ok := out.Result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map result, got %T", out.Result)
	}
	if result["valid"] != true {
		t.Errorf("expected valid true, got %v", result["valid"])
	}
	if result["recordCount"] != 1000 {
		t.Errorf("expected recordCount 1000, got %v", result["recordCount"])
	}
}

func TestModelSelectorTool_Execute(t *testing.T) {
	tool := infraagent.NewModelSelectorTool()

	out, err := tool.Execute(context.Background(), domain.ToolInput{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	result, ok := out.Result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map result, got %T", out.Result)
	}
	if result["modelID"] != "gpt2-medium" {
		t.Errorf("expected modelID 'gpt2-medium', got %v", result["modelID"])
	}
}

func TestTrainingControllerTool_Execute_Valid(t *testing.T) {
	tool := infraagent.NewTrainingControllerTool(nil)

	out, err := tool.Execute(context.Background(), domain.ToolInput{
		Params: map[string]interface{}{
			"baseModel": "gpt2",
			"datasetID": "ds_1",
		},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if out.Error != "" {
		t.Fatalf("expected no error in output, got %q", out.Error)
	}
	result, ok := out.Result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map result, got %T", out.Result)
	}
	if result["status"] != "started" {
		t.Errorf("expected status 'started', got %v", result["status"])
	}
}

func TestTrainingControllerTool_Execute_MissingParams(t *testing.T) {
	tool := infraagent.NewTrainingControllerTool(nil)

	out, err := tool.Execute(context.Background(), domain.ToolInput{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if out.Error == "" {
		t.Error("expected error when baseModel/datasetID missing")
	}
}

func TestTools_ImplementDomainTool(t *testing.T) {
	tools := []domain.Tool{
		infraagent.NewDatasetValidatorTool(nil),
		infraagent.NewModelSelectorTool(),
		infraagent.NewTrainingControllerTool(nil),
	}
	for _, tool := range tools {
		if tool.Name() == "" {
			t.Errorf("expected non-empty tool name")
		}
		if tool.Description() == "" {
			t.Errorf("tool %q: expected non-empty description", tool.Name())
		}
		if tool.InputSchema() == nil {
			t.Errorf("tool %q: expected non-nil schema", tool.Name())
		}
	}
}
