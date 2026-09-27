package domain_test

import (
	"testing"

	"distillery/internal/domain"
)

func TestIsValidPreferenceMethod(t *testing.T) {
	t.Parallel()

	cases := map[domain.PreferenceMethod]bool{
		domain.MethodDPO:  true,
		domain.MethodORPO: true,
		"ppo":             false,
		"":                false,
	}

	for method, want := range cases {
		if got := domain.IsValidPreferenceMethod(method); got != want {
			t.Errorf("IsValidPreferenceMethod(%q) = %v, want %v", method, got, want)
		}
	}
}

func TestPreferenceConfig_ApplyDefaults(t *testing.T) {
	t.Parallel()

	cfg := domain.PreferenceConfig{Method: domain.MethodORPO, Beta: 0.2}.ApplyDefaults()

	if cfg.Method != domain.MethodORPO {
		t.Errorf("expected Method to be preserved, got %v", cfg.Method)
	}

	if cfg.Beta != 0.2 {
		t.Errorf("expected Beta to be preserved, got %v", cfg.Beta)
	}

	defaults := domain.DefaultPreferenceConfig()

	if cfg.LearningRate != defaults.LearningRate {
		t.Errorf("expected LearningRate to fall back to the default %v, got %v", defaults.LearningRate, cfg.LearningRate)
	}

	if cfg.Epochs != defaults.Epochs {
		t.Errorf("expected Epochs to fall back to the default %v, got %v", defaults.Epochs, cfg.Epochs)
	}

	if cfg.MaxPromptLen != defaults.MaxPromptLen || cfg.MaxLen != defaults.MaxLen {
		t.Errorf("expected MaxPromptLen/MaxLen to fall back to defaults, got %v/%v", cfg.MaxPromptLen, cfg.MaxLen)
	}
}

func TestPreferenceConfig_ApplyDefaults_EmptyMethod(t *testing.T) {
	t.Parallel()

	cfg := domain.PreferenceConfig{}.ApplyDefaults()

	if cfg.Method != domain.MethodDPO {
		t.Errorf("expected an empty Method to default to dpo, got %v", cfg.Method)
	}
}
