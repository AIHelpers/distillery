package domain_test

import (
	"testing"

	"distillery/internal/domain"
)

func TestRetentionDeadlinePassed(t *testing.T) {
	t.Parallel()

	const day = int64(24 * 60 * 60)

	now := int64(1_000_000_000)

	cases := []struct {
		name          string
		createdAt     int64
		retentionDays int
		want          bool
	}{
		{"disabled retention (0) never expires", now - 365*day, 0, false},
		{"disabled retention (negative) never expires", now - 365*day, -1, false},
		{"well within window", now - 5*day, 30, false},
		{"exactly at the boundary", now - 30*day, 30, false},
		{"past the window", now - 31*day, 30, true},
		{"far past the window", now - 400*day, 90, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := domain.RetentionDeadlinePassed(tc.createdAt, now, tc.retentionDays)
			if got != tc.want {
				t.Errorf("RetentionDeadlinePassed(%d, %d, %d) = %v, want %v", tc.createdAt, now, tc.retentionDays, got, tc.want)
			}
		})
	}
}

func TestVisionConfig_FreezeVision(t *testing.T) {
	t.Parallel()

	var nilCfg *domain.VisionConfig
	if !nilCfg.FreezeVision() {
		t.Error("expected a nil VisionConfig to default to freezing the vision encoder")
	}

	unset := &domain.VisionConfig{}
	if !unset.FreezeVision() {
		t.Error("expected an unset FreezeVisionEncoder to default to true")
	}

	no := false
	explicit := &domain.VisionConfig{FreezeVisionEncoder: &no}

	if explicit.FreezeVision() {
		t.Error("expected an explicit false to be honored")
	}
}

func TestIsValidModelKind_IncludesVisionLM(t *testing.T) {
	t.Parallel()

	if !domain.IsValidModelKind(domain.KindVisionLM) {
		t.Error("expected vision_lm to be a valid model kind")
	}
}
