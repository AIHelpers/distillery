package domain_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"distillery/internal/domain"
)

func testSchema() *domain.FeatureSchema {
	minV, maxV := 0.0, 100.0

	return &domain.FeatureSchema{
		Target: "churn", Task: domain.TabularClassification, Classes: []string{"no", "yes"},
		Features: []domain.TableFeature{
			{Name: "age", Type: domain.FeatureNumeric, Min: &minV, Max: &maxV},
			{Name: "plan", Type: domain.FeatureCategorical, Categories: []string{"Basic", "Pro"}},
			{Name: "region", Type: domain.FeatureCategorical, Categories: []string{"eu"}, OpenVocabulary: true, Optional: true},
			{Name: "signup", Type: domain.FeatureDatetime, Optional: true},
		},
	}
}

func TestValidateRow(t *testing.T) {
	t.Parallel()

	fs := testSchema()

	t.Run("canonicalises categories and coerces numbers", func(t *testing.T) {
		t.Parallel()

		v, err := fs.ValidateRow(map[string]interface{}{"age": "41", "plan": " PRO ", "extra": 1})
		if err != nil {
			t.Fatal(err)
		}

		if v.Canonical["plan"] != "Pro" || v.Canonical["age"] != 41.0 {
			t.Errorf("canonical = %v", v.Canonical)
		}
	})

	t.Run("open vocabulary accepts unseen values", func(t *testing.T) {
		t.Parallel()

		v, err := fs.ValidateRow(map[string]interface{}{"age": 3, "plan": "basic", "region": "mars"})
		if err != nil || v.Cats["region"] != "mars" {
			t.Fatalf("v=%v err=%v", v, err)
		}
	})

	t.Run("datetime is parsed", func(t *testing.T) {
		t.Parallel()

		v, err := fs.ValidateRow(map[string]interface{}{"age": 3, "plan": "basic", "signup": "2024-03-05"})
		if err != nil || v.Canonical["signup"] != "2024-03-05T00:00:00Z" {
			t.Fatalf("v=%v err=%v", v, err)
		}
	})

	bad := map[string]map[string]interface{}{
		"missing required":    {"plan": "pro"},
		"unknown category":    {"age": 3, "plan": "gold"},
		"not numeric":         {"age": "12abc", "plan": "pro"},
		"NaN":                 {"age": "NaN", "plan": "pro"},
		"infinite":            {"age": math.Inf(1), "plan": "pro"},
		"bad date":            {"age": 3, "plan": "pro", "signup": "yesterday"},
		"empty required text": {"age": 3, "plan": "  "},
	}

	for name, in := range bad {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			_, err := fs.ValidateRow(in)
			if !errors.Is(err, domain.ErrInvalidInput) {
				t.Errorf("want ErrInvalidInput, got %v", err)
			}
		})
	}

	_, err := fs.ValidateRow(map[string]interface{}{"age": 3, "plan": "gold"})
	if err == nil || !strings.Contains(err.Error(), "Basic, Pro") {
		t.Errorf("unknown-category error should list the allowed values: %v", err)
	}

	_, err = (*domain.FeatureSchema)(nil).ValidateRow(nil)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("nil schema: %v", err)
	}
}

func TestTableToFloat(t *testing.T) {
	t.Parallel()

	for in, want := range map[interface{}]float64{"1.5": 1.5, " 2 ": 2, 3.0: 3, 4: 4, "1e3": 1000} {
		got, ok := domain.TableToFloat(in)
		if !ok || got != want {
			t.Errorf("TableToFloat(%v) = %v,%v", in, got, ok)
		}
	}

	for _, in := range []interface{}{"1,5", "abc", "", "NaN", "Inf", math.NaN(), true, nil, "12abc"} {
		if _, ok := domain.TableToFloat(in); ok {
			t.Errorf("TableToFloat(%v) should fail", in)
		}
	}
}

func TestParseTableTime(t *testing.T) {
	t.Parallel()

	for _, s := range []string{"2024-03-05", "2024-03-05 10:20:30", "2024-03-05T10:20:30Z", "05.03.2024", "03/05/2024", "2024-03"} {
		if _, ok := domain.ParseTableTime(s); !ok {
			t.Errorf("%q should parse", s)
		}
	}

	for _, s := range []string{"", "2024", "1999", "hello", "13/45/2024"} {
		if _, ok := domain.ParseTableTime(s); ok {
			t.Errorf("%q should not parse", s)
		}
	}
}
