package domain_test

import (
	"testing"

	"distillery/internal/domain"
)

func TestModelStoreTypes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		got  domain.ModelStoreType
		want string
	}{
		{domain.ModelStoreLocal, "local"},
		{domain.ModelStoreHuggingFace, "huggingface"},
		{domain.ModelStoreS3, "s3"},
	}

	for _, c := range cases {
		if string(c.got) != c.want {
			t.Errorf("expected %q, got %q", c.want, c.got)
		}
	}
}

func TestModelStoreKinds(t *testing.T) {
	t.Parallel()

	if string(domain.ModelStoreBase) != "base" {
		t.Errorf("expected 'base', got %q", domain.ModelStoreBase)
	}

	if string(domain.ModelStoreTrained) != "trained" {
		t.Errorf("expected 'trained', got %q", domain.ModelStoreTrained)
	}
}

func TestErrStoreUnsupported(t *testing.T) {
	t.Parallel()

	if domain.ErrStoreUnsupported == nil {
		t.Fatal("expected non-nil sentinel error")
	}

	if domain.ErrStoreUnsupported.Error() == "" {
		t.Error("expected descriptive error message")
	}
}
