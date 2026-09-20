package simulation_test

import (
	"testing"

	"distillery/internal/infra/simulation"
)

func TestRound2(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   float64
		want float64
	}{
		{1.2356, 1.23},
		{0.999, 0.99},
		{0.05, 0.05},
		{0.049, 0.04},
	}

	for _, c := range cases {
		if got := simulation.Round2(c.in); got != c.want {
			t.Errorf("Round2(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestSafeName(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   string
		want string
	}{
		{"Hello World", "Hello-World"},
		{"My_Task-2", "My-Task-2"},
		{"Symbols!@#", "Symbols"}, // Non-alphanumerics dropped, separators replaced.
		{"", "task"},
		{"123abc", "123abc"},
	}

	for _, c := range cases {
		got := simulation.SafeName(c.in)
		if got != c.want {
			t.Errorf("SafeName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSafeName_EmptyReturnsTask(t *testing.T) {
	t.Parallel()

	got := simulation.SafeName("   ")
	if got != "---" {
		t.Errorf("expected '---' for spaces, got %q", got)
	}

	got = simulation.SafeName("!!!@@@")
	if got != "task" {
		t.Errorf("expected 'task' for pure symbols, got %q", got)
	}
}
