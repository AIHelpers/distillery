// Package memory provides in-process repository implementations that
// satisfy the domain repository interfaces. State is periodically
// snapshotted to a JSON file on disk so data survives restarts both
// for desktop use and inside a container with a mounted volume.
package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"distillery/internal/domain"
)

// snapshot is the full application state, serialized as one JSON document.
type snapshot struct {
	Tasks        map[string]*domain.Task            `json:"tasks"`
	Examples     map[string][]*domain.Example       `json:"examples"`      // keyed by task ID.
	TrainingJobs map[string][]*domain.TrainingJob   `json:"training_jobs"` // keyed by task ID.
	Deployments  map[string][]*domain.Deployment    `json:"deployments"`   // keyed by task ID.
	Feedback     map[string][]*domain.Misprediction `json:"feedback"`      // keyed by task ID.
	Agents       map[string]domain.AgentState       `json:"agents"`
}

// Store is the shared in-memory database backing all repositories, with
// optional persistence to a JSON file.
type Store struct {
	mu   sync.RWMutex
	path string

	Tasks        map[string]*domain.Task
	Examples     map[string][]*domain.Example
	TrainingJobs map[string][]*domain.TrainingJob
	Deployments  map[string][]*domain.Deployment
	Feedback     map[string][]*domain.Misprediction
	Agents       map[string]domain.AgentState
}

// NewStore creates a store. If path is non-empty and an existing snapshot
// is found there, it is loaded; otherwise a fresh in-memory store is used.
func NewStore(path string) *Store {
	s := &Store{
		path:         path,
		Tasks:        map[string]*domain.Task{},
		Examples:     map[string][]*domain.Example{},
		TrainingJobs: map[string][]*domain.TrainingJob{},
		Deployments:  map[string][]*domain.Deployment{},
		Feedback:     map[string][]*domain.Misprediction{},
		Agents:       map[string]domain.AgentState{},
	}
	s.load()
	return s
}

func (s *Store) load() {
	if s.path == "" {
		return
	}
	b, err := os.ReadFile(s.path)
	if err != nil {
		return // no snapshot yet, start fresh.
	}
	var snap snapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return
	}
	if snap.Tasks != nil {
		s.Tasks = snap.Tasks
	}
	if snap.Examples != nil {
		s.Examples = snap.Examples
	}
	if snap.TrainingJobs != nil {
		s.TrainingJobs = snap.TrainingJobs
	}
	if snap.Deployments != nil {
		s.Deployments = snap.Deployments
	}
	if snap.Feedback != nil {
		s.Feedback = snap.Feedback
	}
	if snap.Agents != nil {
		s.Agents = snap.Agents
	}
}

// persist writes the current state to disk. Caller must hold s.mu (read or write).
func (s *Store) persist() {
	if s.path == "" {
		return
	}
	snap := snapshot{
		Tasks:        s.Tasks,
		Examples:     s.Examples,
		TrainingJobs: s.TrainingJobs,
		Deployments:  s.Deployments,
		Feedback:     s.Feedback,
		Agents:       s.Agents,
	}
	b, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(s.path), 0o755)
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, s.path)
}
