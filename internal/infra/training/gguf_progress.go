package training

import (
	"sync"
	"time"
)

// GGUFProgressStatus represents the current state of an async GGUF conversion.
type GGUFProgressStatus string

const (
	GGUFStatusRunning GGUFProgressStatus = "running"
	GGUFStatusReady   GGUFProgressStatus = "ready"
	GGUFStatusError   GGUFProgressStatus = "error"
)

// GGUFProgress tracks a single async GGUF conversion session.
type GGUFProgress struct {
	TaskID   string             `json:"task_id"`
	JobID    string             `json:"job_id,omitempty"`
	Status   GGUFProgressStatus `json:"status"`
	Step     string             `json:"step"`
	Percent  int                `json:"percent"`
	Detail   string             `json:"detail,omitempty"`
	Filename string             `json:"filename,omitempty"`
	Size     int64              `json:"size,omitempty"`
	Error    string             `json:"error,omitempty"`
	Quant    string             `json:"quantization"`

	filePath  string // path to the GGUF file on disk (streamed, not loaded into memory).
	createdAt time.Time
	updatedAt time.Time
}

// GGUFProgressStore is a thread-safe in-memory store for async GGUF
// conversion sessions. It maps session IDs to GGUFProgress entries.
type GGUFProgressStore struct {
	mu       sync.RWMutex
	sessions map[string]*GGUFProgress
}

// NewGGUFProgressStore creates a new empty progress store.
func NewGGUFProgressStore() *GGUFProgressStore {
	return &GGUFProgressStore{
		sessions: make(map[string]*GGUFProgress),
	}
}

// Create registers a new conversion session and returns the progress pointer.
func (s *GGUFProgressStore) Create(sessionID, taskID, jobID, quant string) *GGUFProgress {
	p := &GGUFProgress{
		TaskID:    taskID,
		JobID:     jobID,
		Quant:     quant,
		Status:    GGUFStatusRunning,
		Percent:   0,
		Step:      "Starting…",
		createdAt: time.Now(),
		updatedAt: time.Now(),
	}

	s.mu.Lock()
	s.sessions[sessionID] = p
	s.mu.Unlock()

	return p
}

// Get retrieves the progress for a session ID. Returns nil if not found.
func (s *GGUFProgressStore) Get(sessionID string) *GGUFProgress {
	s.mu.RLock()
	p := s.sessions[sessionID]
	s.mu.RUnlock()

	return p
}

// Update updates the progress for a session.
func (s *GGUFProgressStore) Update(sessionID string, fn func(*GGUFProgress)) {
	s.mu.Lock()
	if p, ok := s.sessions[sessionID]; ok {
		fn(p)
		p.updatedAt = time.Now()
	}
	s.mu.Unlock()
}

// Delete removes a session from the store.
func (s *GGUFProgressStore) Delete(sessionID string) {
	s.mu.Lock()
	delete(s.sessions, sessionID)
	s.mu.Unlock()
}

// Cleanup removes sessions older than the given duration.
func (s *GGUFProgressStore) Cleanup(maxAge time.Duration) {
	cutoff := time.Now().Add(-maxAge)

	s.mu.Lock()
	for id, p := range s.sessions {
		if p.updatedAt.Before(cutoff) {
			delete(s.sessions, id)
		}
	}
	s.mu.Unlock()
}

// EventPercent maps a Python converter event type to an approximate percent
// and step description. Exported for testability.
func EventPercent(eventType string) (int, string) {
	switch eventType {
	case "disk_space_check":
		return 1, "Checking free disk space"
	case "conversion_started":
		return 2, "Starting conversion"
	case "cache_hit", "cache_hit_after_wait":
		return 100, "Served from cache"
	case "merge_started":
		return 5, "Merging LoRA adapter"
	case "merge_completed":
		return 15, "Merge complete"
	case "base_model_download_started":
		return 5, "Downloading base model"
	case "base_model_download_completed":
		return 15, "Base model downloaded"
	case "convert_hf_to_gguf_started":
		return 20, "Converting to GGUF"
	case "loading_safetensors":
		return 30, "Loading model weights"
	case "tokenizer_warning":
		return -1, "" // no percent change.
	case "convert_hf_to_gguf_completed":
		return 55, "GGUF conversion complete"
	case "quantize_started":
		return 60, "Quantizing model"
	case "quantize_tensor_fallback":
		return -1, "" // sub-step, no change.
	case "quantize_completed":
		return 90, "Quantization complete"
	case "emitted":
		return 95, "Finalizing output"
	case "verify_started":
		return 97, "Verifying GGUF"
	case "verify_ok", "verify_ok_header_only":
		return 100, "Verification complete"
	case "verify_warning":
		return -1, "" // keep current.
	case "lock_wait":
		return -1, "Waiting for another conversion to finish"
	case "done":
		return 100, "Complete"
	case "error":
		return -1, "" // error handled separately by the async runner.
	default:
		return -1, ""
	}
}
