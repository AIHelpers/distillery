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

// eventPercentStep pairs the approximate percent and step description
// ggufEventPercents maps a Python converter event type to.
type eventPercentStep struct {
	percent int
	step    string
}

// ggufEventPercents maps a Python converter event type to its approximate
// percent and step description. An event not in this map, or one whose
// value carries no percent change (-1), falls through to eventPercentUnset.
var ggufEventPercents = map[string]eventPercentStep{
	"disk_space_check":              {1, "Checking free disk space"},
	"conversion_started":            {2, "Starting conversion"},
	"cache_hit":                     {100, "Served from cache"},
	"cache_hit_after_wait":          {100, "Served from cache"},
	"merge_started":                 {5, "Merging LoRA adapter"},
	"merge_completed":               {15, "Merge complete"},
	"base_model_download_started":   {5, "Downloading base model"},
	"base_model_download_completed": {15, "Base model downloaded"},
	"convert_hf_to_gguf_started":    {20, "Converting to GGUF"},
	"loading_safetensors":           {30, "Loading model weights"},
	"convert_hf_to_gguf_completed":  {55, "GGUF conversion complete"},
	"quantize_started":              {60, "Quantizing model"},
	"quantize_completed":            {90, "Quantization complete"},
	"emitted":                       {95, "Finalizing output"},
	"verify_started":                {97, "Verifying GGUF"},
	"verify_ok":                     {100, "Verification complete"},
	"verify_ok_header_only":         {100, "Verification complete"},
	"lock_wait":                     {-1, "Waiting for another conversion to finish"},
	"done":                          {100, "Complete"},
	// Sub-step / warning / error events carry no percent change; the async
	// runner handles "error" separately.
	"tokenizer_warning":        {-1, ""},
	"quantize_tensor_fallback": {-1, ""},
	"verify_warning":           {-1, ""},
	"error":                    {-1, ""},
}

// EventPercent maps a Python converter event type to an approximate percent
// and step description. Exported for testability.
func EventPercent(eventType string) (percent int, step string) {
	if ps, ok := ggufEventPercents[eventType]; ok {
		return ps.percent, ps.step
	}

	return -1, ""
}
