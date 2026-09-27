package domain

import "errors"

var (
	ErrNotFound       = errors.New("not found")
	ErrInvalidInput   = errors.New("invalid input")
	ErrNotReady       = errors.New("dataset not ready for training")
	ErrAlreadyRunning = errors.New("a training job is already running for this task")
	ErrNoDeployment   = errors.New("no active deployment for this task")
	ErrNoModel        = errors.New("no completed training job for this task")
	ErrUnauthorized   = errors.New("missing or invalid API key")
	// ErrRegressionFailed is returned when a preference-tuned (DPO/ORPO) job
	// regressed on its parent SFT job's eval metric beyond the allowed
	// tolerance and deploy was not forced.
	ErrRegressionFailed = errors.New("preference-tuned model regressed on the parent's eval metric")
)
