package domain

import "time"

// BlobStore is the port for storing binary payloads (images, rasterized PDF
// pages) that are too large/unwieldy to keep inline in an Example's JSON
// Payload. Examples reference a blob by key; the trainer and importers
// resolve keys to bytes/paths via this interface. Implementations may back
// onto a local directory or an object store (S3-compatible); the local
// implementation lives in internal/infra/blob.
//
// This is the minimal "C12 blob storage" foundation referenced by the
// vision-language plan: a local-dir implementation is provided; an
// S3-compatible implementation can be added later behind the same interface
// without touching callers.
type BlobStore interface {
	// Put stores data under taskID's namespace and returns a stable key that
	// can later be resolved with Get/Path. ext is the file extension
	// (including the dot, e.g. ".png") to preserve for readability/tools
	// that shell out to Python and expect a real file extension.
	Put(taskID string, data []byte, ext string) (key string, err error)
	// Get reads back the bytes for a previously stored key.
	Get(taskID, key string) ([]byte, error)
	// Delete removes one blob. Deleting a missing key is not an error.
	Delete(taskID, key string) error
	// DeleteTask removes every blob stored for a task (used when a task or
	// its dataset is deleted).
	DeleteTask(taskID string) error
	// Dir returns the local filesystem directory backing taskID's blobs, so
	// the trainer subprocess can be pointed at it directly (images_dir) and
	// resolve payload "image" keys as plain relative file reads without a
	// network round trip per example. Implementations that are not
	// filesystem-backed (a future S3 adapter) return "" and false.
	Dir(taskID string) (path string, ok bool)
	// Stat returns the creation time of a stored blob, used by the retention
	// sweep. ok is false when the key does not exist.
	Stat(taskID, key string) (createdAt time.Time, ok bool)
	// ListKeys lists every blob key currently stored for a task.
	ListKeys(taskID string) ([]string, error)
}
