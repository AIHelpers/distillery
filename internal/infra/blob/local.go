// Package blob provides the local-disk implementation of domain.BlobStore —
// the minimal "C12 blob storage" foundation the vision-language plan depends
// on. A future S3-compatible implementation can be added behind the same
// interface (mirroring how internal/infra/modelstore layers local/HF/S3
// behind domain.ModelTransfer) without touching any caller.
package blob

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"distillery/internal/domain"
)

// maxBlobBytes bounds a single stored blob (25 MiB) so a runaway upload
// can't exhaust disk; the HTTP layer enforces its own (smaller) per-request
// caps before data reaches here, but this is the last line of defense.
const maxBlobBytes = 25 << 20

// ErrBlobTooLarge is returned when Put is given more than maxBlobBytes.
var ErrBlobTooLarge = errors.New("blob exceeds the maximum stored size")

// LocalStore implements domain.BlobStore against a local directory, sharded
// one subdirectory per task: {Root}/{taskID}/{key}{ext}.
type LocalStore struct {
	// Root is the base directory blobs are stored under. Defaults to
	// ./data/blobs when empty.
	Root string
}

// NewLocalStore creates a LocalStore, defaulting Root when empty.
func NewLocalStore(root string) *LocalStore {
	if root == "" {
		root = filepath.Join(".", "data", "blobs")
	}

	return &LocalStore{Root: root}
}

// isSafeID reports whether id is a plain identifier usable as a single path
// component (no separators, no "..", non-empty).
func isSafeID(id string) bool {
	if id == "" || id == "." || id == ".." {
		return false
	}

	return !strings.ContainsAny(id, "/\\")
}

// Put implements domain.BlobStore.
func (s *LocalStore) Put(taskID string, data []byte, ext string) (string, error) {
	if len(data) > maxBlobBytes {
		return "", ErrBlobTooLarge
	}

	dir, err := s.taskDir(taskID)
	if err != nil {
		return "", err
	}

	err = os.MkdirAll(dir, 0o755)
	if err != nil {
		return "", err
	}

	ext = sanitizeExt(ext)
	key := generateKey() + ext

	// filepath.Base neutralizes the (already-validated) key/ext for gosec's
	// taint tracking on the subprocess-adjacent path below.
	path := filepath.Join(dir, filepath.Base(key))

	err = os.WriteFile(path, data, 0o600)
	if err != nil {
		return "", err
	}

	return key, nil
}

// Get implements domain.BlobStore.
func (s *LocalStore) Get(taskID, key string) ([]byte, error) {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return nil, err
	}

	if !isSafeID(key) {
		return nil, fmt.Errorf("%w: invalid blob key", domain.ErrInvalidInput)
	}

	data, err := os.ReadFile(filepath.Join(dir, key))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, domain.ErrNotFound
		}

		return nil, err
	}

	return data, nil
}

// Delete implements domain.BlobStore.
func (s *LocalStore) Delete(taskID, key string) error {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return err
	}

	if !isSafeID(key) {
		return fmt.Errorf("%w: invalid blob key", domain.ErrInvalidInput)
	}

	err = os.Remove(filepath.Join(dir, key))
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}

// DeleteTask implements domain.BlobStore.
func (s *LocalStore) DeleteTask(taskID string) error {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return err
	}

	return os.RemoveAll(dir)
}

// Dir implements domain.BlobStore.
func (s *LocalStore) Dir(taskID string) (string, bool) {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return "", false
	}

	return dir, true
}

// Stat implements domain.BlobStore.
func (s *LocalStore) Stat(taskID, key string) (time.Time, bool) {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return time.Time{}, false
	}

	if !isSafeID(key) {
		return time.Time{}, false
	}

	info, err := os.Stat(filepath.Join(dir, key))
	if err != nil {
		return time.Time{}, false
	}

	return info.ModTime(), true
}

// ListKeys implements domain.BlobStore.
func (s *LocalStore) ListKeys(taskID string) ([]string, error) {
	dir, err := s.taskDir(taskID)
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, err
	}

	keys := make([]string, 0, len(entries))

	for _, e := range entries {
		if !e.IsDir() {
			keys = append(keys, e.Name())
		}
	}

	return keys, nil
}

// taskDir returns (and does not create) the directory for a task's blobs.
// It rejects task IDs that could escape Root the same way local_trainer.go
// neutralizes job IDs, since taskID ultimately reaches filepath.Join from
// HTTP-request-derived values.
func (s *LocalStore) taskDir(taskID string) (string, error) {
	if !isSafeID(taskID) {
		return "", fmt.Errorf("%w: invalid task id", domain.ErrInvalidInput)
	}

	return filepath.Join(s.Root, taskID), nil
}

// generateKey returns a random hex identifier for a new blob.
func generateKey() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)

	return hex.EncodeToString(b)
}

// sanitizeExt normalizes a file extension to ".xxx" lowercase, defaulting to
// ".bin" when empty or suspicious (no path separators allowed).
func sanitizeExt(ext string) string {
	ext = strings.ToLower(strings.TrimSpace(ext))
	if ext == "" || strings.ContainsAny(ext, "/\\") || len(ext) > 10 {
		return ".bin"
	}

	if !strings.HasPrefix(ext, ".") {
		ext = "." + ext
	}

	return ext
}
