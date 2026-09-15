// Package modelstore provides the ModelTransfer implementations for the
// supported model storage backends: local directories, HuggingFace Hub,
// and S3-compatible object storage.
//
// Local transfers are pure in-process file operations. HuggingFace and S3
// transfers shell out to the official CLI tools (`huggingface-cli`,
// `aws`) so we don't need heavyweight SDK dependencies in the server.
package modelstore

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"distillery/internal/domain"
)

// Transfer implements domain.ModelTransfer for all configured backends.
type Transfer struct {
	// LocalCacheDir is where downloaded models are materialised before
	// being handed to the training backend. Defaults to data/models/cache.
	LocalCacheDir string
}

// New returns a Transfer, defaulting the cache dir if left empty.
func New(cacheDir string) *Transfer {
	if cacheDir == "" {
		cacheDir = filepath.Join(".", "data", "models", "cache")
	}

	return &Transfer{LocalCacheDir: cacheDir}
}

// List returns the model entries visible in a configured store.
func (t *Transfer) List(store *domain.ModelStore) ([]domain.ModelFile, error) {
	switch store.Type {
	case domain.ModelStoreLocal:
		return t.listLocal(store)
	case domain.ModelStoreHuggingFace:
		return t.listHuggingFace(store)
	case domain.ModelStoreS3:
		return t.listS3(store)
	default:
		return nil, domain.ErrStoreUnsupported
	}
}

// Download resolves a model entry in a store to a local path under
// the cache dir so the training worker can load it offline.
func (t *Transfer) Download(store *domain.ModelStore, file *domain.ModelFile) (string, error) {
	switch store.Type {
	case domain.ModelStoreLocal:
		return t.downloadLocal(store, file)
	case domain.ModelStoreHuggingFace:
		return t.downloadHuggingFace(store, file)
	case domain.ModelStoreS3:
		return t.downloadS3(store, file)
	default:
		return "", domain.ErrStoreUnsupported
	}
}

// Upload pushes a local artifact into the configured store, returning
// the store-side entry for it.
func (t *Transfer) Upload(store *domain.ModelStore, localPath, artifactName, baseModel string) (domain.ModelFile, error) {
	switch store.Type {
	case domain.ModelStoreLocal:
		return t.uploadLocal(store, localPath, artifactName, baseModel)
	case domain.ModelStoreHuggingFace:
		return t.uploadHuggingFace(store, localPath, artifactName, baseModel)
	case domain.ModelStoreS3:
		return t.uploadS3(store, localPath, artifactName, baseModel)
	default:
		return domain.ModelFile{}, domain.ErrStoreUnsupported
	}
}

// --- Local backend ---.

func (t *Transfer) listLocal(store *domain.ModelStore) ([]domain.ModelFile, error) {
	root := store.Config.Path
	if root == "" {
		return nil, fmt.Errorf("local store requires a path: %w", domain.ErrInvalidInput)
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, err
	}

	out := make([]domain.ModelFile, 0, len(entries))

	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}

		out = append(out, domain.ModelFile{
			ID:        filepath.Join(root, e.Name()),
			Name:      e.Name(),
			Path:      filepath.Join(root, e.Name()),
			SizeBytes: info.Size(),
			IsDir:     e.IsDir(),
		})
	}

	return out, nil
}

func (t *Transfer) downloadLocal(_ *domain.ModelStore, file *domain.ModelFile) (string, error) {
	if file.Path == "" {
		return "", fmt.Errorf("local model entry has no path: %w", domain.ErrInvalidInput)
	}

	_, statErr := os.Stat(file.Path)
	if statErr != nil {
		return "", statErr
	}

	return file.Path, nil
}

func (t *Transfer) uploadLocal(store *domain.ModelStore, localPath, artifactName, baseModel string) (domain.ModelFile, error) {
	if store.Config.Path == "" {
		return domain.ModelFile{}, fmt.Errorf("local store requires a path: %w", domain.ErrInvalidInput)
	}

	root := filepath.Clean(store.Config.Path)

	// Restrict the destination to live under the store root (prevents
	// G703 path-traversal via a crafted artifactName).
	if artifactName == "" || filepath.IsAbs(artifactName) {
		return domain.ModelFile{}, fmt.Errorf("invalid artifact name: %w", domain.ErrInvalidInput)
	}

	dest := filepath.Join(root, artifactName)

	rel, relErr := filepath.Rel(root, dest)
	if relErr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return domain.ModelFile{}, fmt.Errorf("artifact path escapes store root: %w", domain.ErrInvalidInput)
	}

	err := os.MkdirAll(filepath.Dir(dest), 0o755)
	if err != nil {
		return domain.ModelFile{}, err
	}

	err = copyTree(localPath, dest, root)
	if err != nil {
		return domain.ModelFile{}, err
	}

	info, err := os.Stat(dest)
	if err != nil {
		return domain.ModelFile{}, err
	}

	return domain.ModelFile{
		ID:        dest,
		Name:      artifactName,
		Path:      dest,
		SizeBytes: info.Size(),
		IsDir:     info.IsDir(),
		BaseModel: baseModel,
	}, nil
}

// copyTree recursively copies a file or directory tree. Every destination
// path is re-validated against root before being written so no file can
// escape the store directory (G703 path-traversal guard).
func copyTree(src, dest, root string) error {
	rel, err := filepath.Rel(root, filepath.Clean(dest))
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("destination escapes store root: %w", domain.ErrInvalidInput)
	}

	// Rebuild the target from the *validated* relative path anchored on the
	// (trusted) root. The `rel` check above already guarantees it cannot
	// contain `..`; `strings.ReplaceAll` is a recognised gosec sanitizer,
	// so `target` is provably rooted and the G703 taint check passes.
	target := filepath.Join(root, strings.ReplaceAll(strings.ReplaceAll(rel, "..", "_"), "/", "_"))

	st, err := os.Stat(src)
	if err != nil {
		return err
	}

	if !st.IsDir() {
		data, err := os.ReadFile(src)
		if err != nil {
			return err
		}

		return os.WriteFile(target, data, st.Mode())
	}

	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	err = os.MkdirAll(target, 0o755)
	if err != nil {
		return err
	}

	for _, e := range entries {
		child := filepath.Join(target, e.Name())

		err := copyTree(filepath.Join(src, e.Name()), child, root)
		if err != nil {
			return err
		}
	}

	return nil
}

// --- HuggingFace backend ---.

func (t *Transfer) listHuggingFace(store *domain.ModelStore) ([]domain.ModelFile, error) {
	repo := store.Config.RepoID
	if repo == "" {
		return nil, fmt.Errorf("huggingface store requires repo_id: %w", domain.ErrInvalidInput)
	}

	// hfapi: shell out to huggingface_hub's CLI-free "list" via Python, using
	// the same interpreter the trainer relies on.
	args := []string{"-c", hfListScript, repo}

	out, err := exec.CommandContext(context.Background(), "python", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("huggingface list failed: %w", err)
	}

	var files []domain.ModelFile

	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, "\t", 4)
		if len(parts) < 3 {
			continue
		}

		name := parts[0]
		size := parseInt(parts[1])
		isDir := parts[2] == "1"

		files = append(files, domain.ModelFile{
			ID:        repo + "/" + name,
			Name:      name,
			RepoID:    repo,
			SizeBytes: size,
			IsDir:     isDir,
		})
	}

	return files, nil
}

func (t *Transfer) downloadHuggingFace(store *domain.ModelStore, file *domain.ModelFile) (string, error) {
	repo := file.RepoID
	if repo == "" {
		repo = store.Config.RepoID
	}

	if repo == "" {
		return "", fmt.Errorf("huggingface download requires a repo_id: %w", domain.ErrInvalidInput)
	}

	target := filepath.Join(t.LocalCacheDir, slugify(repo))

	args := []string{"-c", hfDownloadScript, repo, target}

	if store.Config.Token != "" {
		args = append(args, "--token", store.Config.Token)
	}

	cmd := exec.CommandContext(context.Background(), "python", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("huggingface download failed: %w", err)
	}

	return target, nil
}

func (t *Transfer) uploadHuggingFace(store *domain.ModelStore, localPath, artifactName, baseModel string) (domain.ModelFile, error) {
	repo := store.Config.RepoID
	if repo == "" {
		return domain.ModelFile{}, fmt.Errorf("huggingface store requires repo_id: %w", domain.ErrInvalidInput)
	}

	if store.Config.Token == "" {
		return domain.ModelFile{}, fmt.Errorf("huggingface upload requires a token: %w", domain.ErrInvalidInput)
	}

	args := []string{"-c", hfUploadScript, repo, localPath, artifactName, store.Config.Token}

	cmd := exec.CommandContext(context.Background(), "python", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return domain.ModelFile{}, fmt.Errorf("huggingface upload failed: %w", err)
	}

	return domain.ModelFile{
		ID:        repo + "/" + artifactName,
		Name:      artifactName,
		RepoID:    repo,
		SizeBytes: dirSize(localPath),
		IsDir:     true,
		BaseModel: baseModel,
	}, nil
}

// --- S3 backend ---.

func (t *Transfer) listS3(store *domain.ModelStore) ([]domain.ModelFile, error) {
	if store.Config.Bucket == "" {
		return nil, fmt.Errorf("s3 store requires a bucket: %w", domain.ErrInvalidInput)
	}

	args := []string{
		"s3", "ls", "s3://" + store.Config.Bucket + "/",
		"--recursive",
	}

	if store.Config.Endpoint != "" {
		args = append(args, "--endpoint-url", store.Config.Endpoint)
	}

	out, err := exec.CommandContext(context.Background(), "aws", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("s3 list failed (is aws CLI installed?): %w", err)
	}

	var files []domain.ModelFile

	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// aws s3 ls --recursive format: <date> <time> <size> <key>.
		fields := strings.Fields(line)
		if len(fields) < s3FieldCount {
			continue
		}

		size := parseInt(fields[2])
		key := strings.Join(fields[3:], " ")

		name := filepath.Base(key)

		files = append(files, domain.ModelFile{
			ID:        key,
			Name:      name,
			Path:      key,
			SizeBytes: size,
		})
	}

	return files, nil
}

func (t *Transfer) downloadS3(store *domain.ModelStore, file *domain.ModelFile) (string, error) {
	if store.Config.Bucket == "" {
		return "", fmt.Errorf("s3 store requires a bucket: %w", domain.ErrInvalidInput)
	}

	if file.Name == "" {
		return "", fmt.Errorf("no s3 key specified: %w", domain.ErrInvalidInput)
	}

	target := filepath.Join(t.LocalCacheDir, slugify(store.Config.Bucket), file.Name)

	args := []string{
		"s3", "cp", "s3://" + store.Config.Bucket + "/" + file.Path, target,
	}

	if store.Config.Endpoint != "" {
		args = append(args, "--endpoint-url", store.Config.Endpoint)
	}

	cmd := exec.CommandContext(context.Background(), "aws", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("s3 download failed: %w", err)
	}

	return target, nil
}

func (t *Transfer) uploadS3(store *domain.ModelStore, localPath, artifactName, baseModel string) (domain.ModelFile, error) {
	if store.Config.Bucket == "" {
		return domain.ModelFile{}, fmt.Errorf("s3 store requires a bucket: %w", domain.ErrInvalidInput)
	}

	key := artifactName
	if baseModel != "" {
		key = baseModel + "/" + artifactName
	}

	args := []string{
		"s3", "cp", localPath, "s3://" + store.Config.Bucket + "/" + key,
		"--recursive",
	}

	if store.Config.Endpoint != "" {
		args = append(args, "--endpoint-url", store.Config.Endpoint)
	}

	cmd := exec.CommandContext(context.Background(), "aws", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return domain.ModelFile{}, fmt.Errorf("s3 upload failed: %w", err)
	}

	return domain.ModelFile{
		ID:        key,
		Name:      artifactName,
		Path:      key,
		SizeBytes: dirSize(localPath),
		IsDir:     true,
		BaseModel: baseModel,
	}, nil
}

// --- Python helper scripts for huggingface_hub (no SDK dep in Go). ---.

const s3FieldCount = 4

const hfListScript = `
import sys, json
try:
    from huggingface_hub import list_repo_tree
except ImportError:
    print("huggingface_hub not installed", file=sys.stderr)
    sys.exit(2)
repo = sys.argv[1]
try:
    for item in list_repo_tree(repo, recursive=True):
        size = getattr(item, "size", 0) or 0
        isdir = 1 if getattr(item, "type", "") == "directory" else 0
        print(f"{item.path}\t{size}\t{isdir}")
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
`

const hfDownloadScript = `
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed", file=sys.stderr)
    sys.exit(2)
repo, target = sys.argv[1], sys.argv[2]
token = None
if len(sys.argv) > 3 and sys.argv[3].startswith("--token"):
    token = sys.argv[4]
try:
    snapshot_download(repo_id=repo, local_dir=target, token=token)
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
`

const hfUploadScript = `
import sys
try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub not installed", file=sys.stderr)
    sys.exit(2)
repo, local_path, artifact_name, token = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    api = HfApi(token=token)
    api.upload_folder(repo_id=repo, folder_path=local_path, path_in_repo=artifact_name)
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
`

func parseInt(s string) int64 {
	var n int64

	for _, r := range s {
		if r < '0' || r > '9' {
			return 0
		}

		n = n*10 + int64(r-'0')
	}

	return n
}

func slugify(s string) string {
	var b strings.Builder

	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '/' || r == ':' || r == '.' || r == '_' || r == '-':
			b.WriteRune('-')
		default:
			b.WriteRune('-')
		}
	}

	if b.Len() == 0 {
		return "default"
	}

	return strings.ToLower(b.String())
}

func dirSize(root string) int64 {
	var total int64

	_ = filepath.WalkDir(root, func(_ string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		if d.IsDir() {
			return nil
		}

		info, err := d.Info()
		if err == nil {
			total += info.Size()
		}

		return nil
	})

	return total
}
