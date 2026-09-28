package usecase_test

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// fakeBlobStore is an in-memory domain.BlobStore for testing the vision
// dataset usecase without touching disk.
type fakeBlobStore struct {
	data      map[string][]byte // "taskID/key" -> bytes.
	createdAt map[string]time.Time
	putErr    error
	nextKey   int
}

func newFakeBlobStore() *fakeBlobStore {
	return &fakeBlobStore{data: map[string][]byte{}, createdAt: map[string]time.Time{}}
}

func (f *fakeBlobStore) Put(taskID string, data []byte, ext string) (string, error) {
	if f.putErr != nil {
		return "", f.putErr
	}

	f.nextKey++
	// Mirror LocalStore: the key carries its extension, since GetBlob's
	// content-type inference depends on it.
	key := fmt.Sprintf("key%d%s", f.nextKey, ext)
	f.data[f.blobID(taskID, key)] = data
	f.createdAt[f.blobID(taskID, key)] = time.Now().UTC()

	return key, nil
}

func (f *fakeBlobStore) Get(taskID, key string) ([]byte, error) {
	d, ok := f.data[f.blobID(taskID, key)]
	if !ok {
		return nil, domain.ErrNotFound
	}

	return d, nil
}

func (f *fakeBlobStore) Delete(taskID, key string) error {
	delete(f.data, f.blobID(taskID, key))
	delete(f.createdAt, f.blobID(taskID, key))

	return nil
}

func (f *fakeBlobStore) DeleteTask(taskID string) error {
	for id := range f.data {
		if len(id) > len(taskID) && id[:len(taskID)+1] == taskID+"/" {
			delete(f.data, id)
		}
	}

	return nil
}

func (f *fakeBlobStore) Dir(taskID string) (string, bool) { return "/fake/" + taskID, true }

func (f *fakeBlobStore) Stat(taskID, key string) (time.Time, bool) {
	t, ok := f.createdAt[f.blobID(taskID, key)]
	return t, ok
}

func (f *fakeBlobStore) ListKeys(taskID string) ([]string, error) {
	var keys []string

	prefix := taskID + "/"
	for id := range f.data {
		if len(id) > len(prefix) && id[:len(prefix)] == prefix {
			keys = append(keys, id[len(prefix):])
		}
	}

	return keys, nil
}

func (f *fakeBlobStore) blobID(taskID, key string) string { return taskID + "/" + key }

// setCreatedAt backdates a stored blob's creation time (for retention tests).
func (f *fakeBlobStore) setCreatedAt(taskID, key string, t time.Time) {
	f.createdAt[f.blobID(taskID, key)] = t
}

// fakeRasterizer is an in-memory domain.PDFRasterizer for testing.
type fakeRasterizer struct {
	pages []domain.RasterizedPage
	err   error
}

func (f *fakeRasterizer) Rasterize(_ []byte, _ int) ([]domain.RasterizedPage, error) {
	return f.pages, f.err
}

// validPNG returns a tiny valid 4x4 PNG image, for tests that need bytes
// that actually decode.
func validPNG(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 4, 4))

	for y := range 4 {
		for x := range 4 {
			img.Set(x, y, color.RGBA{R: uint8(x * 60), G: uint8(y * 60), B: 100, A: 255}) //nolint:gosec // x,y range over 0-3 above; x*60,y*60 max 180, fits uint8
		}
	}

	var buf bytes.Buffer

	err := png.Encode(&buf, img)
	if err != nil {
		t.Fatalf("encode test png: %v", err)
	}

	return buf.Bytes()
}

func newTestVisionUsecase(task *domain.Task) (*usecase.DatasetUsecase, *mockExampleRepo, *fakeBlobStore) {
	tasks := &mockTaskRepo{task: task}
	examples := &mockExampleRepo{}
	blobs := newFakeBlobStore()
	uc := usecase.NewDatasetUsecase(tasks, examples, &mockSynthGen{}, &mockIDGen{}).WithBlobStore(blobs)

	return uc, examples, blobs
}

func TestAddVisionExample_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, examples, _ := newTestVisionUsecase(task)

	stats, err := uc.AddVisionExample("task_vis", validPNG(t), "Extract vendor as JSON.", `{"vendor":"Acme"}`)
	if err != nil {
		t.Fatalf("AddVisionExample failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 total, got %d", stats.Total)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example added, got %d", len(examples.addBatch))
	}

	var payload domain.VisionPayload

	err = json.Unmarshal(examples.addBatch[0].Payload, &payload)
	if err != nil {
		t.Fatalf("malformed payload: %v", err)
	}

	if payload.Image == "" {
		t.Error("expected a non-empty blob key")
	}

	if payload.Prompt != "Extract vendor as JSON." {
		t.Errorf("unexpected prompt: %q", payload.Prompt)
	}
}

func TestAddVisionExample_RejectsUndecodableImage(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, _ := newTestVisionUsecase(task)

	_, err := uc.AddVisionExample("task_vis", []byte("not an image"), "Extract vendor.", "")
	if err == nil {
		t.Fatal("expected an error for undecodable image data")
	}
}

func TestAddVisionExample_RejectsEmptyPrompt(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, _ := newTestVisionUsecase(task)

	_, err := uc.AddVisionExample("task_vis", validPNG(t), "  ", "")
	if err == nil {
		t.Fatal("expected an error for an empty prompt")
	}
}

func TestAddVisionExample_NoBlobStoreConfigured(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	tasks := &mockTaskRepo{task: task}
	uc := usecase.NewDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{}, &mockIDGen{})

	_, err := uc.AddVisionExample("task_vis", validPNG(t), "Extract vendor.", "")
	if !errors.Is(err, usecase.ErrBlobStoreUnavailable) {
		t.Fatalf("expected ErrBlobStoreUnavailable, got %v", err)
	}
}

// buildVisionZIP builds a minimal ZIP with a data.jsonl manifest plus one
// referenced image, for ImportVisionZIP tests.
func buildVisionZIP(t *testing.T, pngData []byte, manifestLines []string) []byte {
	t.Helper()

	var buf bytes.Buffer

	zw := zip.NewWriter(&buf)

	imgWriter, err := zw.Create("images/page1.png")
	if err != nil {
		t.Fatalf("create zip entry: %v", err)
	}

	_, err = imgWriter.Write(pngData)
	if err != nil {
		t.Fatalf("write zip entry: %v", err)
	}

	manifestWriter, err := zw.Create("data.jsonl")
	if err != nil {
		t.Fatalf("create manifest entry: %v", err)
	}

	for _, line := range manifestLines {
		_, err = manifestWriter.Write([]byte(line + "\n"))
		if err != nil {
			t.Fatalf("write manifest line: %v", err)
		}
	}

	err = zw.Close()
	if err != nil {
		t.Fatalf("close zip writer: %v", err)
	}

	return buf.Bytes()
}

func TestImportVisionZIP_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, examples, _ := newTestVisionUsecase(task)

	manifest := []string{
		`{"image":"images/page1.png","prompt":"Extract vendor and total as JSON.","answer":"{\"vendor\":\"Acme\",\"total\":\"42\"}"}`,
	}
	zipData := buildVisionZIP(t, validPNG(t), manifest)

	stats, err := uc.ImportVisionZIP("task_vis", zipData)
	if err != nil {
		t.Fatalf("ImportVisionZIP failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 total, got %d", stats.Total)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example added, got %d", len(examples.addBatch))
	}
}

func TestImportVisionZIP_SkipsRecordsWithMissingImage(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, examples, _ := newTestVisionUsecase(task)

	manifest := []string{
		`{"image":"images/does-not-exist.png","prompt":"Extract vendor.","answer":""}`,
		`{"image":"images/page1.png","prompt":"Extract vendor.","answer":"Acme"}`,
	}
	zipData := buildVisionZIP(t, validPNG(t), manifest)

	stats, err := uc.ImportVisionZIP("task_vis", zipData)
	if err != nil {
		t.Fatalf("ImportVisionZIP failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 total (missing-image record skipped), got %d", stats.Total)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example added, got %d", len(examples.addBatch))
	}
}

func TestImportVisionZIP_MissingManifest(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, _ := newTestVisionUsecase(task)

	var buf bytes.Buffer

	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("images/page1.png")
	_, _ = w.Write(validPNG(t))
	_ = zw.Close()

	_, err := uc.ImportVisionZIP("task_vis", buf.Bytes())
	if err == nil {
		t.Fatal("expected an error when data.jsonl is missing")
	}
}

func TestImportVisionZIP_NotAZip(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, _ := newTestVisionUsecase(task)

	_, err := uc.ImportVisionZIP("task_vis", []byte("not a zip file"))
	if err == nil {
		t.Fatal("expected an error for a non-zip body")
	}
}

func TestImportPDF_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	tasks := &mockTaskRepo{task: task}
	examples := &mockExampleRepo{}
	blobs := newFakeBlobStore()
	raster := &fakeRasterizer{pages: []domain.RasterizedPage{
		{Page: 1, PNG: validPNG(t), Width: 4, Height: 4},
		{Page: 2, PNG: validPNG(t), Width: 4, Height: 4},
	}}
	uc := usecase.NewDatasetUsecase(tasks, examples, &mockSynthGen{}, &mockIDGen{}).
		WithBlobStore(blobs).
		WithPDFRasterizer(raster)

	stats, err := uc.ImportPDF("task_vis", []byte("%PDF-1.4 fake"), "Extract vendor.", 150)
	if err != nil {
		t.Fatalf("ImportPDF failed: %v", err)
	}

	if stats.Total != 2 {
		t.Fatalf("expected 2 page examples, got %d", stats.Total)
	}

	// Both pages should share one generated doc_id.
	var p1, p2 domain.VisionPayload

	_ = json.Unmarshal(examples.addBatch[0].Payload, &p1)
	_ = json.Unmarshal(examples.addBatch[1].Payload, &p2)

	if p1.DocID == "" || p1.DocID != p2.DocID {
		t.Errorf("expected both pages to share a doc_id, got %q and %q", p1.DocID, p2.DocID)
	}
}

func TestImportPDF_NoRasterizerConfigured(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	tasks := &mockTaskRepo{task: task}
	uc := usecase.NewDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{}, &mockIDGen{}).
		WithBlobStore(newFakeBlobStore())

	_, err := uc.ImportPDF("task_vis", []byte("%PDF-1.4"), "Extract vendor.", 150)
	if !errors.Is(err, usecase.ErrPDFRasterizerUnavailable) {
		t.Fatalf("expected ErrPDFRasterizerUnavailable, got %v", err)
	}
}

func TestSweepExpiredBlobs_DeletesExpiredImagesAndExamples(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, RetentionDays: 30, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	tasks := &mockTaskRepo{list: []*domain.Task{task}, task: task}
	examples := &mockExampleRepo{}
	blobs := newFakeBlobStore()

	uc := usecase.NewDatasetUsecase(tasks, examples, &mockSynthGen{}, &mockIDGen{}).WithBlobStore(blobs)

	// One expired example (created 60 days ago) and one fresh one.
	expiredKey, _ := blobs.Put("task_vis", validPNG(t), ".png")
	blobs.setCreatedAt("task_vis", expiredKey, time.Now().UTC().AddDate(0, 0, -60))

	freshKey, _ := blobs.Put("task_vis", validPNG(t), ".png")

	expiredPayload, _ := json.Marshal(domain.VisionPayload{Image: expiredKey, Prompt: "p"})
	freshPayload, _ := json.Marshal(domain.VisionPayload{Image: freshKey, Prompt: "p"})

	examples.examples = []*domain.Example{
		{ID: "ex_expired", TaskID: "task_vis", Payload: expiredPayload},
		{ID: "ex_fresh", TaskID: "task_vis", Payload: freshPayload},
	}

	err := uc.SweepExpiredBlobs()
	if err != nil {
		t.Fatalf("SweepExpiredBlobs failed: %v", err)
	}

	_, ok := blobs.Stat("task_vis", expiredKey)
	if ok {
		t.Error("expected the expired blob to be deleted")
	}

	_, ok = blobs.Stat("task_vis", freshKey)
	if !ok {
		t.Error("expected the fresh blob to still exist")
	}

	if examples.deletedID != "ex_expired" {
		t.Errorf("expected the expired example to be deleted, got deletedID=%q", examples.deletedID)
	}
}

func TestSweepExpiredBlobs_NoOpWithoutBlobStore(t *testing.T) {
	t.Parallel()

	uc := usecase.NewDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{}, &mockIDGen{})

	err := uc.SweepExpiredBlobs()
	if err != nil {
		t.Fatalf("expected no error when no blob store is configured, got %v", err)
	}
}

func TestGetBlob_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, blobs := newTestVisionUsecase(task)

	key, err := blobs.Put("task_vis", validPNG(t), ".png")
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	data, contentType, err := uc.GetBlob("task_vis", key)
	if err != nil {
		t.Fatalf("GetBlob failed: %v", err)
	}

	if contentType != "image/png" {
		t.Errorf("expected image/png, got %q", contentType)
	}

	if len(data) == 0 {
		t.Error("expected non-empty blob bytes")
	}
}

func TestGetBlob_UnknownKey(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_vis", Kind: domain.KindVisionLM, Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}
	uc, _, _ := newTestVisionUsecase(task)

	_, _, err := uc.GetBlob("task_vis", "does-not-exist.png")
	if err == nil {
		t.Fatal("expected an error for an unknown blob key")
	}
}

func TestGetBlob_NoBlobStoreConfigured(t *testing.T) {
	t.Parallel()

	uc := usecase.NewDatasetUsecase(&mockTaskRepo{task: &domain.Task{ID: "task_vis", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}, &mockExampleRepo{}, &mockSynthGen{}, &mockIDGen{})

	_, _, err := uc.GetBlob("task_vis", "key.png")
	if !errors.Is(err, usecase.ErrBlobStoreUnavailable) {
		t.Fatalf("expected ErrBlobStoreUnavailable, got %v", err)
	}
}

func TestGetBlob_UnknownTask(t *testing.T) {
	t.Parallel()

	uc, _, _ := newTestVisionUsecase(nil)

	_, _, err := uc.GetBlob("does-not-exist", "key.png")
	if err == nil {
		t.Fatal("expected an error for an unknown task")
	}
}
