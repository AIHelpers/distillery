package http_test

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/infra/blob"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

// visionTestServer bundles the httptest server together with the blob store
// backing it, so tests can seed/inspect blobs directly when needed.
type visionTestServer struct {
	srv   *httptest.Server
	blobs domain.BlobStore
}

// newVisionDatasetHandlerTestServer wires a DatasetHandler against real
// in-memory/on-disk infra (mirroring how cmd/server wires it), seeded with
// one vision_lm task. Mirrors the pattern in agent_handler_test.go.
func newVisionDatasetHandlerTestServer(t *testing.T) *visionTestServer {
	t.Helper()

	store := memory.NewStore("")
	tasks := memory.NewTaskRepo(store)
	examples := memory.NewExampleRepo(store)
	blobs := blob.NewLocalStore(t.TempDir())

	err := tasks.Create(&domain.Task{ID: "task_vis", Name: "doc-ai", Kind: domain.KindVisionLM})
	if err != nil {
		t.Fatalf("seed task: %v", err)
	}

	uc := usecase.NewDatasetUsecase(tasks, examples, nil, usecase.NewRandomIDGenerator()).WithBlobStore(blobs)
	h := deliveryhttp.NewDatasetHandler(uc)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/vision", func(w http.ResponseWriter, r *http.Request) {
		h.AddVisionExample(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/import-vision-zip", func(w http.ResponseWriter, r *http.Request) {
		h.ImportVisionZIP(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/import-vision-pdf", func(w http.ResponseWriter, r *http.Request) {
		h.ImportVisionPDF(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/blobs/{key}", func(w http.ResponseWriter, r *http.Request) {
		h.GetBlob(w, r, r.PathValue("taskID"), r.PathValue("key"))
	})

	return &visionTestServer{srv: httptest.NewServer(mux), blobs: blobs}
}

// validPNG returns a minimal 1x1 PNG, for handler tests that need real
// image bytes rather than a base64 image_base64 string.
func validPNG(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})

	var buf bytes.Buffer

	err := png.Encode(&buf, img)
	if err != nil {
		t.Fatalf("encode png: %v", err)
	}

	return buf.Bytes()
}

func TestDatasetHandler_AddVisionExample_Success(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	body, err := json.Marshal(map[string]string{
		"image_base64": base64.StdEncoding.EncodeToString(validPNG(t)),
		"prompt":       "Extract vendor and total as JSON.",
		"answer":       `{"vendor":"Acme","total":"42"}`,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/vision", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var stats domain.DatasetStats

	err = json.NewDecoder(resp.Body).Decode(&stats)
	if err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if stats.Total != 1 {
		t.Errorf("expected total=1, got %d", stats.Total)
	}
}

func TestDatasetHandler_AddVisionExample_InvalidBase64(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	body, err := json.Marshal(map[string]string{
		"image_base64": "not-valid-base64!!!",
		"prompt":       "Extract vendor.",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/vision", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestDatasetHandler_AddVisionExample_MalformedJSON(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/vision", "application/json", bytes.NewReader([]byte("{not json")))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

// buildVisionZIP builds a minimal ZIP with a data.jsonl manifest plus one
// referenced image, matching the format ImportVisionZIP expects.
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

func TestDatasetHandler_ImportVisionZIP_Success(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	manifest := []string{
		`{"image":"images/page1.png","prompt":"Extract vendor and total as JSON.","answer":"{\"vendor\":\"Acme\",\"total\":\"42\"}"}`,
	}
	zipData := buildVisionZIP(t, validPNG(t), manifest)

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/import-vision-zip", "application/zip", bytes.NewReader(zipData))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var stats domain.DatasetStats

	err = json.NewDecoder(resp.Body).Decode(&stats)
	if err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if stats.Total != 1 {
		t.Errorf("expected total=1, got %d", stats.Total)
	}
}

func TestDatasetHandler_ImportVisionZIP_NotAZip(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/import-vision-zip", "application/zip", bytes.NewReader([]byte("not a zip file")))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestDatasetHandler_ImportVisionZIP_UnknownTask(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	zipData := buildVisionZIP(t, validPNG(t), []string{`{"image":"images/page1.png","prompt":"p"}`})

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/does-not-exist/examples/import-vision-zip", "application/zip", bytes.NewReader(zipData))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", resp.StatusCode)
	}
}

// ImportVisionPDF is wired without a PDF rasterizer in this test server
// (mirrors a server started without one configured), so it should fail
// clearly rather than panic.
func TestDatasetHandler_ImportVisionPDF_RasterizerNotConfigured(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	resp, err := http.Post(ts.srv.URL+"/api/v1/tasks/task_vis/examples/import-vision-pdf?prompt=Extract+fields", "application/pdf", bytes.NewReader([]byte("%PDF-1.4 fake")))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("expected 500 (rasterizer not configured), got %d", resp.StatusCode)
	}
}

func TestDatasetHandler_GetBlob_Success(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	pngBytes := validPNG(t)

	key, err := ts.blobs.Put("task_vis", pngBytes, ".png")
	if err != nil {
		t.Fatalf("seed blob: %v", err)
	}

	resp, err := http.Get(ts.srv.URL + "/api/v1/tasks/task_vis/blobs/" + key)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "image/png" {
		t.Errorf("expected Content-Type image/png, got %q", ct)
	}

	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	if !bytes.Equal(got, pngBytes) {
		t.Error("blob bytes did not round-trip through GetBlob")
	}
}

func TestDatasetHandler_GetBlob_UnknownKey(t *testing.T) {
	t.Parallel()

	ts := newVisionDatasetHandlerTestServer(t)
	defer ts.srv.Close()

	resp, err := http.Get(ts.srv.URL + "/api/v1/tasks/task_vis/blobs/does-not-exist.png")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d", resp.StatusCode)
	}
}
