package usecase

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // register decoders for image.Decode format sniffing.
	_ "image/jpeg"
	_ "image/png"
	"io"
	"path"
	"strconv"
	"strings"
	"time"

	"distillery/internal/domain"
)

// This file implements the vision-language (document AI) dataset importers
// from plan 06: a ZIP of images + a data.jsonl manifest, a single
// image+prompt+answer example, and (via PDFRasterizer) PDF-to-page-images
// import. Every image is decoded and size-checked before being handed to the
// blob store, mirroring the plan's "validate: image decodes, max
// resolution/size" contract.

const (
	// maxImportImageBytes caps one uploaded image (15 MiB) before decode.
	maxImportImageBytes = 15 << 20
	// maxImportImagePixels caps width*height (40 MP -- generous for a
	// high-DPI scanned page) so a decompression-bomb-style image can't blow
	// up memory/VRAM downstream.
	maxImportImagePixels = 40_000_000
	// maxVisionZIPBytes caps an uploaded dataset ZIP (200 MiB).
	maxVisionZIPBytes = 200 << 20
	// maxVisionZIPFiles bounds how many files a ZIP import will read, so a
	// maliciously crafted archive can't force unbounded work.
	maxVisionZIPFiles = 20000
)

// ErrBlobStoreUnavailable is returned by the vision import methods when the
// server wasn't wired with a domain.BlobStore.
var ErrBlobStoreUnavailable = errors.New("vision image storage is not configured on this server")

// ErrPDFRasterizerUnavailable is returned by ImportPDF when the server
// wasn't wired with a domain.PDFRasterizer (e.g. the optional Python/PyMuPDF
// dependency isn't installed).
var ErrPDFRasterizerUnavailable = errors.New("PDF import requires the PDF rasterizer, which is not configured on this server")

// decodedImage is a validated, decoded image ready to be stored.
type decodedImage struct {
	data   []byte
	ext    string
	width  int
	height int
}

// decodeAndValidateImage decodes raw image bytes, rejecting anything that
// fails to decode, exceeds the byte cap, or exceeds the pixel cap.
func decodeAndValidateImage(data []byte) (*decodedImage, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("%w: empty image", domain.ErrInvalidInput)
	}

	if len(data) > maxImportImageBytes {
		return nil, fmt.Errorf("%w: image exceeds %d bytes", domain.ErrInvalidInput, maxImportImageBytes)
	}

	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("%w: image does not decode: %s", domain.ErrInvalidInput, err.Error())
	}

	pixels := cfg.Width * cfg.Height
	if pixels <= 0 || pixels > maxImportImagePixels {
		return nil, fmt.Errorf("%w: image resolution %dx%d exceeds the maximum", domain.ErrInvalidInput, cfg.Width, cfg.Height)
	}

	ext := "." + format
	if format == "jpeg" {
		ext = ".jpg"
	}

	return &decodedImage{data: data, ext: ext, width: cfg.Width, height: cfg.Height}, nil
}

// AddVisionExample adds one user-provided image+prompt(+answer) example to a
// vision_lm task's dataset, storing the image via the blob store.
func (u *DatasetUsecase) AddVisionExample(taskID string, imageData []byte, prompt, answer string) (*domain.DatasetStats, error) {
	if u.blobs == nil {
		return nil, ErrBlobStoreUnavailable
	}

	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return nil, fmt.Errorf("%w: prompt is required", domain.ErrInvalidInput)
	}

	img, err := decodeAndValidateImage(imageData)
	if err != nil {
		return nil, err
	}

	key, err := u.blobs.Put(taskID, img.data, img.ext)
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()

	example := newVisionExample(u.idGen, taskID, key, prompt, strings.TrimSpace(answer), "", 0, now)

	err = u.examples.AddBatch([]*domain.Example{example})
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// visionManifestLine is one record of the ZIP import's data.jsonl manifest.
type visionManifestLine struct {
	Image  string `json:"image"` // path within the ZIP, e.g. "images/inv_0042.png".
	Prompt string `json:"prompt"`
	Answer string `json:"answer"`
	DocID  string `json:"doc_id,omitempty"`
	Page   int    `json:"page,omitempty"`
}

// ImportVisionZIP bulk-loads a ZIP archive containing page images plus a
// data.jsonl manifest (one JSON object per line): {"image": "images/x.png",
// "prompt": "...", "answer": "..."}. Every manifest line's image is decoded,
// size-checked and stored via the blob store; lines whose image is missing
// or fails validation are skipped (partial success, like the other bulk
// importers).
func (u *DatasetUsecase) ImportVisionZIP(taskID string, zipData []byte) (*domain.DatasetStats, error) {
	if u.blobs == nil {
		return nil, ErrBlobStoreUnavailable
	}

	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	if len(zipData) == 0 || len(zipData) > maxVisionZIPBytes {
		return nil, fmt.Errorf("%w: zip must be non-empty and under %d bytes", domain.ErrInvalidInput, maxVisionZIPBytes)
	}

	zr, err := zip.NewReader(bytes.NewReader(zipData), int64(len(zipData)))
	if err != nil {
		return nil, fmt.Errorf("%w: not a valid zip archive", domain.ErrInvalidInput)
	}

	if len(zr.File) > maxVisionZIPFiles {
		return nil, fmt.Errorf("%w: zip contains too many files", domain.ErrInvalidInput)
	}

	files, manifest := indexVisionZIPFiles(zr.File)
	if manifest == nil {
		return nil, fmt.Errorf("%w: zip must contain a data.jsonl manifest", domain.ErrInvalidInput)
	}

	rc, err := manifest.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	batch, err := u.readVisionManifest(taskID, rc, files)
	if err != nil {
		return nil, err
	}

	if len(batch) == 0 {
		return nil, fmt.Errorf("%w: no valid image+prompt records found in the manifest", domain.ErrInvalidInput)
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// indexVisionZIPFiles indexes a vision ZIP archive's non-directory entries
// by their cleaned path, and locates the top-level data.jsonl manifest
// entry (nil if none is present).
func indexVisionZIPFiles(zipFiles []*zip.File) (files map[string]*zip.File, manifest *zip.File) {
	files = make(map[string]*zip.File, len(zipFiles))

	for _, f := range zipFiles {
		if f.FileInfo().IsDir() {
			continue
		}

		clean := path.Clean(strings.TrimPrefix(f.Name, "/"))
		files[clean] = f

		if strings.EqualFold(path.Base(clean), "data.jsonl") {
			manifest = f
		}
	}

	return files, manifest
}

// readVisionManifest scans a vision ZIP's data.jsonl manifest line by line,
// decoding, validating and storing each referenced image via the blob
// store. Lines whose image is missing or fails validation are skipped
// (partial success, like the other bulk importers).
func (u *DatasetUsecase) readVisionManifest(taskID string, rc io.Reader, files map[string]*zip.File) (batch []*domain.Example, err error) {
	now := time.Now().UTC()

	scanner := bufio.NewScanner(rc)
	scanner.Buffer(make([]byte, 128*1024), 1<<20)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		ex := u.visionExampleFromLine(taskID, line, files, now)
		if ex != nil {
			batch = append(batch, ex)
		}
	}

	err = scanner.Err()
	if err != nil {
		return nil, domain.ErrInvalidInput
	}

	return batch, nil
}

// visionExampleFromLine decodes a single data.jsonl manifest line, resolves
// and validates its referenced image within the ZIP's file index, stores
// the image via the blob store, and builds the resulting example. It
// returns nil (no error) for any line that is malformed or references an
// image that is missing or fails validation.
func (u *DatasetUsecase) visionExampleFromLine(taskID, line string, files map[string]*zip.File, now time.Time) *domain.Example {
	var rec visionManifestLine

	err := json.Unmarshal([]byte(line), &rec)
	if err != nil || strings.TrimSpace(rec.Image) == "" || strings.TrimSpace(rec.Prompt) == "" {
		return nil
	}

	imgPath := path.Clean(strings.TrimPrefix(rec.Image, "/"))

	zf, ok := files[imgPath]
	if !ok {
		return nil
	}

	imgRC, err := zf.Open()
	if err != nil {
		return nil
	}

	data, err := io.ReadAll(io.LimitReader(imgRC, maxImportImageBytes+1))
	imgRC.Close()

	if err != nil {
		return nil
	}

	img, err := decodeAndValidateImage(data)
	if err != nil {
		return nil
	}

	key, err := u.blobs.Put(taskID, img.data, img.ext)
	if err != nil {
		return nil
	}

	return newVisionExample(u.idGen, taskID, key, strings.TrimSpace(rec.Prompt), strings.TrimSpace(rec.Answer), rec.DocID, rec.Page, now)
}

// ImportPDF rasterizes an uploaded PDF into one image example per page (all
// sharing the same prompt, per the plan's "one example per page, with an
// optional doc_id for split-by-document"). Answers are left empty for human
// correction (or a later LLM pre-fill pass), matching how NER treats
// unlabeled text as a valid neutral sample.
func (u *DatasetUsecase) ImportPDF(taskID string, pdfData []byte, prompt string, maxDPI int) (*domain.DatasetStats, error) {
	if u.blobs == nil {
		return nil, ErrBlobStoreUnavailable
	}

	if u.pdfRasterizer == nil {
		return nil, ErrPDFRasterizerUnavailable
	}

	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return nil, fmt.Errorf("%w: prompt is required", domain.ErrInvalidInput)
	}

	pages, err := u.pdfRasterizer.Rasterize(pdfData, maxDPI)
	if err != nil {
		return nil, err
	}

	if len(pages) == 0 {
		return nil, fmt.Errorf("%w: PDF produced no pages", domain.ErrInvalidInput)
	}

	docID := "doc_" + strconv.FormatInt(time.Now().UTC().UnixNano(), 36)
	now := time.Now().UTC()

	var batch []*domain.Example

	for _, p := range pages {
		img, err := decodeAndValidateImage(p.PNG)
		if err != nil {
			continue
		}

		key, err := u.blobs.Put(taskID, img.data, img.ext)
		if err != nil {
			continue
		}

		batch = append(batch, newVisionExample(u.idGen, taskID, key, prompt, "", docID, p.Page, now))
	}

	if len(batch) == 0 {
		return nil, fmt.Errorf("%w: no page produced a valid image", domain.ErrInvalidInput)
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// newVisionExample builds a typed Example with the vision payload JSON.
func newVisionExample(idGen IDGenerator, taskID, imageKey, prompt, answer, docID string, page int, now time.Time) *domain.Example {
	payload, _ := json.Marshal(domain.VisionPayload{
		Image:  imageKey,
		Prompt: prompt,
		Answer: answer,
		DocID:  docID,
		Page:   page,
	})

	return &domain.Example{
		ID:        idGen.NewID("ex"),
		TaskID:    taskID,
		Payload:   payload,
		Source:    domain.SourceUser,
		CreatedAt: now, Flagged: false, FlagNote: "", Duplicate: false, Input: "", Output: "",
	}
}

// GetBlob reads back a previously stored image (or rasterized PDF page) by
// key, for the dataset gallery / test-tab drag-drop preview to render. It
// returns the raw bytes plus a best-effort content type inferred from the
// key's extension (blob keys always carry the extension LocalStore stored
// them with, e.g. "a1b2c3.png").
func (u *DatasetUsecase) GetBlob(taskID, key string) (data []byte, contentType string, err error) {
	if u.blobs == nil {
		return nil, "", ErrBlobStoreUnavailable
	}

	_, err = u.tasks.Get(taskID)
	if err != nil {
		return nil, "", err
	}

	data, err = u.blobs.Get(taskID, key)
	if err != nil {
		return nil, "", err
	}

	return data, blobContentType(key), nil
}

// blobContentType maps a blob key's extension to a MIME type. Blob keys are
// only ever written by decodeAndValidateImage (image/png, image/jpeg,
// image/gif) or the PDF rasterizer (always PNG pages), so a small fixed map
// covers every key this store will ever hold.
func blobContentType(key string) string {
	switch {
	case strings.HasSuffix(key, ".png"):
		return "image/png"
	case strings.HasSuffix(key, ".jpg"), strings.HasSuffix(key, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(key, ".gif"):
		return "image/gif"
	default:
		return "application/octet-stream"
	}
}

// SweepExpiredBlobs deletes vision_lm task images (and the examples that
// reference them) past each task's retention window. It is meant to be
// called periodically (see cmd/server's retention ticker); a single sweep
// touches every vision_lm task once. Errors for one task don't stop the
// sweep for the others; the last error (if any) is returned for logging.
func (u *DatasetUsecase) SweepExpiredBlobs() error {
	if u.blobs == nil {
		return nil
	}

	tasks, err := u.tasks.List()
	if err != nil {
		return err
	}

	now := time.Now().UTC().Unix()

	var lastErr error

	for _, t := range tasks {
		if t.Kind != domain.KindVisionLM {
			continue
		}

		retentionDays := t.RetentionDays
		if retentionDays == 0 {
			retentionDays = domain.DefaultImageRetentionDays
		}

		err := u.sweepTaskBlobs(t.ID, retentionDays, now)
		if err != nil {
			lastErr = err
		}
	}

	return lastErr
}

func (u *DatasetUsecase) sweepTaskBlobs(taskID string, retentionDays int, nowUnix int64) error {
	keys, err := u.blobs.ListKeys(taskID)
	if err != nil {
		return err
	}

	if len(keys) == 0 {
		return nil
	}

	expired := make(map[string]bool, len(keys))

	for _, k := range keys {
		createdAt, ok := u.blobs.Stat(taskID, k)
		if !ok {
			continue
		}

		if domain.RetentionDeadlinePassed(createdAt.Unix(), nowUnix, retentionDays) {
			expired[k] = true

			_ = u.blobs.Delete(taskID, k)
		}
	}

	if len(expired) == 0 {
		return nil
	}

	examples, err := u.examples.ListByTask(taskID)
	if err != nil {
		return err
	}

	for _, e := range examples {
		imageKey, _, _, ok := visionPayloadFields(e)
		if !ok || !expired[imageKey] {
			continue
		}

		_ = u.examples.Delete(taskID, e.ID)
	}

	return nil
}
