package domain

// VisionConfig carries the vision-language fine-tuning hyperparameters that
// the Go layer can pass to the trainer worker (mirrored by the Python
// VisionConfig in trainer/tasks/vision.py).
type VisionConfig struct {
	// MaxImageSide caps the longest edge of an image (pixels) fed to the
	// processor; larger images are downscaled before tokenization. 0 = the
	// trainer's own default (dynamic resolution).
	MaxImageSide int `json:"max_image_side,omitempty"`
	// MaxNewTokens caps generation length for both eval and inference.
	MaxNewTokens int `json:"max_new_tokens,omitempty"`
	// FreezeVisionEncoder keeps the vision tower frozen and trains LoRA only
	// on the language tower + projector (default true; considerably cheaper
	// and the recommended default per the plan).
	FreezeVisionEncoder *bool `json:"freeze_vision_encoder,omitempty"`
	// Epochs / LearningRate / BatchSize override the defaults.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
	// GradientAccumulationSteps trades step time for the small per-device
	// batch sizes VRAM-constrained VLM training requires.
	GradientAccumulationSteps int `json:"gradient_accumulation_steps,omitempty"`
}

// FreezeVision reports the effective freeze-vision-encoder setting,
// defaulting to true (freeze) when unset, per the plan's recommendation.
func (c *VisionConfig) FreezeVision() bool {
	if c == nil || c.FreezeVisionEncoder == nil {
		return true
	}

	return *c.FreezeVisionEncoder
}

// VisionPayload is the JSON shape of a vision_lm training example:
//
//	{"image": "<blob key>", "prompt": "Extract vendor, total and date as JSON.",
//	 "answer": "{\"vendor\":\"Acme\",\"total\":4200.00,\"date\":\"2026-05-03\"}"}
//
// Image references a key returned by the task's BlobStore (never raw bytes),
// so examples stay small and the trainer/importer materialize images from
// the blob directory by key.
type VisionPayload struct {
	// Image is the blob store key for the page image (PNG/JPEG).
	Image string `json:"image"`
	// Prompt is the instruction/question posed against the image.
	Prompt string `json:"prompt"`
	// Answer is the expected model output — free text, or JSON text when the
	// task declares a JSON schema.
	Answer string `json:"answer"`
	// DocID optionally groups multi-page examples from the same source
	// document so a curation/eval pass can split or aggregate by document.
	DocID string `json:"doc_id,omitempty"`
	// Page is the 1-based page number within DocID (0 = not applicable).
	Page int `json:"page,omitempty"`
}

// VisionPrediction is the structured result of a vision_lm inference call.
type VisionPrediction struct {
	// Text is the raw generated text.
	Text string `json:"text"`
	// JSON is the parsed value of Text when it is valid JSON, nil otherwise.
	JSON any `json:"json,omitempty"`
	// JSONValid reports whether Text parsed as JSON (and, when the task
	// declares a schema, satisfied it).
	JSONValid bool `json:"json_valid"`
}

// VisionInferenceEngine serves vision-language predictions from a deployed
// fine-tuned model. Implementations must be kind-safe: callers guard on
// Deployment.Kind == KindVisionLM.
type VisionInferenceEngine interface {
	// PredictImage runs the deployed model against one image (raw bytes,
	// already validated/decoded) and a prompt, returning the generated text
	// (and, when it parses, structured JSON).
	PredictImage(
		job *TrainingJob,
		trainingExamples []*Example,
		imageBytes []byte,
		prompt string,
	) (VisionPrediction, error)
}

// --- Field-level evaluation (invoice/receipt-style extraction) ---.

// FieldMetric is the per-field extraction quality for one JSON key
// (e.g. "vendor", "total", "date") aggregated over the held-out split.
type FieldMetric struct {
	ExactMatch float64 `json:"exact_match"`
	F1         float64 `json:"f1"`
	Support    int     `json:"support"`
}

// RasterizedPage is one page image produced by a PDFRasterizer.
type RasterizedPage struct {
	Page   int    // 1-based page number.
	PNG    []byte // page raster, PNG-encoded.
	Width  int
	Height int
}

// PDFRasterizer converts a PDF into one PNG image per page (bounded DPI), so
// each page can be stored as a normal vision_lm image example. Implemented
// by shelling out to a small Python helper (trainer/pdf_rasterize.py),
// mirroring how the local trainer/GGUF exporter shell out to Python workers.
type PDFRasterizer interface {
	Rasterize(pdfBytes []byte, maxDPI int) ([]RasterizedPage, error)
}

// --- Retention policy ---.

// DefaultImageRetentionDays is applied when a vision_lm task does not set
// RetentionDays explicitly. Images often contain PII/financial data, so a
// bounded default (rather than "forever") is the safer default.
const DefaultImageRetentionDays = 90

// RetentionDeadlinePassed reports whether an image/blob created at
// createdAtUnix has exceeded the given retention window (days). A
// non-positive retentionDays means "keep forever" (retention disabled).
func RetentionDeadlinePassed(createdAtUnix, nowUnix int64, retentionDays int) bool {
	if retentionDays <= 0 {
		return false
	}

	const secondsPerDay = 24 * 60 * 60

	return nowUnix-createdAtUnix > int64(retentionDays)*secondsPerDay
}
