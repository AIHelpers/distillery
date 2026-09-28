package domain

// ASRConfig carries the speech-to-text fine-tuning hyperparameters the Go
// layer passes to the trainer worker (mirrored by the Python Config in
// trainer/tasks/asr.py).
type ASRConfig struct {
	// Language is the ISO 639-1 tag of the training/serving language
	// (e.g. "en"). Empty = multilingual (the trainer auto-detects).
	Language string `json:"language,omitempty"`
	// Task selects what the fine-tune optimizes for: "transcribe"
	// (same-language speech recognition, the default) or "translate"
	// (X->English translation, Whisper's second supervised head).
	Task string `json:"task,omitempty"`
	// Epochs / LearningRate / BatchSize override the trainer defaults.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
	// GradientAccumulationSteps trades step time for the small per-device
	// batch sizes Whisper's 30-second feature window forces.
	GradientAccumulationSteps int `json:"gradient_accumulation_steps,omitempty"`
	// SpecAugment enables SpecAugment-style time/frequency masking over the
	// input log-mel spectrogram (off by default; helpful for noisy
	// channels once the dataset exceeds a few hours).
	SpecAugment *bool `json:"spec_augment,omitempty"`
	// UseLoRA forces LoRA for tiny/base/small models (the plan's default is
	// full fine-tune for tiny/base/small when VRAM allows, LoRA for
	// medium/large; nil = auto by model size).
	UseLoRA *bool `json:"use_lora,omitempty"`
}

// ASRTaskTranscribe is the fine-tune target for speech recognition.
const ASRTaskTranscribe = "transcribe"

// ASRTaskTranslate is the fine-tune target for speech translation (X->English).
const ASRTaskTranslate = "translate"

// ValidASRTasks lists the supported ASRConfig.Task values.
func ValidASRTasks() []string {
	return []string{ASRTaskTranscribe, ASRTaskTranslate}
}

// IsValidASRTask reports whether t is a supported ASRConfig.Task value.
func IsValidASRTask(t string) bool {
	for _, valid := range ValidASRTasks() {
		if t == valid {
			return true
		}
	}

	return false
}

// ASRConfiguredTask reports the effective task, defaulting to
// "transcribe" when unset (per the plan's default).
func (c *ASRConfig) ConfiguredTask() string {
	if c == nil || c.Task == "" {
		return ASRTaskTranscribe
	}

	return c.Task
}

// SpecAugmentEnabled reports the effective SpecAugment setting (default off).
func (c *ASRConfig) SpecAugmentEnabled() bool {
	if c == nil || c.SpecAugment == nil {
		return false
	}

	return *c.SpecAugment
}

// ASRPayload is the JSON shape of an asr training example:
//
//	{"audio": "<blob key>", "text": "Please reset my Acme router to factory settings.",
//	 "speaker": "agent_04", "duration": 4.2}
//
// Audio references a key returned by the task's BlobStore (never raw
// bytes), so examples stay small and the trainer resolves audio from the
// blob directory by key — the same contract vision_lm images follow.
type ASRPayload struct {
	// Audio is the blob store key for the audio file (wav/mp3/flac/m4a,
	// stored normalized as 16 kHz mono WAV by the importer).
	Audio string `json:"audio"`
	// Text is the reference transcript.
	Text string `json:"text"`
	// Speaker optionally identifies the speaker/recording the segment came
	// from so evaluation can split by speaker (avoiding leakage) and report
	// per-speaker counts.
	Speaker string `json:"speaker,omitempty"`
	// Duration is the segment duration in seconds (computed by the
	// importer; informational for stats and WER weighting).
	Duration float64 `json:"duration,omitempty"`
}

// ASRPrediction is the structured result of an asr inference call.
type ASRPrediction struct {
	// Text is the full transcript.
	Text string `json:"text"`
	// Language is the detected/requested language tag.
	Language string `json:"language,omitempty"`
	// Segments are timestamped transcript chunks for long audio; empty
	// when the engine returns only a flat transcript.
	Segments []ASRSegment `json:"segments,omitempty"`
}

// ASRSegment is one timestamped chunk of a transcript.
type ASRSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// ASRInferenceEngine serves speech-to-text predictions from a deployed
// fine-tuned model. Implementations must be kind-safe: callers guard on
// Deployment.Kind == KindASR.
type ASRInferenceEngine interface {
	// Transcribe runs the deployed model against one audio clip (raw
	// bytes, already validated) and returns the transcript (plus
	// timestamped segments when the engine produces them).
	Transcribe(
		job *TrainingJob,
		trainingExamples []*Example,
		audioBytes []byte,
		language string,
	) (ASRPrediction, error)
}

// --- Metrics ---.

// DefaultAudioRetentionDays is applied when an asr task does not set
// RetentionDays explicitly. Voice is personal/biometric data in many
// jurisdictions (plan 07, risks), so a bounded default is the safer choice;
// 30 days matches common call-center retention policies.
const DefaultAudioRetentionDays = 30

// WhisperSampleRate is the sampling rate all imported audio is normalized
// to (Whisper's own input contract): 16 kHz mono.
const WhisperSampleRate = 16000

// WhisperMaxSegmentSeconds is Whisper's fixed 30-second feature window;
// longer recordings must be chunked to at most this length.
const WhisperMaxSegmentSeconds = 30

// MaxASRAudioSeconds caps a single uploaded audio clip for dataset import
// (10 minutes); longer recordings should be split upstream or imported via
// the ZIP bulk path.
const MaxASRAudioSeconds = 600

// DurationBucket is one bar of the ASR dataset duration histogram reported
// in the task's dataset stats: how many clips fall in [Start, End) seconds.
type DurationBucket struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Count int     `json:"count"`
}
