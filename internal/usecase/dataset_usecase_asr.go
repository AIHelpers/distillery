package usecase

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"strings"
	"time"

	"distillery/internal/domain"
)

// This file implements the speech-to-text (ASR) dataset importers from plan
// 07: a ZIP of audio files plus a manifest.jsonl manifest, and a single
// audio+transcript example. Audio is sniffed (container headers) and
// byte-capped before being handed to the blob store, mirroring the plan's
// "validation: decodable audio, duration limits, non-empty transcript"
// contract. The trainer resamples everything to 16 kHz mono via ffmpeg and
// recomputes true durations during feature extraction.

const (
	// maxASRAudioBytes caps one uploaded audio clip (50 MiB) before sniffing.
	maxASRAudioBytes = 50 << 20
	// maxASRZIPBytes caps an uploaded dataset ZIP (500 MiB).
	maxASRZIPBytes = 500 << 20
	// maxASRZIPFiles bounds how many files a ZIP import will read.
	maxASRZIPFiles = 20000
)

// asrAudioExt maps an uploaded audio file's extension to the normalized
// extension used for storage. The original container is preserved so the
// served content type stays correct; ffmpeg handles decode/resample later.
func asrAudioExt(name string) (string, bool) {
	lower := strings.ToLower(name)

	for _, ext := range []string{".wav", ".wave", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"} {
		if strings.HasSuffix(lower, ext) {
			if ext == ".wave" {
				return ".wav", true
			}

			return ext, true
		}
	}

	return "", false
}

// sniffAudio validates raw audio bytes: non-empty, within the byte cap, and
// carrying a plausible container header (RIFF/WAVE, ID3/MPEG sync, fLaC).
// For m4a/ogg/opus/webm only a plausible non-empty payload is required;
// the trainer's ffmpeg decode step is the authoritative validation.
func sniffAudio(data []byte, ext string) error {
	if len(data) == 0 {
		return fmt.Errorf("%w: empty audio", domain.ErrInvalidInput)
	}

	if len(data) > maxASRAudioBytes {
		return fmt.Errorf("%w: audio exceeds %d bytes", domain.ErrInvalidInput, maxASRAudioBytes)
	}

	switch ext {
	case ".wav":
		if len(data) < 12 || !bytes.Equal(data[:4], []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WAVE")) {
			return fmt.Errorf("%w: not a valid WAV file", domain.ErrInvalidInput)
		}
	case ".mp3":
		if len(data) < 3 {
			return fmt.Errorf("%w: audio too short to decode", domain.ErrInvalidInput)
		}

		if !(bytes.Equal(data[:3], []byte("ID3")) || (data[0] == 0xFF && data[1]&0xE0 == 0xE0)) {
			return fmt.Errorf("%w: not a valid MP3 file", domain.ErrInvalidInput)
		}
	case ".flac":
		if len(data) < 4 || !bytes.Equal(data[:4], []byte("fLaC")) {
			return fmt.Errorf("%w: not a valid FLAC file", domain.ErrInvalidInput)
		}
	case ".m4a", ".ogg", ".opus", ".webm":
		if len(data) < 64 {
			return fmt.Errorf("%w: audio too short to decode", domain.ErrInvalidInput)
		}
	default:
		return fmt.Errorf("%w: unsupported audio format", domain.ErrInvalidInput)
	}

	return nil
}

// AddASRAudioExample adds one user-provided audio+transcript example to an
// asr task's dataset, storing the audio via the blob store. speaker is
// optional (used for split-by-speaker evaluation).
func (u *DatasetUsecase) AddASRAudioExample(taskID string, audioData []byte, filename, text, speaker string) (*domain.DatasetStats, error) {
	if u.blobs == nil {
		return nil, ErrBlobStoreUnavailable
	}

	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("%w: transcript text is required", domain.ErrInvalidInput)
	}

	ext, ok := asrAudioExt(filename)
	if !ok {
		return nil, fmt.Errorf("%w: unsupported audio format %q (expected wav/mp3/flac/m4a)", domain.ErrInvalidInput, filename)
	}

	if err := sniffAudio(audioData, ext); err != nil {
		return nil, err
	}

	key, err := u.blobs.Put(taskID, audioData, ext)
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()

	example := newASRExample(u.idGen, taskID, key, text, strings.TrimSpace(speaker), 0, now)

	err = u.examples.AddBatch([]*domain.Example{example})
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// asrManifestLine is one record of the ZIP import's manifest.jsonl, per the
// plan's data format:
//
//	{"audio": "audio/call_0193.wav", "text": "Please reset my Acme router...", "speaker": "agent_04"}
type asrManifestLine struct {
	Audio    string  `json:"audio"` // path within the ZIP, e.g. "audio/call_0193.wav".
	Text     string  `json:"text"`
	Speaker  string  `json:"speaker,omitempty"`
	Duration float64 `json:"duration,omitempty"` // seconds; optional hint, recomputed by the trainer.
}

// ImportASRZIP bulk-loads a ZIP archive containing audio files plus a
// manifest.jsonl (one JSON object per line). Every manifest line's audio is
// sniffed and stored via the blob store; lines whose audio is missing or
// fails validation are skipped (partial success, like the other bulk
// importers).
func (u *DatasetUsecase) ImportASRZIP(taskID string, zipData []byte) (*domain.DatasetStats, error) {
	if u.blobs == nil {
		return nil, ErrBlobStoreUnavailable
	}

	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	if len(zipData) == 0 || len(zipData) > maxASRZIPBytes {
		return nil, fmt.Errorf("%w: zip must be non-empty and under %d bytes", domain.ErrInvalidInput, maxASRZIPBytes)
	}

	zr, err := zip.NewReader(bytes.NewReader(zipData), int64(len(zipData)))
	if err != nil {
		return nil, fmt.Errorf("%w: not a valid zip archive", domain.ErrInvalidInput)
	}

	if len(zr.File) > maxASRZIPFiles {
		return nil, fmt.Errorf("%w: zip contains too many files", domain.ErrInvalidInput)
	}

	files, manifest := indexASRZIPFiles(zr.File)
	if manifest == nil {
		return nil, fmt.Errorf("%w: zip must contain a manifest.jsonl (or data.jsonl) manifest", domain.ErrInvalidInput)
	}

	rc, err := manifest.Open()
	if err != nil {
		return nil, err
	}

	defer rc.Close()

	batch, err := u.readASRManifest(taskID, rc, files)
	if err != nil {
		return nil, err
	}

	if len(batch) == 0 {
		return nil, fmt.Errorf("%w: no valid audio+transcript records found in the manifest", domain.ErrInvalidInput)
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// indexASRZIPFiles indexes an ASR ZIP archive's non-directory entries by
// their cleaned path, and locates the manifest.jsonl (or data.jsonl
// fallback) manifest entry (nil if none is present).
func indexASRZIPFiles(zipFiles []*zip.File) (files map[string]*zip.File, manifest *zip.File) {
	files = make(map[string]*zip.File, len(zipFiles))

	for _, f := range zipFiles {
		if f.FileInfo().IsDir() {
			continue
		}

		clean := path.Clean(strings.TrimPrefix(f.Name, "/"))
		files[clean] = f

		if strings.EqualFold(path.Base(clean), "manifest.jsonl") {
			manifest = f
		}
	}

	if manifest != nil {
		return files, manifest
	}

	// data.jsonl fallback for parity with the vision importer's manifest name.
	for _, f := range zipFiles {
		if !f.FileInfo().IsDir() && strings.EqualFold(path.Base(path.Clean(strings.TrimPrefix(f.Name, "/"))), "data.jsonl") {
			manifest = f

			break
		}
	}

	return files, manifest
}

// readASRManifest scans an ASR ZIP's manifest.jsonl line by line, sniffing
// and storing each referenced audio via the blob store. Lines whose audio is
// missing or fails validation are skipped (partial success).
func (u *DatasetUsecase) readASRManifest(taskID string, rc io.Reader, files map[string]*zip.File) (batch []*domain.Example, err error) {
	now := time.Now().UTC()

	scanner := bufio.NewScanner(rc)
	scanner.Buffer(make([]byte, 128*1024), 4<<20)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if ex := u.asrExampleFromLine(taskID, line, files, now); ex != nil {
			batch = append(batch, ex)
		}
	}

	if scanErr := scanner.Err(); scanErr != nil {
		return nil, domain.ErrInvalidInput
	}

	return batch, nil
}

// asrExampleFromLine decodes a single manifest.jsonl line, resolves and
// sniffs its referenced audio within the ZIP's file index, stores the audio
// via the blob store, and builds the resulting example. It returns nil for
// any line that is malformed or references audio that is missing or fails
// validation (partial success semantics).
func (u *DatasetUsecase) asrExampleFromLine(taskID, line string, files map[string]*zip.File, now time.Time) *domain.Example {
	var rec asrManifestLine

	if json.Unmarshal([]byte(line), &rec) != nil {
		return nil
	}

	if strings.TrimSpace(rec.Audio) == "" || strings.TrimSpace(rec.Text) == "" {
		return nil
	}

	audioPath := path.Clean(strings.TrimPrefix(rec.Audio, "/"))

	zf, ok := files[audioPath]
	if !ok {
		return nil
	}

	audioRC, err := zf.Open()
	if err != nil {
		return nil
	}

	data, err := io.ReadAll(io.LimitReader(audioRC, maxASRAudioBytes+1))
	audioRC.Close()

	if err != nil || len(data) > maxASRAudioBytes {
		return nil
	}

	ext, ok := asrAudioExt(audioPath)
	if !ok {
		return nil
	}

	if sniffAudio(data, ext) != nil {
		return nil
	}

	key, err := u.blobs.Put(taskID, data, ext)
	if err != nil {
		return nil
	}

	return newASRExample(u.idGen, taskID, key, strings.TrimSpace(rec.Text), strings.TrimSpace(rec.Speaker), rec.Duration, now)
}

// newASRExample builds a typed Example with the ASR payload JSON. The
// example's Kind is set so curation and the trainer feed only asr rows to
// the ASR worker.
func newASRExample(idGen IDGenerator, taskID, audioKey, text, speaker string, duration float64, now time.Time) *domain.Example {
	payload, _ := json.Marshal(domain.ASRPayload{
		Audio:    audioKey,
		Text:     text,
		Speaker:  speaker,
		Duration: duration,
	})

	return &domain.Example{
		ID: idGen.NewID("ex"), TaskID: taskID, Kind: domain.KindASR,
		Payload: payload, Source: domain.SourceUser,
		CreatedAt: now, Flagged: false, FlagNote: "", Duplicate: false, Input: "", Output: "",
	}
}

// asrPayloadFields extracts audio/text/speaker from an asr payload. It
// returns ok=false for other payload shapes (distinguished by requiring a
// non-empty "audio" field, which no other kind's payload uses).
func asrPayloadFields(e *domain.Example) (audio, text, speaker string, ok bool) {
	if len(e.Payload) == 0 {
		return "", "", "", false
	}

	var p domain.ASRPayload

	if json.Unmarshal(e.Payload, &p) != nil || strings.TrimSpace(p.Audio) == "" {
		return "", "", "", false
	}

	return strings.TrimSpace(p.Audio), strings.TrimSpace(p.Text), strings.TrimSpace(p.Speaker), true
}

// asrDurationBounds are the clip-duration histogram bucket upper bounds
// (seconds): [0,5), [5,10), [10,15), [15,20), [20,25), [25,30), [30,45),
// [45,60), [60,120), [120,600), then 600+.
var asrDurationBounds = []float64{5, 10, 15, 20, 25, 30, 45, 60, 120, 600}

// ASRAudioStats is the audio-side dataset summary reported by the ASR
// dataset tab. Durations come from the payload hints; the trainer
// recomputes true durations during its feature pass and overwrites them.
type ASRAudioStats struct {
	// Clips is the total number of asr examples in the task.
	Clips int `json:"clips"`
	// TotalSeconds is the sum of known clip durations.
	TotalSeconds float64 `json:"total_seconds"`
	// KnownDurationClips is how many clips carry a duration payload.
	KnownDurationClips int `json:"known_duration_clips"`
	// AverageSeconds is the mean known duration (0 when none known).
	AverageSeconds float64 `json:"average_seconds"`
	// EmptyTranscripts counts clips with no transcript text.
	EmptyTranscripts int `json:"empty_transcripts"`
	// DurationHistogram buckets clips by duration; the final bucket is
	// the 600s+ overflow bucket.
	DurationHistogram []domain.DurationBucket `json:"duration_histogram"`
}

// ASRDatasetStats computes the audio-specific stats for an asr task:
// total hours and a clip-duration histogram (from payload duration hints).
func (u *DatasetUsecase) ASRDatasetStats(taskID string) (*ASRAudioStats, error) {
	if _, err := u.tasks.Get(taskID); err != nil {
		return nil, err
	}

	examples, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	stats := &ASRAudioStats{
		DurationHistogram: newASRDurationHistogram(),
	}

	for _, e := range examples {
		var p domain.ASRPayload

		if json.Unmarshal(e.Payload, &p) != nil || strings.TrimSpace(p.Audio) == "" {
			continue
		}

		stats.Clips++

		if strings.TrimSpace(p.Text) == "" {
			stats.EmptyTranscripts++
		}

		if p.Duration > 0 {
			stats.KnownDurationClips++
			stats.TotalSeconds += p.Duration
		}

		stats.DurationHistogram[asrDurationBucket(p.Duration)].Count++
	}

	if stats.KnownDurationClips > 0 {
		stats.AverageSeconds = stats.TotalSeconds / float64(stats.KnownDurationClips)
	}

	return stats, nil
}

// newASRDurationHistogram builds the empty bucket list matching
// asrDurationBounds (plus the 600s+ overflow bucket).
func newASRDurationHistogram() []domain.DurationBucket {
	hist := make([]domain.DurationBucket, 0, len(asrDurationBounds)+1)

	prev := 0.0
	for _, bound := range asrDurationBounds {
		hist = append(hist, domain.DurationBucket{Start: prev, End: bound})
		prev = bound
	}

	return append(hist, domain.DurationBucket{Start: prev, End: -1}) // -1 encodes "no upper bound".
}

// asrDurationBucket returns the histogram bucket index for a duration; 0
// duration clips land in the first bucket, over-600s clips in the overflow.
func asrDurationBucket(d float64) int {
	for i, bound := range asrDurationBounds {
		if d < bound {
			return i
		}
	}

	return len(asrDurationBounds)
}
