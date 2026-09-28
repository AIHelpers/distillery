package simulation

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strings"

	"distillery/internal/domain"
)

// This file implements domain.ASRInferenceEngine for the simulation backend.
// Like PredictImage, it does not run real trained weights: it estimates the
// uploaded clip's duration from its WAV header (falling back to a 16 kHz
// mono byte-rate estimate), picks the training example whose recorded
// duration is closest, and projects that example's transcript onto the
// audio as timestamped segments — exercising the /transcribe API shape
// (text + segments + language) without real trained weights.

// Transcribe implements domain.ASRInferenceEngine.
func (e *InferenceEngine) Transcribe(
	job *domain.TrainingJob,
	trainingExamples []*domain.Example,
	audioBytes []byte,
	language string,
) (domain.ASRPrediction, error) {
	duration := estimateAudioDuration(audioBytes)

	// Resolve the effective language: request > job ASR config > "en".
	lang := strings.TrimSpace(language)
	if lang == "" && job != nil && job.ASR != nil {
		lang = job.ASR.Language
	}

	if lang == "" {
		lang = "en"
	}

	// Collect usable asr training examples (payload with transcript).
	type candidate struct {
		text     string
		duration float64
	}

	var candidates []candidate

	for _, ex := range trainingExamples {
		var p domain.ASRPayload
		if len(ex.Payload) == 0 || json.Unmarshal(ex.Payload, &p) != nil {
			continue
		}

		if strings.TrimSpace(p.Text) == "" {
			continue
		}

		candidates = append(candidates, candidate{text: p.Text, duration: p.Duration})
	}

	if len(candidates) == 0 {
		return domain.ASRPrediction{
			Text:     "(no training data available)",
			Language: lang,
		}, nil
	}

	// Pick the example whose duration is closest to the clip's; examples
	// without a duration hint rank last (treated as 0 proximity).
	var best candidate

	bestDist := math.MaxFloat64

	for _, c := range candidates {
		dist := math.Abs(c.duration - duration)
		if c.duration <= 0 {
			dist = math.MaxFloat64 / 2 // known-duration examples win.
		}

		if dist < bestDist {
			bestDist = dist
			best = c
		}
	}

	pred := domain.ASRPrediction{
		Text:     best.text,
		Language: lang,
		Segments: projectSegments(best.text, duration),
	}

	return pred, nil
}

// projectSegments splits a transcript into sentence-ish chunks and spreads
// them evenly across the clip duration, mirroring faster-whisper's segment
// output shape for long files. When the duration is unknown the whole
// transcript becomes one segment starting at 0.
func projectSegments(text string, duration float64) []domain.ASRSegment {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if duration <= 0 {
		return []domain.ASRSegment{{Start: 0, End: 0, Text: text}}
	}

	chunks := splitTranscript(text)
	if len(chunks) == 0 {
		return nil
	}

	per := duration / float64(len(chunks))

	segments := make([]domain.ASRSegment, 0, len(chunks))

	cursor := 0.0
	for _, c := range chunks {
		end := cursor + per

		if end > duration {
			end = duration
		}

		segments = append(segments, domain.ASRSegment{
			Start: Round2(cursor),
			End:   Round2(end),
			Text:  c,
		})

		cursor = end
	}

	return segments
}

// splitTranscript breaks a transcript into sentence-ish chunks (on ". ", "! ",
// "? "), falling back to comma splits, then to a single chunk.
func splitTranscript(text string) []string {
	if s := strings.Split(text, ". "); len(s) > 1 {
		return trimAll(s)
	}

	if s := strings.Split(text, ", "); len(s) > 1 {
		return trimAll(s)
	}

	return []string{text}
}

func trimAll(parts []string) []string {
	out := make([]string, 0, len(parts))

	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}

	return out
}

// estimateAudioDuration reads a WAV (RIFF) header to compute the clip's
// duration in seconds. For non-WAV payloads it falls back to assuming
// 16 kHz 16-bit mono (Whisper's normalized import format) and estimating
// from the byte count. Returns 0 when the size is unusable.
func estimateAudioDuration(audioBytes []byte) float64 {
	if len(audioBytes) == 0 {
		return 0
	}

	if d, ok := wavDuration(audioBytes); ok {
		return d
	}

	// 16000 samples/s * 2 bytes/sample = 32000 bytes/s.
	return float64(len(audioBytes)) / 32000.0
}

// wavDuration parses the fmt/data chunks of a RIFF WAVE header. Returns
// ok=false for anything that isn't a well-formed PCM/float WAV.
func wavDuration(b []byte) (float64, bool) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, false
	}

	var (
		sampleRate uint32
		blockAlign uint16
		dataSize   uint32
		haveFmt    bool
		haveData   bool
	)

	off := 12

	for off+8 <= len(b) {
		chunkID := string(b[off : off+4])
		chunkSize := binary.LittleEndian.Uint32(b[off+4 : off+8])
		body := off + 8

		switch chunkID {
		case "fmt ":
			if chunkSize >= 16 && body+16 <= len(b) {
				blockAlign = binary.LittleEndian.Uint16(b[body+12 : body+14])
				sampleRate = binary.LittleEndian.Uint32(b[body+4 : body+8])
				haveFmt = true
			}
		case "data":
			// The declared size can exceed the remaining bytes for streamed
			// files; trust the actual payload length instead.
			dataSize = uint32(len(b) - body)
			haveData = true
		}

		if haveFmt && haveData {
			break
		}

		// Chunks are word-aligned.
		off = body + int(chunkSize) + (int(chunkSize) & 1)
	}

	if !haveFmt || !haveData || sampleRate == 0 || blockAlign == 0 || dataSize == 0 {
		return 0, false
	}

	return float64(dataSize) / (float64(sampleRate) * float64(blockAlign)), true
}
