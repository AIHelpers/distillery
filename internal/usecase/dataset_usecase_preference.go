package usecase

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"distillery/internal/domain"
	"distillery/internal/usecase/dataset"
)

// maxPreferenceLine caps one imported JSONL line (10 MiB) to bound memory.
const maxPreferenceLine = 10 << 20

// preferenceSchema is the single source of truth for what makes a valid
// {"prompt","chosen","rejected"} payload (non-empty, chosen != rejected,
// length limits) — the same validator the dataset.Registry exposes for
// preference_lm, reused here so the two paths can't drift apart.
var preferenceSchema = &dataset.PreferenceLMSchema{} //nolint:gochecknoglobals // stateless validator.

// preferencePayload is the JSON object shape for a preference_lm example:
// {"prompt": "...", "chosen": "...", "rejected": "..."}.
type preferencePayload struct {
	Prompt   string `json:"prompt"`
	Chosen   string `json:"chosen"`
	Rejected string `json:"rejected"`
}

// AddPreferencePair adds a single (prompt, chosen, rejected) example to a
// task's preference dataset (POST /tasks/{id}/preferences) and re-curates
// the task's preference-kind examples.
//
// A preference pair may live on the same task as its causal_lm SFT dataset
// (fed by the Feedback tab's "use as preference" action) or on a task
// created directly with Kind == preference_lm for an ORPO run with no
// parent SFT job; either way it is tagged Kind == KindPreferenceLM so
// Curate (the causal_lm/other-kind dataset view) excludes it and
// PreferenceStats can find it.
func (u *DatasetUsecase) AddPreferencePair(taskID, prompt, chosen, rejected string) (*domain.PreferencePairStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(preferencePayload{
		Prompt:   strings.TrimSpace(prompt),
		Chosen:   strings.TrimSpace(chosen),
		Rejected: strings.TrimSpace(rejected),
	})
	if err != nil {
		return nil, domain.ErrInvalidInput
	}

	err = preferenceSchema.Validate(payload)
	if err != nil {
		return nil, err
	}

	e := &domain.Example{
		ID:        u.idGen.NewID("ex"),
		TaskID:    taskID,
		Kind:      domain.KindPreferenceLM,
		Payload:   payload,
		Source:    domain.SourceUser,
		CreatedAt: time.Now().UTC(),
	}

	// AddBatch (rather than Add) so a single pair lands in storage the same
	// way a bulk import does, keeping PreferenceStats' immediate re-read
	// consistent regardless of how the pair arrived.
	err = u.examples.AddBatch([]*domain.Example{e})
	if err != nil {
		return nil, err
	}

	return u.PreferenceStats(taskID)
}

// ImportPreferenceJSONL bulk-loads {"prompt","chosen","rejected"} JSONL
// records (POST /tasks/{id}/preferences/import). Lines that fail validation
// are skipped; at least one valid line is required.
func (u *DatasetUsecase) ImportPreferenceJSONL(taskID, content string) (*domain.PreferencePairStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(make([]byte, 128*1024), maxPreferenceLine)

	now := time.Now().UTC()

	var batch []*domain.Example

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var p preferencePayload

		err := json.Unmarshal([]byte(line), &p)
		if err != nil {
			continue
		}

		payload, err := json.Marshal(preferencePayload{
			Prompt:   strings.TrimSpace(p.Prompt),
			Chosen:   strings.TrimSpace(p.Chosen),
			Rejected: strings.TrimSpace(p.Rejected),
		})
		if err != nil {
			continue
		}

		if preferenceSchema.Validate(payload) != nil {
			continue
		}

		batch = append(batch, &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Kind:      domain.KindPreferenceLM,
			Payload:   payload,
			Source:    domain.SourceUser,
			CreatedAt: now,
		})
	}

	err = scanner.Err()
	if err != nil {
		return nil, domain.ErrInvalidInput
	}

	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return nil, err
	}

	return u.PreferenceStats(taskID)
}

// PreferenceStats dedupes and quality-flags a task's preference-kind
// examples (independently of any other-kind examples sharing the task) and
// reports the plan's length-bias diagnostic: the average chosen/rejected
// completion length and the fraction of pairs where chosen is the longer
// one, so a UI can warn when the model risks learning "longer == better"
// rather than the intended preference.
func (u *DatasetUsecase) PreferenceStats(taskID string) (*domain.PreferencePairStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	all, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	stats := &domain.PreferencePairStats{TaskID: taskID}
	seen := make(map[string]bool)

	var (
		chosenLenSum, rejectedLenSum float64
		chosenLonger, usable         int
	)

	for _, e := range all {
		if e.Kind != domain.KindPreferenceLM {
			continue
		}

		key := curationKey(e)
		e.Duplicate = seen[key]
		seen[key] = true

		flagged, note := qualityFlag(e)
		e.Flagged = flagged
		e.FlagNote = note

		_ = u.examples.Update(e)

		stats.Total++

		if e.Duplicate {
			stats.Duplicates++
		}

		if e.Flagged {
			stats.Flagged++
		}

		if e.Duplicate || e.Flagged {
			continue
		}

		_, chosen, rejected, ok := preferencePayloadFields(e)
		if !ok {
			continue
		}

		usable++
		chosenLenSum += float64(len(chosen))
		rejectedLenSum += float64(len(rejected))

		if len(chosen) > len(rejected) {
			chosenLonger++
		}
	}

	stats.UsableCount = usable

	if usable > 0 {
		stats.AvgChosenLen = round4(chosenLenSum / float64(usable))
		stats.AvgRejectedLen = round4(rejectedLenSum / float64(usable))
		stats.ChosenLongerPct = round4(float64(chosenLonger) / float64(usable))
		stats.LengthBiasWarning = stats.ChosenLongerPct >= domain.LengthBiasThreshold
	}

	stats.ReadyToTrain, stats.ReadinessReason = preferenceReadiness(stats)

	return stats, nil
}

// preferenceReadiness applies the plan's risk #3 guidance: DPO/ORPO needs
// at least a handful of pairs to run at all, and is flagged (not blocked)
// as unstable below ~200 pairs.
func preferenceReadiness(stats *domain.PreferencePairStats) (ready bool, reason string) {
	if stats.UsableCount < 3 {
		return false, "need at least 3 usable (non-duplicate, non-flagged) preference pairs"
	}

	if stats.UsableCount < domain.MinPreferencePairs {
		return true, fmt.Sprintf(
			"dataset is usable but has fewer than %d pairs; DPO/ORPO results can be unstable at this size",
			domain.MinPreferencePairs,
		)
	}

	return true, ""
}

// round4 rounds to 4 decimal places, matching the precision other stats
// (e.g. retrieval's NDCG/MRR) are reported at.
func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
