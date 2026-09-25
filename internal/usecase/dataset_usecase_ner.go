package usecase

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"distillery/internal/domain"
)

// This file implements the NER (token_classifier) dataset importers from
// plan 04: JSONL spans, CoNLL (token-per-line, BIO tags), and CSV with span
// columns. Every importer validates char offsets (bounds, end>start), label
// membership against the task's LabelSet (when set), and rejects overlapping
// spans, mirroring the plan's "validate offsets, overlaps and label set".

// maxNERLine caps one imported line/document (10 MiB) to bound memory.
const maxNERLine = 10 << 20

// nerJSONLPayload is the accepted JSONL span record (spans may be listed
// under "entities", "spans", or "labels").
type nerJSONLPayload struct {
	Text  string              `json:"text"`
	Spans []domain.EntitySpan `json:"entities"`
	Alt   []domain.EntitySpan `json:"spans"`
	Alt2  []domain.EntitySpan `json:"labels"`
}

// ImportNERJSONL bulk-loads span-annotated JSONL records:
//
//	{"text": "Acme paid $4,200 on 3 May.", "entities": [{"start":0,"end":4,"label":"ORG"}, ...]}
//
// Lines that fail validation are skipped; at least one valid line is required.
func (u *DatasetUsecase) ImportNERJSONL(taskID, content string) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(make([]byte, 128*1024), maxNERLine)

	var batch []*domain.Example

	now := time.Now().UTC()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var p nerJSONLPayload

		err := json.Unmarshal([]byte(line), &p)
		if err != nil {
			continue
		}

		spans := p.Spans
		if len(spans) == 0 {
			spans = p.Alt
		}

		if len(spans) == 0 {
			spans = p.Alt2
		}

		err = u.validateNERSpans(task, p.Text, spans)
		if err != nil {
			continue
		}

		batch = append(batch, u.newNERExample(taskID, p.Text, spans, now))
	}

	scanErr := scanner.Err()
	if scanErr != nil {
		return nil, domain.ErrInvalidInput
	}

	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}

	if err := u.examples.AddBatch(batch); err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// conllSentence is one parsed CoNLL sentence: whitespace-joined tokens with
// the char offsets they occupied in the reconstructed text.
type conllSentence struct {
	text  string
	spans []domain.EntitySpan
}

// ImportCoNLL parses classic token-per-line, BIO-tagged CoNLL content:
//
//	Acme      B-ORG
//	paid      O
//	$         B-AMOUNT
//	4,200     I-AMOUNT
//	...
//
// Sentences are separated by blank lines. Tokens are whitespace-joined with
// single spaces; spans are derived from B-/I- tag runs and re-emitted as
// character offsets over the reconstructed sentence text. BILOU tags
// (B-/I-/L-/U-) are accepted for compatibility.
func (u *DatasetUsecase) ImportCoNLL(taskID, content string) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	sentences, err := parseCoNLL(content)
	if err != nil {
		return nil, err
	}

	if len(sentences) == 0 {
		return nil, domain.ErrInvalidInput
	}

	now := time.Now().UTC()

	var batch []*domain.Example

	for _, s := range sentences {
		err := u.validateNERSpans(task, s.text, s.spans)
		if err != nil {
			continue
		}

		batch = append(batch, u.newNERExample(taskID, s.text, s.spans, now))
	}

	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}

	if err := u.examples.AddBatch(batch); err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// parseCoNLL reconstructs sentences and BIO spans. Offsets are computed by
// re-tokenizing on whitespace: token i starts at its char position in the
// joined text (tokens joined with single spaces).
func parseCoNLL(content string) ([]conllSentence, error) {
	var (
		sentences []conllSentence
		cur       conllSentence
		toks      []string
		tags      []string
	)

	flush := func() {
		if len(toks) == 0 {
			return
		}

		cur.text = strings.Join(toks, " ")
		cur.spans = bioTagsToSpans(toks, tags)
		sentences = append(sentences, cur)
		toks, tags, cur = nil, nil, conllSentence{}
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(make([]byte, 128*1024), maxNERLine)

	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r\n")
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			flush()
			continue
		}

		// Skip CoNLL-style comment/document header lines.
		if trimmed == "-DOCSTART-" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		fields := strings.Fields(trimmed)

		// The token is the first column, the tag the last column; this
		// tolerates both "token tag" and "token POS chunk tag" layouts.
		if len(fields) < 2 {
			// A bare token with no tag column: treat as "O".
			toks = append(toks, fields[0])
			tags = append(tags, "O")

			continue
		}

		toks = append(toks, fields[0])
		tags = append(tags, fields[len(fields)-1])
	}

	err := scanner.Err()
	if err != nil {
		return nil, err
	}

	flush()

	return sentences, nil
}

// bioTagsToSpans converts BIO/BILOU tag runs into character spans over the
// whitespace-joined token text.
func bioTagsToSpans(tokens, tags []string) []domain.EntitySpan {
	if len(tokens) == 0 {
		return nil
	}

	tokStarts := make([]int, len(tokens))
	tokEnds := make([]int, len(tokens))

	curr := 0
	for i, tok := range tokens {
		tokStarts[i] = curr
		tokEnds[i] = curr + len(tok)
		curr = tokEnds[i] + 1
	}

	spans := make([]domain.EntitySpan, 0, len(tokens)/2+1)

	type openSpan struct {
		label string
		start int
		end   int
	}

	var open *openSpan

	closeSpan := func(end int) {
		if open != nil {
			spans = append(spans, domain.EntitySpan{Start: open.start, End: end, Label: open.label})
			open = nil
		}
	}

	for i := range tokens {
		tag := strings.TrimSpace(tags[i])
		if tag == "" {
			tag = "O"
		}

		prefix := tag
		label := tag

		if len(tag) > 2 && tag[1] == '-' {
			prefix = tag[:1]
			label = tag[2:]
		}

		tokStart := tokStarts[i]
		tokEnd := tokEnds[i]

		switch prefix {
		case "B":
			if open != nil {
				closeSpan(tokEnds[i-1])
			}

			open = &openSpan{label: label, start: tokStart, end: tokEnd}
		case "I":
			if open == nil || open.label != label {
				if open != nil {
					closeSpan(tokEnds[i-1])
				}

				open = &openSpan{label: label, start: tokStart, end: tokEnd}
			} else {
				open.end = tokEnd
			}
		case "L":
			if open != nil && open.label == label {
				closeSpan(tokEnd)
			} else {
				if open != nil {
					closeSpan(tokEnds[i-1])
				}

				spans = append(spans, domain.EntitySpan{Start: tokStart, End: tokEnd, Label: label})
			}
		case "U":
			if open != nil {
				closeSpan(tokEnds[i-1])
			}

			spans = append(spans, domain.EntitySpan{Start: tokStart, End: tokEnd, Label: label})
		default:
			// "O" or other.
			if open != nil {
				closeSpan(tokEnds[i-1])
			}
		}
	}

	if open != nil {
		closeSpan(open.end)
	}

	return spans
}

// ImportNERCSV bulk-loads span annotations from CSV. Two shapes are accepted:
//
//  1. JSON-in-CSV (recommended): columns "text" and "entities" where the
//     entities cell is a JSON array of spans ({"start","end","label"}).
//  2. Flat rows: columns text,start,end,label[,start2,end2,label2 ...] —
//     repeated span triplets per row.
func (u *DatasetUsecase) ImportNERCSV(taskID, content string) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil || len(records) < 2 {
		return nil, domain.ErrInvalidInput
	}

	textCol, entsCol, spanCols, start := detectNERColumns(records[0])

	var batch []*domain.Example

	now := time.Now().UTC()

	for _, rec := range records[start:] {
		if len(rec) == 0 {
			continue
		}

		get := func(idx int) string {
			if idx < 0 || idx >= len(rec) {
				return ""
			}

			return rec[idx]
		}

		text := get(textCol)
		if text == "" {
			continue
		}

		var spans []domain.EntitySpan

		if entsCol >= 0 {
			var raw []domain.EntitySpan

			err := json.Unmarshal([]byte(get(entsCol)), &raw)
			if err != nil {
				continue
			}

			spans = raw
		} else {
			for i := 0; i+2 < len(spanCols); i += 3 {
				s, err1 := strconv.Atoi(strings.TrimSpace(get(spanCols[i])))
				e, err2 := strconv.Atoi(strings.TrimSpace(get(spanCols[i+1])))

				if err1 != nil || err2 != nil {
					continue
				}

				spans = append(spans, domain.EntitySpan{Start: s, End: e, Label: strings.TrimSpace(get(spanCols[i+2]))})
			}
		}

		err := u.validateNERSpans(task, text, spans)
		if err != nil {
			continue
		}

		batch = append(batch, u.newNERExample(taskID, text, spans, now))
	}

	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}

	if err := u.examples.AddBatch(batch); err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// detectNERColumns finds the text column and either the entities (JSON) column
// or the repeating start/end/label column triplets. Returns startRow=1 when a
// recognized header is present.
func detectNERColumns(header []string) (textCol, entsCol int, spanCols []int, startRow int) {
	textCol, entsCol = -1, -1

	for i, col := range header {
		switch strings.ToLower(strings.TrimSpace(col)) {
		case "text", "sentence", "input":
			if textCol < 0 {
				textCol = i
			}
		case "entities", "spans", "labels", "annotations":
			if entsCol < 0 {
				entsCol = i
			}
		case "start", "begin", "offset_start":
			spanCols = append(spanCols, i)
		case "end", "offset_end":
			spanCols = append(spanCols, i)
		case "label", "type", "entity", "tag":
			spanCols = append(spanCols, i)
		}
	}

	if textCol < 0 {
		// Positional fallback: text,entities or text,start,end,label.
		if len(header) >= 2 {
			textCol, entsCol = 0, 1
		} else {
			textCol = 0
		}
	}

	return textCol, entsCol, spanCols, 1
}

// validateNERSpans applies the plan's NER validation contract against the
// task's LabelSet.
func (u *DatasetUsecase) validateNERSpans(task *domain.Task, text string, spans []domain.EntitySpan) error {
	if strings.TrimSpace(text) == "" {
		return errors.New("text is empty")
	}

	return domain.ValidateEntitySpans(text, spans, task.LabelSet)
}

// newNERExample builds a typed Example with the span payload JSON.
func (u *DatasetUsecase) newNERExample(taskID, text string, spans []domain.EntitySpan, now time.Time) *domain.Example {
	payload, _ := json.Marshal(map[string]any{
		"text":     text,
		"entities": spans,
	})

	return &domain.Example{
		ID:        u.idGen.NewID("ex"),
		TaskID:    taskID,
		Payload:   payload,
		Source:    domain.SourceUser,
		CreatedAt: now, Flagged: false, FlagNote: "", Duplicate: false,
	}
}
