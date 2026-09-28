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

// errNERTextEmpty is returned when a NER example's text field is blank.
var errNERTextEmpty = errors.New("text is empty")

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

	err = u.examples.AddBatch(batch)
	if err != nil {
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

	err = u.examples.AddBatch(batch)
	if err != nil {
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

	tokStarts, tokEnds := tokenOffsets(tokens)

	b := &bioSpanBuilder{spans: make([]domain.EntitySpan, 0, len(tokens)/2+1)}

	for i := range tokens {
		prefix, label := splitBIOTag(tags[i])
		prevEnd := 0

		if i > 0 {
			prevEnd = tokEnds[i-1]
		}

		b.step(prefix, label, tokStarts[i], tokEnds[i], prevEnd)
	}

	b.closeSpan(b.openEnd())

	return b.spans
}

// tokenOffsets computes the start/end character offsets of each token in
// the whitespace-joined token text.
func tokenOffsets(tokens []string) (starts, ends []int) {
	starts = make([]int, len(tokens))
	ends = make([]int, len(tokens))

	curr := 0
	for i, tok := range tokens {
		starts[i] = curr
		ends[i] = curr + len(tok)
		curr = ends[i] + 1
	}

	return starts, ends
}

// splitBIOTag splits a BIO/BILOU tag such as "B-PERSON" into its prefix
// ("B") and label ("PERSON"). A blank tag is treated as "O", and a tag with
// no "-" separator is returned unsplit (prefix == label == tag).
func splitBIOTag(raw string) (prefix, label string) {
	tag := strings.TrimSpace(raw)
	if tag == "" {
		tag = "O"
	}

	prefix, label = tag, tag

	if len(tag) > 2 && tag[1] == '-' {
		prefix, label = tag[:1], tag[2:]
	}

	return prefix, label
}

// bioSpanBuilder accumulates entity spans while walking a BIO/BILOU tag
// sequence token by token.
type bioSpanBuilder struct {
	spans []domain.EntitySpan
	open  *bioOpenSpan
}

type bioOpenSpan struct {
	label string
	start int
	end   int
}

// openEnd returns the end offset of the currently open span, or 0 if none
// is open (closeSpan is a no-op in that case).
func (b *bioSpanBuilder) openEnd() int {
	if b.open == nil {
		return 0
	}

	return b.open.end
}

// closeSpan finalizes the currently open span (if any) at the given end
// offset and appends it to spans.
func (b *bioSpanBuilder) closeSpan(end int) {
	if b.open != nil {
		b.spans = append(b.spans, domain.EntitySpan{Start: b.open.start, End: end, Label: b.open.label})
		b.open = nil
	}
}

// step processes one token's BIO/BILOU prefix and label, updating the
// currently open span and/or emitting a completed span as needed.
// prevTokEnd is the end offset of the previous token (used to close a
// span that ended before the current token).
func (b *bioSpanBuilder) step(prefix, label string, tokStart, tokEnd, prevTokEnd int) {
	switch prefix {
	case "B":
		b.closeSpan(prevTokEnd)
		b.open = &bioOpenSpan{label: label, start: tokStart, end: tokEnd}
	case "I":
		if b.open == nil || b.open.label != label {
			b.closeSpan(prevTokEnd)
			b.open = &bioOpenSpan{label: label, start: tokStart, end: tokEnd}
		} else {
			b.open.end = tokEnd
		}
	case "L":
		if b.open != nil && b.open.label == label {
			b.closeSpan(tokEnd)
		} else {
			b.closeSpan(prevTokEnd)
			b.spans = append(b.spans, domain.EntitySpan{Start: tokStart, End: tokEnd, Label: label})
		}
	case "U":
		b.closeSpan(prevTokEnd)
		b.spans = append(b.spans, domain.EntitySpan{Start: tokStart, End: tokEnd, Label: label})
	default:
		// "O" or other.
		b.closeSpan(prevTokEnd)
	}
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

	err = u.examples.AddBatch(batch)
	if err != nil {
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
		return errNERTextEmpty
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
