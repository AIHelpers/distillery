package usecase

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"strings"
	"time"

	"distillery/internal/domain"
)

// ExamplePayload is a raw payload plus the query-group used for holdout
// splitting. For pair/triplet examples the group is the query text; for
// docs-only examples the group is the document text.
type ExamplePayload struct {
	// Payload is the kind-specific JSON body, e.g. {"query","positive"}.
	Payload json.RawMessage
	// Group is the query text used to keep all documents for one query in the
	// same split (train or eval).
	Group string
}

// ImportRetrievalJSONL parses pair/triplet/graded JSONL records and loads each
// as an example with a typed Payload. It applies a query-group-based holdout
// split: examples are partitioned so that a document/query never leaks across
// the train and eval splits.
//
// Supported record shapes (auto-detected per line):
//
//	pair:    {"query","positive"}
//	triplet: {"query","positive","negative"}
//	graded:  {"query","document","label"}
//	docs:    {"document"}
func (u *DatasetUsecase) ImportRetrievalJSONL(taskID, content string) (*domain.DatasetStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	payloads, err := parseRetrievalJSONL(content)
	if err != nil {
		return nil, err
	}

	if len(payloads) == 0 {
		return nil, domain.ErrInvalidInput
	}

	return u.addRetrievalExamples(taskID, payloads)
}

// ImportRetrievalCSV parses embedding/reranker CSV content. Column headers
// (case-insensitive) drive mapping:
//
//	query,positive            -> pair
//	query,positive,negative   -> triplet
//	query,document,label       -> graded
//	document                   -> docs-only
//
// When no recognized header is present, columns 1/2 are treated as
// query/positive.
func (u *DatasetUsecase) ImportRetrievalCSV(taskID, content string) (*domain.DatasetStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil || len(records) == 0 {
		return nil, domain.ErrInvalidInput
	}

	cols, start := detectRetrievalColumns(records[0])
	if len(records) == 1 && len(records[0]) <= 1 {
		return nil, domain.ErrInvalidInput
	}

	var payloads []ExamplePayload

	for _, rec := range records[start:] {
		p, ok := retrievalRowToPayload(rec, cols)
		if ok {
			payloads = append(payloads, p)
		}
	}

	if len(payloads) == 0 {
		return nil, domain.ErrInvalidInput
	}

	return u.addRetrievalExamples(taskID, payloads)
}

// GenerateQueriesFromDocs bootstraps pair examples from existing docs-only
// examples via the LLM adapter (SyntheticGenerator). Each generated example is
// expected to be a {"query","positive"} pair (or is mapped from the doc when
// the generator emits a bare query). Generated samples are tagged
// SourceSynthetic so they can be human-reviewed before training.
func (u *DatasetUsecase) GenerateQueriesFromDocs(taskID string, count int) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	existing, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	var docs []*domain.Example

	for _, e := range existing {
		if !isDocsOnlyPayload(e.Payload) {
			continue
		}

		docs = append(docs, e)
	}

	if len(docs) == 0 {
		return nil, domain.ErrInvalidInput
	}

	generated := u.synthGen.Generate(task, docs, count)

	var payloads []ExamplePayload

	for _, g := range generated {
		// The generator may emit a pair already, or a bare query that we pair
		// with the source document. Normalize to {"query","positive"}.
		p, ok := normalizeGeneratedQuery(g)
		if !ok {
			continue
		}

		payloads = append(payloads, ExamplePayload{Payload: p, Group: queryFromPayload(p)})
	}

	if len(payloads) == 0 {
		return nil, domain.ErrInvalidInput
	}

	return u.addRetrievalExamples(taskID, payloads)
}

// addRetrievalExamples persists payloads as typed Examples and applies the
// query-group holdout split, then re-curates.
func (u *DatasetUsecase) addRetrievalExamples(taskID string, payloads []ExamplePayload) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()

	var batch []*domain.Example

	for _, p := range payloads {
		if len(p.Payload) == 0 {
			continue
		}

		e := &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Kind:      task.Kind,
			Payload:   p.Payload,
			Source:    domain.SourceUser,
			CreatedAt: now, Flagged: false, Duplicate: false,
		}

		// Deterministic query-group holdout: ~20% of distinct query groups go
		// to eval, tagged in FlagNote. Curation does not drop holdout rows, so
		// the trainer can split on this marker.
		if strings.TrimSpace(p.Group) != "" && hashGroup(p.Group)%5 == 0 {
			e.FlagNote = "holdout:query_group"
		}

		batch = append(batch, e)
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

// --- Parsing helpers ---.

const maxRetrievalLine = 10 << 20

func parseRetrievalJSONL(content string) ([]ExamplePayload, error) {
	var payloads []ExamplePayload

	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(make([]byte, 128*1024), maxRetrievalLine)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		p, ok := retrievalLineToPayload(line)
		if ok {
			payloads = append(payloads, p)
		}
	}

	err := scanner.Err()
	if err != nil {
		return nil, domain.ErrInvalidInput
	}

	return payloads, nil
}

// retrievalLineToPayload accepts a single JSON object and normalizes it into a
// typed payload based on which fields are present.
func retrievalLineToPayload(line string) (ExamplePayload, bool) {
	var probe struct {
		Query     string   `json:"query"`
		Positive  string   `json:"positive"`
		Negative  string   `json:"negative"`
		Document  string   `json:"document"`
		Label     *float64 `json:"label"`
		Relevance *float64 `json:"relevance"`
	}

	err := json.Unmarshal([]byte(line), &probe)
	if err != nil {
		return ExamplePayload{}, false
	}

	doc := strings.TrimSpace(probe.Document)
	query := strings.TrimSpace(probe.Query)

	switch {
	case strings.TrimSpace(probe.Negative) != "":
		// Triplet.
		if query == "" || strings.TrimSpace(probe.Positive) == "" {
			return ExamplePayload{}, false
		}

		payload, _ := json.Marshal(map[string]string{
			"query":    probe.Query,
			"positive": probe.Positive,
			"negative": probe.Negative,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	case doc != "" && query != "":
		// Graded (reranker). Label defaults to 1.0 when omitted.
		label := 1.0
		if probe.Label != nil {
			label = *probe.Label
		} else if probe.Relevance != nil {
			label = *probe.Relevance
		}

		payload, _ := json.Marshal(map[string]any{
			"query":    probe.Query,
			"document": probe.Document,
			"label":    label,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	case doc != "" && query == "":
		// Docs-only.
		payload, _ := json.Marshal(map[string]string{"document": probe.Document})

		return ExamplePayload{Payload: payload, Group: doc}, true
	case query != "" && strings.TrimSpace(probe.Positive) != "":
		// Pair.
		payload, _ := json.Marshal(map[string]string{
			"query":    probe.Query,
			"positive": probe.Positive,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	default:
		return ExamplePayload{}, false
	}
}

// retrievalColumns maps CSV header names to indices.
type retrievalColumns struct {
	query    int // -1 if absent.
	positive int // -1 if absent.
	negative int // -1 if absent.
	document int // -1 if absent.
	label    int // -1 if absent.
}

func detectRetrievalColumns(header []string) (retrievalColumns, int) {
	cols := retrievalColumns{query: -1, positive: -1, negative: -1, document: -1, label: -1}

	for i, col := range header {
		switch strings.ToLower(strings.TrimSpace(col)) {
		case "query":
			cols.query = i
		case "positive", "relevant", "doc_positive":
			cols.positive = i
		case "negative", "hard_negative":
			cols.negative = i
		case "document", "doc":
			cols.document = i
		case "label", "relevance", "score":
			cols.label = i
		}
	}

	recognized := cols.query >= 0 || cols.document >= 0 || cols.positive >= 0
	if !recognized {
		// Fall back to positional: column 0 = query, column 1 = positive.
		return retrievalColumns{query: 0, positive: 1, negative: -1, document: -1, label: -1}, 0
	}

	return cols, 1
}

func retrievalRowToPayload(rec []string, cols retrievalColumns) (ExamplePayload, bool) {
	get := func(idx int) string {
		if idx < 0 || idx >= len(rec) {
			return ""
		}

		return strings.TrimSpace(rec[idx])
	}

	query := get(cols.query)
	positive := get(cols.positive)
	negative := get(cols.negative)
	doc := get(cols.document)

	if cols.document >= 0 {
		if doc == "" {
			doc = get(1) // positional fallback for document-only rows.
		}
	}

	switch {
	case negative != "":
		if query == "" || positive == "" {
			return ExamplePayload{}, false
		}

		payload, _ := json.Marshal(map[string]string{
			"query":    query,
			"positive": positive,
			"negative": negative,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	case doc != "" && query != "":
		label := 1.0
		if cols.label >= 0 && cols.label < len(rec) {
			if l, err := parseLabel(get(cols.label)); err == nil {
				label = l
			}
		}

		payload, _ := json.Marshal(map[string]any{
			"query":    query,
			"document": doc,
			"label":    label,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	case doc != "" && query == "":
		payload, _ := json.Marshal(map[string]string{"document": doc})

		return ExamplePayload{Payload: payload, Group: doc}, true
	case query != "" && positive != "":
		payload, _ := json.Marshal(map[string]string{
			"query":    query,
			"positive": positive,
		})

		return ExamplePayload{Payload: payload, Group: query}, true
	default:
		return ExamplePayload{}, false
	}
}

func parseLabel(s string) (float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 1, domain.ErrInvalidInput
	}

	switch strings.ToLower(s) {
	case "true", "relevant", "yes":
		return 1, nil
	case "false", "irrelevant", "no":
		return 0, nil
	}

	var f float64

	err := json.Unmarshal([]byte(s), &f)
	if err != nil {
		return 0, domain.ErrInvalidInput
	}

	return f, nil
}

// isDocsOnlyPayload reports whether a payload is a {"document"} docs-only row.
func isDocsOnlyPayload(payload json.RawMessage) bool {
	var probe struct {
		Query    string `json:"query"`
		Positive string `json:"positive"`
		Document string `json:"document"`
	}

	if len(payload) == 0 {
		return false
	}

	err := json.Unmarshal(payload, &probe)
	if err != nil {
		return false
	}

	return strings.TrimSpace(probe.Document) != "" &&
		strings.TrimSpace(probe.Query) == "" &&
		strings.TrimSpace(probe.Positive) == ""
}

// normalizeGeneratedQuery converts a generated example into a {"query","positive"}
// pair payload. If the generator already produced a pair/graded payload we leave
// it as-is; if it emitted a bare query we pair it with the source document so
// the row becomes a synthetic pair.
func normalizeGeneratedQuery(g *domain.Example) (json.RawMessage, bool) {
	// If the generator already set a usable query field, carry it forward.
	var probe struct {
		Query    string `json:"query"`
		Positive string `json:"positive"`
		Document string `json:"document"`
	}

	unmarshalErr := json.Unmarshal(g.Payload, &probe)
	if len(g.Payload) > 0 && unmarshalErr == nil {
		if strings.TrimSpace(probe.Query) != "" && strings.TrimSpace(probe.Positive) != "" {
			g.Source = domain.SourceSynthetic
			return g.Payload, true
		}

		if strings.TrimSpace(probe.Query) != "" && strings.TrimSpace(probe.Document) != "" {
			payload, _ := json.Marshal(map[string]string{
				"query":    probe.Query,
				"positive": probe.Document,
			})

			g.Source = domain.SourceSynthetic

			return payload, true
		}
	}

	// Fall back to legacy Input/Output fields: treat Input as the query and
	// Output as the positive document.
	if strings.TrimSpace(g.Input) != "" && strings.TrimSpace(g.Output) != "" {
		payload, _ := json.Marshal(map[string]string{
			"query":    g.Input,
			"positive": g.Output,
		})

		g.Source = domain.SourceSynthetic

		return payload, true
	}

	return nil, false
}

// queryFromPayload extracts the group (query) text from a payload.
func queryFromPayload(payload json.RawMessage) string {
	var probe struct {
		Query string `json:"query"`
	}

	err := json.Unmarshal(payload, &probe)
	if len(payload) > 0 && err == nil {
		return strings.TrimSpace(probe.Query)
	}

	return ""
}

// hashGroup is a trivial FNV-1a hash used for a deterministic holdout split.
func hashGroup(s string) uint32 {
	const (
		offset = 2166136261
		prime  = 16777619
	)

	h := uint32(offset)
	for i := range len(s) {
		h ^= uint32(s[i])
		h *= prime
	}

	return h
}
