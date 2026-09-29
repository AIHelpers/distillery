package usecase

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"distillery/internal/domain"
)

// tableRow is one parsed table row: column -> string value (nil = missing).
// Thresholds for frequency inference and target-leak screening.
const (
	hourlyMaxHours = 20   // a median gap below this many hours is hourly.
	dailyMaxHours  = 36   // below this many hours (and above hourly) is daily.
	minLeakSamples = 20   // fewer paired samples than this are not screened.
	leakPurity     = 0.99 // share of rows explained by a value->target lookup that is flagged.
	leakSimilarity = 0.98 // agreement / correlation / purity at or above this is flagged.
)

type tableRow = map[string]interface{}

var missingTokens = map[string]bool{
	"": true, "na": true, "n/a": true, "nan": true, "null": true, "none": true, "#n/a": true,
}

// isMissingToken reports whether a CSV cell counts as a missing value.
func isMissingToken(s string) bool {
	return missingTokens[strings.ToLower(strings.TrimSpace(s))]
}

// detectDelimiter picks the field delimiter from the header line: comma,
// semicolon (common in European/Russian Excel exports), tab or pipe.
func detectDelimiter(content []byte) rune {
	line := content

	if i := bytes.IndexByte(content, '\n'); i >= 0 {
		line = content[:i]
	}

	best, bestCount := ',', 0

	for _, d := range []rune{',', ';', '\t', '|'} {
		if c := bytes.Count(line, []byte(string(d))); c > bestCount {
			best, bestCount = d, c
		}
	}

	return best
}

// parseCSV parses CSV bytes into header + row maps. It strips a UTF-8 BOM,
// auto-detects the delimiter, trims headers, and rejects empty/duplicate
// headers and ragged rows with a row number so the user can fix the file.
func parseCSV(content []byte) (rows []tableRow, headers []string, err error) {
	content = bytes.TrimPrefix(content, []byte("\xef\xbb\xbf"))

	if !utf8.Valid(content) {
		return nil, nil, fmt.Errorf("%w: file is not valid UTF-8 text", domain.ErrInvalidInput)
	}

	r := csv.NewReader(bytes.NewReader(content))
	r.Comma = detectDelimiter(content)
	r.FieldsPerRecord = -1
	r.LazyQuotes = true

	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(records) == 0 {
		return nil, nil, fmt.Errorf("%w: empty CSV", domain.ErrInvalidInput)
	}

	headers = make([]string, len(records[0]))
	seen := map[string]bool{}

	for i, h := range records[0] {
		h = strings.TrimSpace(h)
		if h == "" {
			return nil, nil, fmt.Errorf("%w: column %d has an empty header", domain.ErrInvalidInput, i+1)
		}

		if seen[h] {
			return nil, nil, fmt.Errorf("%w: duplicate column header %q", domain.ErrInvalidInput, h)
		}

		seen[h] = true
		headers[i] = h
	}

	rows = make([]tableRow, 0, len(records)-1)

	for n, rec := range records[1:] {
		if len(rec) != len(headers) {
			return nil, nil, fmt.Errorf("%w: row %d has %d fields, expected %d", domain.ErrInvalidInput, n+2, len(rec), len(headers))
		}

		row := make(tableRow, len(headers))

		for i, h := range headers {
			if isMissingToken(rec[i]) {
				row[h] = nil
			} else {
				row[h] = strings.TrimSpace(rec[i])
			}
		}

		rows = append(rows, row)
	}

	return rows, headers, nil
}

var idNameRE = regexp.MustCompile(`(?i)(^|[_\s-])(id|uuid|guid|index|idx|key|no|number)$|^id($|[_\s-])`)

// inferColumns detects each column's type from its values.
func inferColumns(headers []string, rows []tableRow) []domain.TableColumn {
	cols := make([]domain.TableColumn, 0, len(headers))

	for _, h := range headers {
		col := domain.TableColumn{Name: h}

		var (
			numeric, datetime, integer int
			total, spaced              int
			uniqueVals                 = map[string]bool{}
		)

		for _, r := range rows {
			s, _ := r[h].(string)
			if s == "" {
				col.Missing++

				continue
			}

			total++
			uniqueVals[s] = true

			if strings.ContainsAny(s, " \t") {
				spaced++
			}

			if len(col.Sample) < 3 {
				col.Sample = append(col.Sample, s)
			}

			f, err := strconv.ParseFloat(s, 64)
			if err == nil && !math.IsNaN(f) && !math.IsInf(f, 0) {
				numeric++

				if f == math.Trunc(f) && !strings.ContainsAny(s, ".eE") {
					integer++
				}
			} else if _, ok := domain.ParseTableTime(s); ok {
				datetime++
			}
		}

		col.Unique = len(uniqueVals)
		col.IntegerLike = total > 0 && integer == numeric && numeric > 0
		col.Type = classifyColumn(h, total, numeric, datetime, integer, len(uniqueVals), spaced)
		cols = append(cols, col)
	}

	return cols
}

// classifyColumn decides a column's type from observed value statistics.
func classifyColumn(name string, total, numeric, datetime, integer, unique, spaced int) domain.TableColumnType {
	switch {
	case total == 0:
		return domain.ColIgnored
	case datetime*10 >= total*9:
		return domain.ColDatetime
	case numeric*10 >= total*9:
		// Only integer-valued, unique-per-row columns with an id-like name are
		// identifiers; continuous numerics are never IDs.
		if unique == total && total > 20 && integer == numeric && idNameRE.MatchString(name) {
			return domain.ColID
		}

		return domain.ColNumeric
	case unique == total && total > 20 && spaced*2 >= total:
		return domain.ColText // unique free text, not an identifier.
	case unique == total && total > 20:
		return domain.ColID
	case unique > 50 && float64(unique)/float64(total) > 0.5:
		return domain.ColText
	default:
		return domain.ColCategorical
	}
}

// retypeTarget makes sure the chosen target column is usable as a label: an
// auto-detected ID/text column that is really a number or a label becomes
// numeric/categorical.
func retypeTarget(col *domain.TableColumn, rows []tableRow) {
	if col.Type != domain.ColID && col.Type != domain.ColText && col.Type != domain.ColIgnored {
		return
	}

	numeric, total := 0, 0

	for _, r := range rows {
		s, _ := r[col.Name].(string)
		if s == "" {
			continue
		}

		total++

		_, err := strconv.ParseFloat(s, 64)
		if err == nil {
			numeric++
		}
	}

	if total > 0 && numeric*10 >= total*9 {
		col.Type = domain.ColNumeric
	} else {
		col.Type = domain.ColCategorical
	}
}

// normalizeDatetime rewrites a column's values to naive-UTC ISO timestamps.
// It reports false (leaving rows untouched) when fewer than 90% of the
// non-missing values parse.
func normalizeDatetime(rows []tableRow, col string) bool {
	parsed := make([]time.Time, len(rows))
	ok := make([]bool, len(rows))
	total, good := 0, 0

	for i, r := range rows {
		s, _ := r[col].(string)
		if s == "" {
			continue
		}

		total++

		if t, valid := domain.ParseTableTime(s); valid {
			parsed[i], ok[i] = t, true
			good++
		}
	}

	if total == 0 || good*10 < total*9 {
		return false
	}

	for i, r := range rows {
		if ok[i] {
			r[col] = parsed[i].UTC().Format("2006-01-02T15:04:05")
		} else if s, _ := r[col].(string); s != "" {
			r[col] = nil
		}
	}

	return true
}

// inferFrequency guesses H/D/W/M/Q/Y from the median spacing between the
// distinct timestamps of the timestamp column.
func inferFrequency(rows []tableRow, tsCol string) string {
	if tsCol == "" {
		return ""
	}

	seen := map[int64]bool{}

	var times []int64

	for _, r := range rows {
		s, _ := r[tsCol].(string)

		t, ok := domain.ParseTableTime(s)
		if !ok {
			continue
		}

		u := t.Unix()
		if !seen[u] {
			seen[u] = true
			times = append(times, u)
		}
	}

	if len(times) < 3 {
		return ""
	}

	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })

	diffs := make([]float64, 0, len(times)-1)

	for i := 1; i < len(times); i++ {
		diffs = append(diffs, float64(times[i]-times[i-1]))
	}

	sort.Float64s(diffs)

	med := diffs[len(diffs)/2] / 3600 // hours.

	switch {
	case med < hourlyMaxHours:
		return "H"
	case med < dailyMaxHours:
		return "D"
	case med < 24*10:
		return "W"
	case med < 24*45:
		return "M"
	case med < 24*135:
		return "Q"
	default:
		return "Y"
	}
}

// warnLeakage recomputes the mapper warnings of every column.
func warnLeakage(td *domain.TableDataset, rows []tableRow) {
	for i := range td.Columns {
		c := &td.Columns[i]
		c.Warnings = nil

		switch {
		case c.Type == domain.ColID:
			c.Warnings = append(c.Warnings, "ID-like column: unique per row; excluded from features by default")
		case c.Type == domain.ColText:
			c.Warnings = append(c.Warnings, "free-text column: ignored as a feature by default")
		case c.Type == domain.ColDatetime && td.Kind == domain.KindTabular && c.Name != td.Target:
			c.Warnings = append(c.Warnings,
				"datetime column: time-based split is the default so the model is scored on later rows")
		}

		if c.Unique == 1 && c.Type != domain.ColIgnored {
			c.Warnings = append(c.Warnings, "constant column: carries no information")
		}
	}

	if td.Kind != domain.KindTabular || td.Target == "" || len(rows) == 0 {
		return
	}

	for i := range td.Columns {
		c := &td.Columns[i]
		if c.Name == td.Target || td.IsExcluded(c.Name) {
			continue
		}

		if msg := targetLeak(rows, c.Name, c.Type, td.Target, td.ColumnType(td.Target)); msg != "" {
			c.Warnings = append(c.Warnings, msg)
		}
	}
}

// targetLeak returns a warning when column col is (nearly) a copy of, or
// fully determines, the target.
func targetLeak(rows []tableRow, col string, colType domain.TableColumnType, target string, targetType domain.TableColumnType) string { //nolint:cyclop,lll // long scenario test/decision list
	same, n := 0, 0
	groups := map[string]map[string]int{}

	var xs, ys []float64

	for _, r := range rows {
		a, _ := r[col].(string)
		b, _ := r[target].(string)

		if a == "" || b == "" {
			continue
		}

		n++

		if a == b {
			same++
		}

		if groups[a] == nil {
			groups[a] = map[string]int{}
		}

		groups[a][b]++

		fa, ea := strconv.ParseFloat(a, 64)
		fb, eb := strconv.ParseFloat(b, 64)

		if ea == nil && eb == nil {
			xs, ys = append(xs, fa), append(ys, fb)
		}
	}

	if n < minLeakSamples {
		return ""
	}

	if float64(same)/float64(n) >= leakSimilarity {
		return "identical to the target: almost certainly target leakage"
	}

	if colType == domain.ColNumeric && targetType == domain.ColNumeric && len(xs) >= minLeakSamples {
		if r := math.Abs(pearson(xs, ys)); r >= leakSimilarity {
			return fmt.Sprintf("|correlation| with the target is %.3f: likely derived from the target", r)
		}
	}

	if colType == domain.ColCategorical && len(groups) > 1 && float64(n)/float64(len(groups)) >= 3 {
		pure := 0

		for _, g := range groups {
			mx, tot := 0, 0

			for _, c := range g {
				tot += c

				if c > mx {
					mx = c
				}
			}

			pure += mx
			_ = tot
		}

		if float64(pure)/float64(n) >= leakPurity {
			return "each value maps to a single target value: likely a proxy of the target"
		}
	}

	return ""
}

func pearson(x, y []float64) float64 {
	n := float64(len(x))

	var sx, sy float64

	for i := range x {
		sx += x[i]
		sy += y[i]
	}

	mx, my := sx/n, sy/n

	var sxy, sxx, syy float64

	for i := range x {
		dx, dy := x[i]-mx, y[i]-my
		sxy += dx * dy
		sxx += dx * dx
		syy += dy * dy
	}

	if sxx <= 0 || syy <= 0 {
		return 0
	}

	return sxy / math.Sqrt(sxx*syy)
}

// ParseTableCSV parses an inference-time CSV (batch predict / forecast
// history) with the same delimiter/BOM handling as table uploads. Cells are
// strings; missing cells are omitted so schema validation reports them.
func ParseTableCSV(content []byte) (rows []map[string]interface{}, headers []string, err error) {
	rows, headers, err = parseCSV(content)
	if err != nil {
		return nil, nil, err
	}

	out := make([]map[string]interface{}, len(rows))

	for i, r := range rows {
		m := make(map[string]interface{}, len(r))

		for k, v := range r {
			if v != nil {
				m[k] = v
			}
		}

		out[i] = m
	}

	return out, headers, nil
}
