package usecase //nolint:testpackage // needs unexported helpers

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"distillery/internal/domain"
)

func TestParseCSV(t *testing.T) {
	t.Parallel()

	t.Run("BOM and semicolons", func(t *testing.T) {
		t.Parallel()

		rows, headers, err := parseCSV([]byte("\xef\xbb\xbfa;b\n1;x\n2;\n"))
		if err != nil || len(rows) != 2 || headers[0] != "a" {
			t.Fatalf("rows=%v headers=%v err=%v", rows, headers, err)
		}

		if rows[1]["b"] != nil {
			t.Errorf("empty cell should be nil, got %#v", rows[1]["b"])
		}
	})

	t.Run("tab and pipe delimiters", func(t *testing.T) {
		t.Parallel()

		for _, sep := range []string{"\t", "|"} {
			rows, headers, err := parseCSV([]byte("a" + sep + "b\n1" + sep + "2\n"))
			if err != nil || len(rows) != 1 || len(headers) != 2 {
				t.Errorf("sep %q: rows=%v headers=%v err=%v", sep, rows, headers, err)
			}
		}
	})

	t.Run("missing tokens", func(t *testing.T) {
		t.Parallel()

		rows, _, err := parseCSV([]byte("a,b\nNA,1\nnull,2\nN/A,3\n"))
		if err != nil {
			t.Fatal(err)
		}

		for i, r := range rows {
			if r["a"] != nil {
				t.Errorf("row %d: %v should be missing", i, r["a"])
			}
		}
	})

	for name, in := range map[string]string{
		"empty":      "",
		"duplicate":  "a,a\n1,2\n",
		"ragged":     "a,b\n1,2,3\n",
		"blank name": "a,\n1,2\n",
	} {
		t.Run("rejects "+name, func(t *testing.T) {
			t.Parallel()

			_, _, err := parseCSV([]byte(in))
			if err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func buildCSV(header string, n int, row func(i int) string) []byte {
	var sb strings.Builder

	sb.WriteString(header + "\n")

	for i := range n {
		sb.WriteString(row(i) + "\n")
	}

	return []byte(sb.String())
}

func TestInferColumns(t *testing.T) {
	t.Parallel()

	content := buildCSV("id,user_uuid,age,plan,signup,note,zip,score", 200, func(i int) string {
		return fmt.Sprintf("%d,u-%d,%d,%s,2024-01-%02d,free text number %d with words,%d,%d.5",
			i+1, i, 20+i%40, []string{"a", "b", "c"}[i%3], 1+i%28, i, 10000+i%50, i%9)
	})

	rows, headers, err := parseCSV(content)
	if err != nil {
		t.Fatal(err)
	}

	types := map[string]domain.TableColumnType{}
	for _, c := range inferColumns(headers, rows) {
		types[c.Name] = c.Type
	}

	want := map[string]domain.TableColumnType{
		"age": domain.ColNumeric, "plan": domain.ColCategorical, "signup": domain.ColDatetime,
		"note": domain.ColText, "score": domain.ColNumeric,
	}

	for name, typ := range want {
		if types[name] != typ {
			t.Errorf("%s: got %q, want %q", name, types[name], typ)
		}
	}

	if types["id"] == domain.ColNumeric || types["user_uuid"] == domain.ColNumeric {
		t.Errorf("identifier columns must not be numeric features: id=%q uuid=%q", types["id"], types["user_uuid"])
	}
}

func TestInferFrequency(t *testing.T) {
	t.Parallel()

	gen := func(step func(i int) string) []tableRow {
		var rows []tableRow
		for i := range 30 {
			rows = append(rows, tableRow{"ts": step(i)})
		}

		return rows
	}

	cases := map[string]struct {
		rows []tableRow
		want string
	}{
		"daily":   {gen(dayN), "D"},
		"weekly":  {gen(func(i int) string { return dayN(7 * i) }), "W"},
		"monthly": {gen(func(i int) string { return fmt.Sprintf("%d-%02d-01T00:00:00", 2020+i/12, i%12+1) }), "M"},
		"hourly":  {gen(func(i int) string { return fmt.Sprintf("2024-01-%02dT%02d:00:00", 1+i/24, i%24) }), "H"},
	}

	for name, c := range cases {
		if got := inferFrequency(c.rows, "ts"); got != c.want {
			t.Errorf("%s: inferFrequency = %q, want %q", name, got, c.want)
		}
	}

	// Duplicate timestamps (multi-series data) must not collapse the median gap to zero.
	var multi []tableRow

	for i := range 30 {
		for range 3 {
			multi = append(multi, tableRow{"ts": dayN(i)})
		}
	}

	if got := inferFrequency(multi, "ts"); got != "D" {
		t.Errorf("multi-series daily: %q", got)
	}
}

func dayN(n int) string {
	d := 1 + n
	m := 1 + d/28
	d = 1 + (d-1)%28

	return fmt.Sprintf("2024-%02d-%02dT00:00:00", m, d)
}

func TestTargetLeak(t *testing.T) {
	t.Parallel()

	var rows []tableRow //nolint:prealloc // test fixture

	for i := range 100 {
		y := i % 2
		rows = append(rows, tableRow{
			"y": strconv.Itoa(y), "copy": strconv.Itoa(y), "scaled": strconv.Itoa(y*3 + 1),
			"noise": strconv.Itoa((i * 37) % 11), "cat": map[bool]string{true: "p", false: "n"}[i%2 == 1],
		})
	}

	if msg := targetLeak(rows, "copy", domain.ColNumeric, "y", domain.ColNumeric); msg == "" {
		t.Error("identical column should be flagged")
	}

	if msg := targetLeak(rows, "scaled", domain.ColNumeric, "y", domain.ColNumeric); msg == "" {
		t.Error("linear proxy should be flagged")
	}

	if msg := targetLeak(rows, "cat", domain.ColCategorical, "y", domain.ColNumeric); msg == "" {
		t.Error("pure category proxy should be flagged")
	}

	if msg := targetLeak(rows, "noise", domain.ColNumeric, "y", domain.ColNumeric); msg != "" {
		t.Errorf("noise flagged: %s", msg)
	}
}
