package http_test

import (
	"net/http"
	"testing"

	"distillery/internal/domain"
)

func TestTabularHandlerNotFoundAndBadBodies(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)
	td := e.upload(t, churnCSV(30), map[string]string{"target": "churned"})
	base := "/api/v1/tasks/t1/tables/"

	cases := []struct {
		name   string
		method string
		path   string
		body   string
		want   int
	}{
		{"get unknown table", "GET", base + "nope", "", http.StatusNotFound},
		{"preview unknown table", "GET", base + "nope/preview", "", http.StatusNotFound},
		{"delete unknown table", "DELETE", base + "nope", "", http.StatusNotFound},
		{"table of unknown task", "GET", "/api/v1/tasks/ghost/tables/" + td.ID, "", http.StatusNotFound},
		{"mapping with broken json", "PUT", base + td.ID + "/mapping", `{"target":`, http.StatusBadRequest},
		{"mapping with unknown column", "PUT", base + td.ID + "/mapping", `{"target":"nope"}`, http.StatusBadRequest},
		{"mapping retypes unknown column", "PUT", base + td.ID + "/mapping", `{"types":{"nope":"numeric"}}`, http.StatusBadRequest},
		{"training with broken json", "POST", "/api/v1/tasks/t1/training/tabular", `[`, http.StatusBadRequest},
		{"training without table id", "POST", "/api/v1/tasks/t1/training/tabular", `{}`, http.StatusBadRequest},
		{"training on unknown table", "POST", "/api/v1/tasks/t1/training/tabular", `{"table_id":"nope"}`, http.StatusNotFound},
		{"forecast on tabular task", "POST", "/api/v1/tasks/t1/training/forecast", `{"table_id":"` + td.ID + `","horizon":3}`, http.StatusBadRequest},
	}

	for _, tc := range cases {
		code, body := e.do(t, tc.method, tc.path, "application/json", "", []byte(tc.body))
		if code != tc.want {
			// The mapping route may be POST or PUT; accept 404/405 only for path shape, never a 2xx.
			t.Errorf("%s: status %d, want %d (%s)", tc.name, code, tc.want, body)
		}
	}
}

func TestTabularHandlerPreviewLimitAndDelete(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)
	td := e.upload(t, churnCSV(40), map[string]string{"target": "churned"})

	var pv struct {
		Rows []map[string]interface{} `json:"rows"`
	}

	e.json(t, "GET", "/api/v1/tasks/t1/tables/"+td.ID+"/preview?limit=7", "", nil, http.StatusOK, &pv)

	if len(pv.Rows) != 7 {
		t.Errorf("preview honours limit: got %d rows", len(pv.Rows))
	}

	var list []domain.TableDataset

	e.json(t, "GET", "/api/v1/tasks/t1/tables", "", nil, http.StatusOK, &list)

	if len(list) != 1 {
		t.Errorf("list: %d tables", len(list))
	}

	code, body := e.do(t, "DELETE", "/api/v1/tasks/t1/tables/"+td.ID, "", "", nil)
	if code != http.StatusNoContent && code != http.StatusOK {
		t.Fatalf("delete: %d %s", code, body)
	}

	code, _ = e.do(t, "GET", "/api/v1/tasks/t1/tables/"+td.ID, "", "", nil)
	if code != http.StatusNotFound {
		t.Errorf("deleted table still readable: %d", code)
	}
}

func TestTabularHandlerRejectsTableUploadOnNonTableTask(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindCausalLM)

	code, _ := e.do(t, "POST", "/api/v1/tasks/t1/tables", "application/json", "", []byte(`{}`))
	if code < 400 || code >= 500 {
		t.Errorf("want a 4xx, got %d", code)
	}
}

func TestPredictRoutesNeedAValidKeyAndKnownDeployment(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)
	e.upload(t, churnCSV(240), map[string]string{"target": "churned"})

	code, _ := e.do(t, "POST", "/api/v1/inference/ghost/predict", "application/json", "k", []byte(`{"input":{}}`))
	if code != http.StatusUnauthorized && code != http.StatusNotFound {
		t.Errorf("unknown deployment: %d", code)
	}

	for _, p := range []string{"/predict", "/forecast", "/predict-batch"} {
		code, _ = e.do(t, "POST", "/api/v1/inference/ghost"+p, "application/json", "", []byte(`{}`))
		if code != http.StatusUnauthorized && code != http.StatusNotFound {
			t.Errorf("%s without key: %d", p, code)
		}
	}
}
