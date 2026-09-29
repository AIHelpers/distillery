package http_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/infra/blob"
	"distillery/internal/infra/simulation"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

type depResp struct {
	domain.Deployment

	APIKey string `json:"api_key"`
}

type tabEnv struct {
	srv    *httptest.Server
	jobs   *memory.TrainingRepo
	taskID string
}

func newTabEnv(t *testing.T, kind domain.ModelKind) *tabEnv {
	t.Helper()

	store := memory.NewStore("")
	tasks := memory.NewTaskRepo(store)
	tables := memory.NewTableDatasetRepo(store)
	jobs := memory.NewTrainingRepo(store)
	idGen := usecase.NewRandomIDGenerator()
	sim := simulation.NewTableInferenceEngine()

	tabUC := usecase.NewTabularUsecase(tasks, tables, jobs, blob.NewLocalStore(t.TempDir()), simulation.NewTableFineTuner(), idGen)
	depUC := usecase.NewDeploymentUsecaseFull(
		tasks, jobs, memory.NewExampleRepo(store), memory.NewDeploymentRepo(store),
		simulation.NewInferenceEngine(), simulation.NewExporter(), sim, sim, tabUC, idGen,
	)
	tabH := deliveryhttp.NewTabularHandler(tabUC, depUC)
	router := deliveryhttp.NewRouter(deliveryhttp.Handlers{
		Deployment: deliveryhttp.NewDeploymentHandler(depUC).WithTabular(tabH),
		Tabular:    tabH,
	}, fstest.MapFS{})

	err := tasks.Create(&domain.Task{ID: "t1", Name: "table", Kind: kind, CreatedAt: time.Now()})
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(router)
	t.Cleanup(srv.Close)

	return &tabEnv{srv: srv, jobs: jobs, taskID: "t1"}
}

func (e *tabEnv) do(t *testing.T, method, path, ctype, key string, body []byte) (code int, payload []byte) {
	t.Helper()

	req, _ := http.NewRequest(method, e.srv.URL+path, bytes.NewReader(body))
	if ctype != "" {
		req.Header.Set("Content-Type", ctype)
	}

	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	out, _ := io.ReadAll(resp.Body)

	return resp.StatusCode, out
}

func (e *tabEnv) json(t *testing.T, method, path, key string, in interface{}, want int, out interface{}) {
	t.Helper()

	var b []byte
	if in != nil {
		b, _ = json.Marshal(in) //nolint:errchkjson // test fixture marshalling
	}

	code, body := e.do(t, method, path, "application/json", key, b)
	if code != want {
		t.Fatalf("%s %s: status %d (want %d): %s", method, path, code, want, body)
	}

	if out != nil {
		err := json.Unmarshal(body, out)
		if err != nil {
			t.Fatalf("decode %s: %v\n%s", body, err, body)
		}
	}
}

func (e *tabEnv) upload(t *testing.T, csvData string, fields map[string]string) domain.TableDataset {
	t.Helper()

	var buf bytes.Buffer

	mw := multipart.NewWriter(&buf)
	fw, _ := mw.CreateFormFile("file", "data.csv")
	_, _ = fw.Write([]byte(csvData))

	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}

	_ = mw.Close()

	code, body := e.do(t, "POST", "/api/v1/tasks/"+e.taskID+"/tables", mw.FormDataContentType(), "", buf.Bytes())
	if code != http.StatusCreated && code != http.StatusOK {
		t.Fatalf("upload: %d %s", code, body)
	}

	var td domain.TableDataset

	err := json.Unmarshal(body, &td)
	if err != nil {
		t.Fatal(err)
	}

	return td
}

func (e *tabEnv) wait(t *testing.T, jobID string) *domain.TrainingJob {
	t.Helper()

	deadline := time.Now().Add(20 * time.Second)

	for time.Now().Before(deadline) {
		j, err := e.jobs.Get(jobID)
		if err == nil && j.Status != domain.TrainingRunning {
			return j
		}

		time.Sleep(20 * time.Millisecond)
	}

	t.Fatal("training did not finish")

	return nil
}

func churnCSV(n int) string {
	var sb strings.Builder

	sb.WriteString("customer_id;tenure_months;plan;monthly_spend;churned\n")

	plans := []string{"basic", "pro", "enterprise"}

	for i := range n {
		tenure := 1 + (i*7)%60
		plan := plans[i%3]
		spend := 20 + (i*13)%80
		churn := "no"

		if tenure < 15 && plan == "basic" || (i%11 == 0 && tenure < 30) {
			churn = "yes"
		}

		fmt.Fprintf(&sb, "%d;%d;%s;%d,5;%s\n", 1000+i, tenure, plan, spend, churn)
	}

	return sb.String()
}

func TestTabularEndToEnd(t *testing.T) { //nolint:cyclop // long scenario test/decision list
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)
	td := e.upload(t, churnCSV(240), map[string]string{"target": "churned"})

	if td.Target != "churned" || len(td.Columns) != 5 {
		t.Fatalf("unexpected mapping: %+v", td)
	}

	types := map[string]domain.TableColumnType{}
	for _, c := range td.Columns {
		types[c.Name] = c.Type
	}

	if types["customer_id"] != domain.ColIgnored && types["customer_id"] != "id" {
		t.Errorf("customer_id should not be a feature, got %q", types["customer_id"])
	}

	if types["tenure_months"] != domain.ColNumeric || types["plan"] != domain.ColCategorical {
		t.Errorf("bad type inference: %v", types)
	}

	// Decimal comma is not silently accepted as a number: monthly_spend is text/categorical, not numeric.
	if types["monthly_spend"] == domain.ColNumeric {
		t.Errorf("'23,5' must not be inferred as numeric")
	}

	var job domain.TrainingJob
	e.json(t, "POST", "/api/v1/tasks/t1/training/tabular", "", map[string]interface{}{"table_id": td.ID, "metric": "roc_auc"}, http.StatusAccepted, &job)

	done := e.wait(t, job.ID)
	if done.Status != domain.TrainingCompleted || done.Metrics == nil {
		t.Fatalf("job did not complete: %+v", done)
	}

	m := done.Metrics
	if m.Primary != "roc_auc" || m.Baselines["roc_auc"] != 0.5 || m.PrimaryVal <= 0.5 || m.FeatureSchema == nil {
		t.Fatalf("metrics look wrong: %+v", m)
	}

	if len(m.ConfusionMatrix) != 2 || len(m.ROCCurve) < 2 {
		t.Errorf("missing confusion matrix / ROC curve: %+v %v", m.ConfusionMatrix, m.ROCCurve)
	}

	// Deploy is explicit; the raw key is returned once.
	var dep depResp

	e.json(t, "POST", "/api/v1/tasks/t1/deploy", "", map[string]interface{}{}, http.StatusCreated, &dep)

	if dep.APIKey == "" || dep.FeatureSchema == nil {
		t.Fatalf("deploy response incomplete: %+v", dep)
	}

	base := "/api/v1/inference/" + dep.ID
	in := map[string]interface{}{"input": map[string]interface{}{"tenure_months": 3, "plan": "BASIC", "monthly_spend": "40"}}

	code, _ := e.do(t, "POST", base+"/predict", "application/json", "", mustJSON(in))
	if code != http.StatusUnauthorized {
		t.Errorf("predict without key: %d, want 401", code)
	}

	var pred struct {
		Kind   string                 `json:"kind"`
		Result domain.TablePrediction `json:"result"`
	}

	e.json(t, "POST", base+"/predict", dep.APIKey, in, http.StatusOK, &pred)

	if pred.Kind != "tabular" || pred.Result.Probability == nil || pred.Result.Prediction == nil {
		t.Fatalf("bad prediction: %+v", pred)
	}

	// Schema validation errors are 4xx with a clear message.
	bad := map[string]interface{}{"input": map[string]interface{}{"tenure_months": "abc", "plan": "gold"}}

	code, body := e.do(t, "POST", base+"/predict", "application/json", dep.APIKey, mustJSON(bad))
	if code != http.StatusBadRequest && code != http.StatusUnprocessableEntity {
		t.Errorf("invalid input: status %d %s", code, body)
	}

	// Batch: CSV in, CSV out; a bad row does not fail the batch.
	batch := "tenure_months,plan,monthly_spend\n3,basic,40\nnope,pro,10\n50,enterprise,90\n=cmd|' /C calc'!A0,pro,1\n"
	code, out := e.do(t, "POST", base+"/predict-batch", "text/csv", dep.APIKey, []byte(batch))

	if code != http.StatusOK {
		t.Fatalf("batch: %d %s", code, out)
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) != 5 || !strings.Contains(lines[0], "prediction") || !strings.Contains(lines[0], "error") {
		t.Fatalf("unexpected batch output:\n%s", out)
	}

	if strings.Contains(lines[2], ",=") || strings.HasPrefix(strings.Split(lines[4], ",")[0], "=") {
		t.Errorf("formula injection not neutralised:\n%s", out)
	}
}

func TestTabularUploadValidation(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)

	for name, csvData := range map[string]string{
		"empty":      "",
		"header":     "a,b,c\n",
		"duplicates": "a,a\n1,2\n3,4\n",
		"ragged":     "a,b\n1,2\n3\n",
	} {
		var buf bytes.Buffer

		mw := multipart.NewWriter(&buf)
		fw, _ := mw.CreateFormFile("file", "x.csv")
		_, _ = fw.Write([]byte(csvData))
		_ = mw.Close()

		code, _ := e.do(t, "POST", "/api/v1/tasks/t1/tables", mw.FormDataContentType(), "", buf.Bytes())
		if code != http.StatusBadRequest && code != http.StatusUnprocessableEntity {
			t.Errorf("%s: status %d, want 4xx", name, code)
		}
	}

	code, _ := e.do(t, "POST", "/api/v1/tasks/t1/tables", "application/json", "", []byte(`{}`))
	if code != http.StatusBadRequest {
		t.Errorf("non-multipart upload: %d", code)
	}
}

func TestTabularTrainingRequiresRows(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTabular)
	td := e.upload(t, "a,y\n1,x\n2,y\n3,x\n", map[string]string{"target": "y"})

	code, body := e.do(t, "POST", "/api/v1/tasks/t1/training/tabular", "application/json", "", mustJSON(map[string]interface{}{"table_id": td.ID}))
	if code != http.StatusBadRequest && code != http.StatusUnprocessableEntity {
		t.Fatalf("tiny table should be rejected: %d %s", code, body)
	}
}

func TestForecastEndToEnd(t *testing.T) {
	t.Parallel()

	e := newTabEnv(t, domain.KindTimeSeries)

	var sb strings.Builder

	sb.WriteString("date,sku,units\n")

	day := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	for i := range 140 {
		for _, sku := range []string{"a", "b"} {
			v := 50 + 10*math.Sin(2*math.Pi*float64(i)/7) + float64(i%5)*0.3
			fmt.Fprintf(&sb, "%s,%s,%.2f\n", day.AddDate(0, 0, i).Format("2006-01-02"), sku, v)
		}
	}

	td := e.upload(t, sb.String(), map[string]string{"target": "units", "timestamp": "date", "item_id_column": "sku"})
	if td.Frequency != "D" {
		t.Fatalf("frequency = %q, want D", td.Frequency)
	}

	// Unconfirmed frequency is refused ...
	code, _ := e.do(t, "POST", "/api/v1/tasks/t1/training/forecast", "application/json", "", mustJSON(map[string]interface{}{"table_id": td.ID, "horizon": 7}))
	if code == http.StatusAccepted {
		t.Fatalf("training must require a confirmed frequency")
	}

	// ... and confirmed through the mapping.
	e.json(t, "PUT", "/api/v1/tasks/t1/tables/"+td.ID+"/mapping", "", map[string]interface{}{"frequency": "D", "horizon": 7}, http.StatusOK, nil)

	var job domain.TrainingJob
	e.json(t, "POST", "/api/v1/tasks/t1/training/forecast", "", map[string]interface{}{"table_id": td.ID, "horizon": 7}, http.StatusAccepted, &job)

	done := e.wait(t, job.ID)
	if done.Status != domain.TrainingCompleted || done.Metrics == nil {
		t.Fatalf("forecast job failed: %+v", done)
	}

	if done.Metrics.SeasonLength != 7 || done.Metrics.BacktestPlot == nil || len(done.Metrics.Leaderboard) < 2 {
		t.Fatalf("bad forecast metrics: %+v", done.Metrics)
	}

	var dep depResp

	e.json(t, "POST", "/api/v1/tasks/t1/deploy", "", map[string]interface{}{}, http.StatusCreated, &dep)

	if !strings.HasSuffix(dep.Endpoint, "/forecast") {
		t.Errorf("time-series endpoint = %q", dep.Endpoint)
	}

	base := "/api/v1/inference/" + dep.ID

	var fc struct {
		Result struct {
			Forecast []float64 `json:"forecast"`
			Lower    []float64 `json:"lower"`
			Upper    []float64 `json:"upper"`
		} `json:"result"`
	}

	e.json(t, "POST", base+"/forecast", dep.APIKey, map[string]interface{}{"horizon": 5, "item_id": "a"}, http.StatusOK, &fc)

	if len(fc.Result.Forecast) != 5 {
		t.Fatalf("forecast length %d", len(fc.Result.Forecast))
	}

	for i := range fc.Result.Forecast {
		if fc.Result.Lower[i] > fc.Result.Forecast[i] || fc.Result.Upper[i] < fc.Result.Forecast[i] {
			t.Errorf("interval does not bracket the forecast at %d", i)
		}
	}

	// Two series and no item_id: a clear client error.
	code, body := e.do(t, "POST", base+"/forecast", "application/json", dep.APIKey, mustJSON(map[string]interface{}{"horizon": 3}))
	if code != http.StatusBadRequest || !strings.Contains(string(body), "item_id") {
		t.Errorf("missing item_id: %d %s", code, body)
	}
}

func mustJSON(v interface{}) []byte {
	b, _ := json.Marshal(v) //nolint:errchkjson // test fixture marshalling

	return b
}
