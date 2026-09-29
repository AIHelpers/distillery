package usecase_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/infra/blob"
	"distillery/internal/infra/simulation"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

type tabFixture struct {
	uc    *usecase.TabularUsecase
	jobs  *memory.TrainingRepo
	tasks *memory.TaskRepo
}

func newTabFixture(t *testing.T, kind domain.ModelKind) *tabFixture {
	t.Helper()

	store := memory.NewStore("")
	tasks := memory.NewTaskRepo(store)
	jobs := memory.NewTrainingRepo(store)
	uc := usecase.NewTabularUsecase(
		tasks, memory.NewTableDatasetRepo(store), jobs, blob.NewLocalStore(t.TempDir()),
		simulation.NewTableFineTuner(), usecase.NewRandomIDGenerator(),
	)

	err := tasks.Create(&domain.Task{ID: "t1", Name: "tab", Kind: kind, CreatedAt: time.Now()})
	if err != nil {
		t.Fatal(err)
	}

	return &tabFixture{uc: uc, jobs: jobs, tasks: tasks}
}

func churnCSV(n int) []byte {
	var b strings.Builder

	b.WriteString("id,tenure,plan,signup,churned\n")

	for i := range n {
		plan := []string{"basic", "pro", "team"}[i%3]
		churned := "no"

		if i%4 == 0 {
			churned = "yes"
		}

		fmt.Fprintf(&b, "%d,%d,%s,2024-01-%02d,%s\n", i, i%40, plan, i%28+1, churned)
	}

	return []byte(b.String())
}

func salesCSV(days int) []byte {
	var b strings.Builder

	b.WriteString("day,sales,store\n")

	for i := range days {
		fmt.Fprintf(&b, "2024-%02d-%02d,%d,A\n", i/28+1, i%28+1, 100+(i%7)*5)
	}

	return []byte(b.String())
}

func TestTabularUsecase_UploadInfersColumnsAndDefaults(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	td, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "churn.csv", Content: churnCSV(60), Target: "churned"})
	if err != nil {
		t.Fatal(err)
	}

	if td.RowCount != 60 || td.Kind != domain.KindTabular || td.Target != "churned" {
		t.Fatalf("unexpected dataset: %+v", td)
	}

	want := map[string]domain.TableColumnType{
		"id": domain.ColID, "tenure": domain.ColNumeric, "plan": domain.ColCategorical,
		"signup": domain.ColDatetime, "churned": domain.ColCategorical,
	}
	for name, typ := range want {
		if got := td.ColumnType(name); got != typ {
			t.Errorf("column %s: got %q want %q", name, got, typ)
		}
	}

	if td.Extra.Split != string(domain.SplitTime) {
		t.Errorf("a table with a datetime column defaults to a time split, got %q", td.Extra.Split)
	}
}

func TestTabularUsecase_UploadRejections(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	cases := map[string][]byte{
		"empty":        {},
		"header only":  []byte("a,b\n"),
		"ragged":       []byte("a,b\n1,2\n3\n"),
		"dup headers":  []byte("a,a\n1,2\n"),
		"binary":       {0xff, 0xfe, 0x00, 0x01},
		"blank header": []byte("a,\n1,2\n"),
	}
	for name, content := range cases {
		_, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "x.csv", Content: content})
		if !errors.Is(err, domain.ErrInvalidInput) {
			t.Errorf("%s: want ErrInvalidInput, got %v", name, err)
		}
	}

	_, err := f.uc.UploadTable("missing", usecase.UploadTableRequest{Content: []byte("a\n1\n")})
	if err == nil {
		t.Error("an unknown task must be an error")
	}

	other := newTabFixture(t, domain.KindCausalLM)

	_, err = other.uc.UploadTable("t1", usecase.UploadTableRequest{Content: []byte("a,b\n1,2\n")})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("a non-table task cannot upload a table, got %v", err)
	}
}

func TestTabularUsecase_UploadTooManyRows(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	var b strings.Builder

	b.WriteString("a\n")

	for range usecase.MaxTableRows + 1 {
		b.WriteString("1\n")
	}

	_, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Content: []byte(b.String())})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("want ErrInvalidInput for an oversized table, got %v", err)
	}
}

func TestTabularUsecase_GetListDeletePreview(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	td, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "c.csv", Content: churnCSV(30), Target: "churned"})
	if err != nil {
		t.Fatal(err)
	}

	got, err := f.uc.GetTable("t1", td.ID)
	if err != nil || got.ID != td.ID {
		t.Fatalf("get: %v %+v", err, got)
	}

	list, err := f.uc.ListTables("t1")
	if err != nil || len(list) != 1 {
		t.Fatalf("list: %v %d", err, len(list))
	}

	rows, err := f.uc.TablePreview("t1", td.ID, 5)
	if err != nil || len(rows) != 5 {
		t.Fatalf("preview: %v %d", err, len(rows))
	}

	if rows[0]["plan"] != "basic" {
		t.Errorf("preview rows should carry the CSV values, got %+v", rows[0])
	}

	_, err = f.uc.GetTable("other-task", td.ID)
	if err == nil {
		t.Error("a table is scoped to its task")
	}

	err = f.uc.DeleteTable("t1", td.ID)
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.uc.GetTable("t1", td.ID)
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("deleted table: want ErrNotFound, got %v", err)
	}

	_, err = f.uc.TablePreview("t1", td.ID, 5)
	if err == nil {
		t.Error("preview of a deleted table must fail")
	}
}

func TestTabularUsecase_UpdateMapping(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	td, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "c.csv", Content: churnCSV(60)})
	if err != nil {
		t.Fatal(err)
	}

	excl := []string{"signup"}
	group := "plan"

	upd, err := f.uc.UpdateTableMapping("t1", td.ID, usecase.MappingUpdate{
		Target: "churned", Excluded: &excl, Split: domain.SplitGroup, GroupColumn: &group,
		Types: map[string]domain.TableColumnType{"tenure": domain.ColCategorical},
	})
	if err != nil {
		t.Fatal(err)
	}

	if upd.Target != "churned" || upd.ColumnType("tenure") != domain.ColCategorical {
		t.Errorf("mapping not applied: %+v", upd)
	}

	if len(upd.Excluded) != 1 || upd.Extra.Split != string(domain.SplitGroup) || upd.Extra.GroupColumn != "plan" {
		t.Errorf("exclusion/split not applied: %+v", upd)
	}

	bad := []usecase.MappingUpdate{
		{Target: "nope"},
		{Types: map[string]domain.TableColumnType{"nope": domain.ColNumeric}},
		{Types: map[string]domain.TableColumnType{"plan": "weird"}},
		{Split: "sideways"},
		{Split: domain.SplitGroup, GroupColumn: ptr("nope")},
		{Task: "clustering"},
	}
	for i, m := range bad {
		_, err = f.uc.UpdateTableMapping("t1", td.ID, m)
		if !errors.Is(err, domain.ErrInvalidInput) {
			t.Errorf("case %d: want ErrInvalidInput, got %v", i, err)
		}
	}
}

func ptr[T any](v T) *T { return &v }

func TestTabularUsecase_StartTabularTrainingValidation(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTabular)

	td, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "c.csv", Content: churnCSV(60)})
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.uc.StartTabularTraining("t1", td.ID, domain.TabularConfig{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("training without a target must be rejected, got %v", err)
	}

	_, err = f.uc.StartForecastTraining("t1", td.ID, domain.ForecastConfig{Horizon: 3})
	if err == nil {
		t.Error("a tabular task cannot start a forecast run")
	}

	_, err = f.uc.UpdateTableMapping("t1", td.ID, usecase.MappingUpdate{Target: "churned"})
	if err != nil {
		t.Fatal(err)
	}

	job, err := f.uc.StartTabularTraining("t1", td.ID, domain.TabularConfig{TimeBudgetSec: 5})
	if err != nil {
		t.Fatal(err)
	}

	if job.Kind != domain.KindTabular || job.TaskID != "t1" || job.Version != 1 {
		t.Fatalf("unexpected job: %+v", job)
	}

	waitJob(t, f.jobs, job.ID)

	done, err := f.jobs.Get(job.ID)
	if err != nil || done.Status != domain.TrainingCompleted || done.Metrics == nil || done.Metrics.Primary == "" || done.Metrics.FeatureSchema == nil {
		t.Fatalf("job did not complete with table metrics: %v %+v", err, done)
	}

	rows, err := f.uc.JobRows(done)
	if err != nil || len(rows) != 60 {
		t.Errorf("JobRows: %v %d", err, len(rows))
	}

	next, err := f.uc.StartTabularTraining("t1", td.ID, domain.TabularConfig{TimeBudgetSec: 5})
	if err != nil || next.Version != 2 {
		t.Errorf("second run should be version 2: %v %+v", err, next)
	}

	waitJob(t, f.jobs, next.ID)
}

func TestTabularUsecase_ForecastNeedsConfirmedFrequency(t *testing.T) {
	t.Parallel()

	f := newTabFixture(t, domain.KindTimeSeries)

	td, err := f.uc.UploadTable("t1", usecase.UploadTableRequest{Filename: "s.csv", Content: salesCSV(80)})
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.uc.StartForecastTraining("t1", td.ID, domain.ForecastConfig{Horizon: 7})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("no timestamp/target mapped: want ErrInvalidInput, got %v", err)
	}

	td, err = f.uc.UpdateTableMapping("t1", td.ID, usecase.MappingUpdate{
		Target: "sales", Timestamp: "day", ItemID: "store", Frequency: "D", Horizon: 7,
	})
	if err != nil {
		t.Fatal(err)
	}

	if !td.FrequencyConfirmed || td.Extra.TimestampColumn != "day" {
		t.Fatalf("mapping not stored: %+v", td)
	}

	_, err = f.uc.StartForecastTraining("t1", td.ID, domain.ForecastConfig{Horizon: 100000})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("an absurd horizon must be rejected, got %v", err)
	}

	_, err = f.uc.StartForecastTraining("t1", td.ID, domain.ForecastConfig{Horizon: 7, Frequency: "X"})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("an unknown frequency must be rejected, got %v", err)
	}

	job, err := f.uc.StartForecastTraining("t1", td.ID, domain.ForecastConfig{Horizon: 7})
	if err != nil {
		t.Fatal(err)
	}

	waitJob(t, f.jobs, job.ID)

	done, _ := f.jobs.Get(job.ID)
	if done.Status != domain.TrainingCompleted || done.Kind != domain.KindTimeSeries {
		t.Fatalf("forecast job: %+v", done)
	}
}

func waitJob(t *testing.T, jobs *memory.TrainingRepo, id string) {
	t.Helper()

	deadline := time.Now().Add(20 * time.Second)

	for time.Now().Before(deadline) {
		j, err := jobs.Get(id)
		if err == nil && (j.Status == domain.TrainingCompleted || j.Status == domain.TrainingFailed) {
			return
		}

		time.Sleep(20 * time.Millisecond)
	}

	t.Fatalf("job %s did not finish", id)
}
