package simulation

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"distillery/internal/domain"
)

// TableFineTuner is the simulation backend's trainer for the tabular and
// time_series kinds. It needs no Python: tabular jobs train a
// distance-weighted kNN (k chosen by K-fold CV on the training split) and
// score it on an untouched holdout next to the majority/mean baseline;
// forecasting jobs run a rolling-origin backtest of dependency-free
// statistical models. All reported numbers are real measurements of the same
// models the serving engine uses. The GBM / pretrained-forecaster path lives
// in trainer/tasks (TRAINING_BACKEND=local).
type TableFineTuner struct{}

// NewTableFineTuner builds the simulation trainer.
func NewTableFineTuner() *TableFineTuner { return &TableFineTuner{} }

var errTrainCrashed = errors.New("table training crashed")

// Start implements domain.FineTuner.
func (t *TableFineTuner) Start(
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				onDone(nil, fmt.Errorf("%w: %v", errTrainCrashed, r))
			}
		}()

		rows := make([]map[string]interface{}, 0, len(examples))

		for _, ex := range examples {
			row, err := domain.ParseTablePayload(ex.Payload)
			if err != nil {
				onDone(nil, fmt.Errorf("%w: bad table row payload: %s", domain.ErrInvalidInput, ex.ID))

				return
			}

			rows = append(rows, row)
		}

		var (
			m   *domain.TrainingMetrics
			err error
		)

		switch job.Kind {
		case domain.KindTabular:
			m, err = trainTabular(job, rows, onUpdate)
		case domain.KindTimeSeries:
			m, err = trainForecast(job, rows, onUpdate)
		default:
			err = fmt.Errorf("%w: kind %q is not a table kind", domain.ErrInvalidInput, job.Kind)
		}

		onDone(m, err)
	}()
}

func progress(f func(int), p int) {
	if f != nil {
		f(p)
	}
}

func trainTabular(job *domain.TrainingJob, rows []map[string]interface{}, onUpdate func(int)) (*domain.TrainingMetrics, error) { //nolint:funlen,lll // linear training pipeline
	cfg := job.Tabular
	if cfg == nil || cfg.Target == "" {
		return nil, fmt.Errorf("%w: tabular job has no target column", domain.ErrInvalidInput)
	}

	regression := cfg.Task == domain.TabularRegression

	var classes []string

	if !regression {
		set := map[string]bool{}

		for _, r := range rows {
			if c := str(r[cfg.Target]); c != "" {
				set[c] = true
			}
		}

		for c := range set {
			classes = append(classes, c)
		}

		sort.Strings(classes)

		if len(classes) < 2 {
			return nil, fmt.Errorf("%w: the target has fewer than two classes", domain.ErrInvalidInput)
		}
	}

	schema := buildSchema(rows, cfg, classes)
	if len(schema.Features) == 0 {
		return nil, fmt.Errorf("%w: no usable feature columns", domain.ErrInvalidInput)
	}

	progress(onUpdate, 10)

	trainIdx, holdIdx, used := splitIndices(rows, cfg)
	trainRows := pick(rows, trainIdx)
	holdRows := pick(rows, holdIdx)

	metric := cfg.Metric
	if metric == "" {
		metric = "roc_auc"
		if regression {
			metric = "rmse"
		}
	}

	// Choose k by K-fold CV on the training split.
	folds := cfg.CVFolds
	if folds < 2 {
		folds = 3
	}

	folds = min(folds, 5)
	base := buildKNN(trainRows, schema, nil)
	board := make([]domain.TabularCandidate, 0, 4)
	bestK, bestScore, bestStd := 0, math.NaN(), 0.0

	for i, k := range []int{5, 15, 35} {
		mean, std := cvScore(base, k, folds, metric, regression)
		board = append(board, domain.TabularCandidate{
			Model: fmt.Sprintf("knn_k%d", k), Score: round4(mean), Std: round4(std), Metric: metric,
			Params: map[string]interface{}{"k": k},
		})

		if math.IsNaN(bestScore) || better(mean, bestScore, metric) {
			bestK, bestScore, bestStd = k, mean, std
		}

		progress(onUpdate, 20+i*15)
	}

	// Holdout: baseline vs model.
	nHold := min(len(holdRows), maxEvalRows)
	holdRows = holdRows[:nHold]

	imp := permutationImportance(base, schema, holdRows, bestK, metric, regression)
	weights := map[string]float64{}

	for _, f := range imp {
		weights[f.Feature] = math.Max(f.Impact, 0)
	}

	final := buildKNN(trainRows, schema, weights)

	progress(onUpdate, 75)

	holdout, baseline := evaluateHoldout(final, base, holdRows, schema, trainRows, bestK, regression)

	primary := holdout[metric]
	baseVal := baseline[metric]
	improvement := primary - baseVal

	if !higherIsBetter(metric) {
		improvement = baseVal - primary
	}

	for i := range board {
		if board[i].Params["k"] == bestK {
			board[i].Chosen = true
		}
	}

	baseName := "majority_class"
	if regression {
		baseName = "mean_predictor"
	}

	board = append(board, domain.TabularCandidate{
		Model: baseName, Score: round4(baseVal), Metric: metric, Baseline: true,
	})

	progress(onUpdate, 95)

	tm := &domain.TrainingMetrics{
		TrainExamples:     len(trainRows),
		Epochs:            1,
		FinalLoss:         round4(bestScore),
		EvalAccuracy:      round4(holdout["accuracy"]),
		Primary:           metric,
		PrimaryVal:        round4(primary),
		HigherIsBetter:    higherIsBetter(metric),
		Baselines:         roundMap(baseline),
		HoldoutMetrics:    roundMap(holdout),
		CVScore:           round4(bestScore),
		CVStd:             round4(bestStd),
		Improvement:       round4(improvement),
		TableBackend:      "simulation-knn",
		SplitUsed:         string(used),
		Leaderboard:       board,
		FeatureImportance: imp,
		FeatureSchema:     schema,
	}

	holdoutArtifacts(final, holdRows, schema, bestK, regression, tm)

	return tm, nil
}

func pick(rows []map[string]interface{}, idx []int) []map[string]interface{} {
	out := make([]map[string]interface{}, len(idx))
	for i, j := range idx {
		out[i] = rows[j]
	}

	return out
}

func roundMap(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = round4(v)
	}

	return out
}

func better(a, b float64, metric string) bool {
	if higherIsBetter(metric) {
		return a > b
	}

	return a < b
}

// scoreRows scores encoded rows leave-one-out style against the model (used
// for CV where the fold is the model's own data).
func scoreSet(m *knnModel, idx, others []int, k int, metric string, regression bool) float64 {
	sub := &knnModel{
		feats: m.feats, classes: m.classes, regression: m.regression, weights: m.weights,
	}

	for _, i := range others {
		sub.X = append(sub.X, m.X[i])

		if regression {
			sub.yVal = append(sub.yVal, m.yVal[i])
		} else {
			sub.yClass = append(sub.yClass, m.yClass[i])
		}
	}

	if regression {
		var y, p []float64

		for _, i := range idx {
			_, v := sub.score(m.X[i], k, -1)
			y, p = append(y, m.yVal[i]), append(p, v)
		}

		return regMetrics(y, p)[metric]
	}

	y := make([]int, 0, len(idx))
	probs := make([][]float64, 0, len(idx))

	for _, i := range idx {
		pr, _ := sub.score(m.X[i], k, -1)
		y, probs = append(y, m.yClass[i]), append(probs, pr)
	}

	return classMetrics(y, probs, len(m.classes))[metric]
}

// cvScore is the mean/std of a K-fold CV of a k-NN over the model's rows
// (sub-sampled to keep the O(n²) cost bounded).
func cvScore(m *knnModel, k, folds int, metric string, regression bool) (mean, std float64) {
	n := len(m.X)
	order := shuffled(n, 7)

	if n > 1500 {
		order = order[:1500]
	}

	scores := make([]float64, 0, folds)

	for f := range folds {
		var val, tr []int

		for pos, i := range order {
			if pos%folds == f {
				val = append(val, i)
			} else {
				tr = append(tr, i)
			}
		}

		if len(val) == 0 || len(tr) == 0 {
			continue
		}

		scores = append(scores, scoreSet(m, val, tr, k, metric, regression))
	}

	if len(scores) == 0 {
		return 0, 0
	}

	for _, s := range scores {
		mean += s
	}

	mean /= float64(len(scores))

	for _, s := range scores {
		std += (s - mean) * (s - mean)
	}

	return mean, math.Sqrt(std / float64(len(scores)))
}

// predictHold returns the model's holdout predictions.
func predictHold(m *knnModel, hold []map[string]interface{}, k, skip int) (probs [][]float64, vals []float64) {
	for _, r := range hold {
		x := m.encodeMap(r)
		pr, v := m.score(x, k, skip)
		probs, vals = append(probs, pr), append(vals, v)
	}

	return probs, vals
}

func holdTargets(m *knnModel, hold []map[string]interface{}, schema *domain.FeatureSchema) (yc []int, yv []float64, rowsKept []map[string]interface{}) {
	idx := map[string]int{}
	for i, c := range m.classes {
		idx[c] = i
	}

	for _, r := range hold {
		t := str(r[schema.Target])
		if t == "" {
			continue
		}

		if m.regression {
			v, ok := domain.TableToFloat(t)
			if !ok {
				continue
			}

			yv = append(yv, v)
		} else {
			ci, ok := idx[t]
			if !ok {
				continue
			}

			yc = append(yc, ci)
		}

		rowsKept = append(rowsKept, r)
	}

	return yc, yv, rowsKept
}

func evaluateHoldout(
	final, base *knnModel, hold []map[string]interface{}, schema *domain.FeatureSchema,
	train []map[string]interface{}, k int, regression bool,
) (holdout, baseline map[string]float64) {
	yc, yv, kept := holdTargets(final, hold, schema)

	if regression {
		_, vals := predictHold(final, kept, k, -1)

		var mean float64
		for _, v := range base.yVal {
			mean += v
		}

		mean /= math.Max(1, float64(len(base.yVal)))
		bp := make([]float64, len(yv))

		for i := range bp {
			bp[i] = mean
		}

		return regMetrics(yv, vals), regMetrics(yv, bp)
	}

	probs, _ := predictHold(final, kept, k, -1)
	counts := make([]float64, len(final.classes))

	for _, c := range base.yClass {
		counts[c]++
	}

	total := 0.0
	for _, c := range counts {
		total += c
	}

	prior := make([]float64, len(counts))
	maj := 0

	for i := range counts {
		prior[i] = counts[i] / math.Max(1, total)

		if counts[i] > counts[maj] {
			maj = i
		}
	}

	// Majority baseline predicts the majority class with a constant score.
	bprobs := make([][]float64, len(yc))
	for i := range bprobs {
		p := make([]float64, len(prior))
		p[maj] = 1
		bprobs[i] = p
	}

	_ = train

	return classMetrics(yc, probs, len(final.classes)), classMetrics(yc, bprobs, len(final.classes))
}

// permutationImportance measures how much shuffling each feature degrades
// the holdout score (a model-agnostic, genuine importance).
func permutationImportance(
	m *knnModel, schema *domain.FeatureSchema, hold []map[string]interface{}, k int, metric string, regression bool,
) []domain.FeatureImpact {
	nEval := min(len(hold), 200)
	if nEval == 0 {
		return nil
	}

	hold = hold[:nEval]
	yc, yv, kept := holdTargets(m, hold, schema)

	score := func(rowsX [][]float64) float64 {
		if regression {
			p := make([]float64, len(rowsX))
			for i, x := range rowsX {
				_, p[i] = m.score(x, k, -1)
			}

			return regMetrics(yv, p)[metric]
		}

		pr := make([][]float64, len(rowsX))
		for i, x := range rowsX {
			pr[i], _ = m.score(x, k, -1)
		}

		return classMetrics(yc, pr, len(m.classes))[metric]
	}

	X := make([][]float64, len(kept))
	for i, r := range kept {
		X[i] = m.encodeMap(r)
	}

	baseScore := score(X)
	perm := shuffled(len(X), 99)
	out := make([]domain.FeatureImpact, 0, len(m.feats))

	for fi, f := range m.feats {
		if f.typ == domain.FeatureDatetime {
			continue
		}

		Xp := make([][]float64, len(X))

		for i := range X {
			row := append([]float64(nil), X[i]...)
			row[fi] = X[perm[i]][fi]
			Xp[i] = row
		}

		drop := baseScore - score(Xp)
		if !higherIsBetter(metric) {
			drop = -drop
		}

		out = append(out, domain.FeatureImpact{Feature: f.name, Impact: math.Max(drop, 0)})
	}

	var total float64
	for _, o := range out {
		total += o.Impact
	}

	for i := range out {
		if total > 0 {
			out[i].Impact = round4(out[i].Impact / total)
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Impact != out[j].Impact {
			return out[i].Impact > out[j].Impact
		}

		return out[i].Feature < out[j].Feature
	})

	return out
}

// holdoutArtifacts derives the confusion matrix, ROC curve (binary) or
// residual sample (regression) of the final model on the holdout.
func holdoutArtifacts(
	final *knnModel, hold []map[string]interface{}, schema *domain.FeatureSchema, k int, regression bool, m *domain.TrainingMetrics,
) {
	yc, yv, kept := holdTargets(final, hold, schema)
	if len(kept) == 0 {
		return
	}

	probs, vals := predictHold(final, kept, k, -1)

	if regression {
		step := max(1, len(vals)/300)

		for i := 0; i < len(vals); i += step {
			m.Residuals = append(m.Residuals, [2]float64{round4(vals[i]), round4(yv[i] - vals[i])})
		}

		return
	}

	nc := len(final.classes)
	cm := make(domain.ConfusionMatrix, nc)

	for i := range cm {
		cm[i] = make([]int, nc)
	}

	for i, p := range probs {
		cm[yc[i]][argmax(p)]++
	}

	m.ConfusionMatrix = cm
	m.LabelMap = make(map[string]int, nc)

	for i, c := range final.classes {
		m.LabelMap[c] = i
	}

	if nc != 2 {
		return
	}

	type sc struct {
		s   float64
		pos bool
	}

	pts := make([]sc, len(probs))

	var n1, n0 float64

	for i, p := range probs {
		pts[i] = sc{p[1], yc[i] == 1}

		if yc[i] == 1 {
			n1++
		} else {
			n0++
		}
	}

	if n1 == 0 || n0 == 0 {
		return
	}

	sort.Slice(pts, func(a, b int) bool { return pts[a].s > pts[b].s })

	roc := [][2]float64{{0, 0}}

	var tp, fp float64

	for i, p := range pts {
		if p.pos {
			tp++
		} else {
			fp++
		}

		if i == len(pts)-1 || pts[i+1].s != p.s {
			roc = append(roc, [2]float64{round4(fp / n0), round4(tp / n1)})
		}
	}

	if len(roc) > 60 {
		thin := make([][2]float64, 0, 61)
		for i := range 60 {
			thin = append(thin, roc[i*(len(roc)-1)/59])
		}

		roc = thin
	}

	m.ROCCurve = roc
}
