package simulation

import (
	"fmt"
	"math"
	"sort"
	"time"

	"distillery/internal/domain"
)

// Dependency-free statistical forecasters used by the simulation backend and
// its serving engine: seasonal naive, naive, drift and simple exponential
// smoothing, evaluated with a rolling-origin backtest. The richer model set
// (Holt-Winters, statsforecast, LightGBM lags, Chronos) lives in
// trainer/tasks/forecast.py.

const maxGridPoints = 200000

type point struct {
	t time.Time
	v float64
}

type seriesSet struct {
	ids    []string
	series map[string][]point
	filled int
}

// alignTime truncates t to the start of its frequency period.
func alignTime(t time.Time, freq string) time.Time {
	t = t.UTC()

	switch freq {
	case "H":
		return t.Truncate(time.Hour)
	case "W":
		d := t.Truncate(24 * time.Hour)
		back := (int(d.Weekday()) + 6) % 7

		return d.AddDate(0, 0, -back)
	case "M":
		return time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, time.UTC)
	case "Q":
		return time.Date(t.Year(), time.Month((int(t.Month())-1)/3*3+1), 1, 0, 0, 0, 0, time.UTC)
	case "Y":
		return time.Date(t.Year(), 1, 1, 0, 0, 0, 0, time.UTC)
	default:
		return t.Truncate(24 * time.Hour)
	}
}

// stepTime advances an aligned time by n periods.
func stepTime(t time.Time, freq string, n int) time.Time {
	switch freq {
	case "H":
		return t.Add(time.Duration(n) * time.Hour)
	case "W":
		return t.AddDate(0, 0, 7*n)
	case "M":
		return t.AddDate(0, n, 0)
	case "Q":
		return t.AddDate(0, 3*n, 0)
	case "Y":
		return t.AddDate(n, 0, 0)
	default:
		return t.AddDate(0, 0, n)
	}
}

func seasonFor(freq string, override int) int {
	if override > 0 {
		return override
	}

	switch freq {
	case "H":
		return 24
	case "W":
		return 52
	case "M":
		return 12
	case "Q":
		return 4
	case "Y":
		return 1
	default:
		return 7
	}
}

// buildSeries groups rows into per-item regular series on the frequency grid
// (duplicates in a period are averaged, gaps linearly interpolated).
func buildSeries(rows []map[string]interface{}, cfg *domain.ForecastConfig) (*seriesSet, error) {
	type agg struct {
		sum float64
		n   int
	}

	buckets := map[string]map[int64]*agg{}

	for _, r := range rows {
		ts, ok := domain.ParseTableTime(str(r[cfg.Timestamp]))
		if !ok {
			continue
		}

		v, ok := domain.TableToFloat(r[cfg.Target])
		if !ok {
			continue
		}

		id := ""
		if cfg.ItemID != "" {
			id = str(r[cfg.ItemID])
		}

		if buckets[id] == nil {
			buckets[id] = map[int64]*agg{}
		}

		k := alignTime(ts, cfg.Frequency).Unix()
		if buckets[id][k] == nil {
			buckets[id][k] = &agg{}
		}

		buckets[id][k].sum += v
		buckets[id][k].n++
	}

	out := &seriesSet{series: map[string][]point{}}

	for id, b := range buckets {
		keys := make([]int64, 0, len(b))
		for k := range b {
			keys = append(keys, k)
		}

		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

		first, last := time.Unix(keys[0], 0).UTC(), time.Unix(keys[len(keys)-1], 0).UTC()

		var pts []point

		for t := first; !t.After(last); t = stepTime(t, cfg.Frequency, 1) {
			if len(pts) >= maxGridPoints {
				return nil, fmt.Errorf("%w: series %q spans too many %s periods; check the frequency", domain.ErrInvalidInput, id, cfg.Frequency)
			}

			if a := b[t.Unix()]; a != nil {
				pts = append(pts, point{t, a.sum / float64(a.n)})
			} else {
				pts = append(pts, point{t, nan})
			}
		}

		out.filled += interpolate(pts)
		out.series[id] = pts
		out.ids = append(out.ids, id)
	}

	sort.Strings(out.ids)

	if len(out.ids) == 0 {
		return nil, fmt.Errorf("%w: no usable (timestamp, target) rows", domain.ErrInvalidInput)
	}

	return out, nil
}

// interpolate fills NaN gaps linearly; returns how many points it filled.
func interpolate(p []point) int {
	filled := 0

	for i := range p {
		if !math.IsNaN(p[i].v) {
			continue
		}

		lo := i - 1
		hi := i

		for hi < len(p) && math.IsNaN(p[hi].v) {
			hi++
		}

		if lo < 0 || hi >= len(p) { // Leading/trailing NaN cannot occur (grid is data-bounded).
			continue
		}

		frac := float64(i-lo) / float64(hi-lo)
		p[i].v = p[lo].v + frac*(p[hi].v-p[lo].v)
		filled++
	}

	return filled
}

func values(p []point) []float64 {
	y := make([]float64, len(p))
	for i, pt := range p {
		y[i] = pt.v
	}

	return y
}

// ---------- models ----------.

type predictor func(h int) (mean, sigma []float64)

type fmodel struct {
	name string
	fit  func(y []float64, m int) predictor
}

func stddev(x []float64) float64 {
	if len(x) < 2 {
		return 0
	}

	var mean float64
	for _, v := range x {
		mean += v
	}

	mean /= float64(len(x))

	var sq float64
	for _, v := range x {
		sq += (v - mean) * (v - mean)
	}

	return math.Sqrt(sq / float64(len(x)-1))
}

func fill(h int, f func(i int) float64) []float64 {
	out := make([]float64, h)
	for i := range out {
		out[i] = f(i)
	}

	return out
}

var simModels = []fmodel{
	{"seasonal_naive", func(y []float64, m int) predictor {
		if m < 2 || len(y) < m+1 {
			return naiveFit(y)
		}

		res := make([]float64, 0, len(y)-m)
		for t := m; t < len(y); t++ {
			res = append(res, y[t]-y[t-m])
		}

		sd := stddev(res)
		n := len(y)

		return func(h int) ([]float64, []float64) {
			return fill(h, func(i int) float64 { return y[n-m+(i%m)] }),
				fill(h, func(i int) float64 { return sd * math.Sqrt(float64(i/m+1)) })
		}
	}},
	{"naive", func(y []float64, _ int) predictor { return naiveFit(y) }},
	{"drift", func(y []float64, _ int) predictor {
		n := len(y)
		if n < 3 {
			return naiveFit(y)
		}

		slope := (y[n-1] - y[0]) / float64(n-1)
		res := make([]float64, 0, n-1)

		for t := 1; t < n; t++ {
			res = append(res, y[t]-y[t-1]-slope)
		}

		sd := stddev(res)

		return func(h int) ([]float64, []float64) {
			return fill(h, func(i int) float64 { return y[n-1] + slope*float64(i+1) }),
				fill(h, func(i int) float64 { return sd * math.Sqrt(float64(i+1)*(1+float64(i+1)/float64(n))) })
		}
	}},
	{"ses", func(y []float64, _ int) predictor {
		n := len(y)
		if n < 3 {
			return naiveFit(y)
		}

		bestA, bestSSE := 0.5, math.Inf(1)

		for a := 0.1; a < 0.95; a += 0.1 {
			level := y[0]
			sse := 0.0

			for t := 1; t < n; t++ {
				e := y[t] - level
				sse += e * e
				level += a * e
			}

			if sse < bestSSE {
				bestA, bestSSE = a, sse
			}
		}

		level := y[0]
		errs := make([]float64, 0, n-1)

		for t := 1; t < n; t++ {
			e := y[t] - level
			errs = append(errs, e)
			level += bestA * e
		}

		sd := stddev(errs)

		return func(h int) ([]float64, []float64) {
			return fill(h, func(int) float64 { return level }),
				fill(h, func(i int) float64 { return sd * math.Sqrt(1+float64(i)*bestA*bestA) })
		}
	}},
}

func naiveFit(y []float64) predictor {
	n := len(y)
	diffs := make([]float64, 0, n)

	for t := 1; t < n; t++ {
		diffs = append(diffs, y[t]-y[t-1])
	}

	sd := stddev(diffs)

	return func(h int) ([]float64, []float64) {
		return fill(h, func(int) float64 { return y[n-1] }),
			fill(h, func(i int) float64 { return sd * math.Sqrt(float64(i+1)) })
	}
}

func modelByName(name string) *fmodel {
	for i := range simModels {
		if simModels[i].name == name {
			return &simModels[i]
		}
	}

	return nil
}

// ---------- metrics ----------.

const z90 = 1.2815515655446004 // Normal quantile for the 10%/90% interval.

func maseScale(train []float64, m int) float64 {
	lag := m
	if lag < 1 || len(train) <= lag {
		lag = 1
	}

	var s float64

	n := 0

	for t := lag; t < len(train); t++ {
		s += math.Abs(train[t] - train[t-lag])
		n++
	}

	if n == 0 || s == 0 {
		return 1
	}

	return s / float64(n)
}

type scores struct{ mase, smape, wql float64 }

func scoreForecast(train, actual, mean, sigma []float64, m int) scores {
	h := len(actual)
	sc := maseScale(train, m)

	var ae, sm, pin, den float64

	for i := range h {
		e := actual[i] - mean[i]
		ae += math.Abs(e)

		if d := math.Abs(actual[i]) + math.Abs(mean[i]); d > 0 {
			sm += 2 * math.Abs(e) / d
		}

		den += math.Abs(actual[i])

		for _, q := range []struct{ z, tau float64 }{{-z90, 0.1}, {0, 0.5}, {z90, 0.9}} {
			qv := mean[i] + q.z*sigma[i]
			if actual[i] >= qv {
				pin += q.tau * (actual[i] - qv)
			} else {
				pin += (1 - q.tau) * (qv - actual[i])
			}
		}
	}

	s := scores{mase: ae / float64(h) / sc, smape: sm / float64(h)}
	if den > 0 {
		s.wql = 2 * pin / 3 / den
	}

	return s
}

func metricOf(s scores, metric string) float64 {
	switch metric {
	case "smape":
		return s.smape
	case "wql":
		return s.wql
	default:
		return s.mase
	}
}

// ---------- training ----------.

//nolint:cyclop,funlen,gocognit // sequential pipeline; splitting adds indirection without clarity
func trainForecast(job *domain.TrainingJob, rows []map[string]interface{}, onUpdate func(int)) (*domain.TrainingMetrics, error) {
	cfg := job.Forecast
	if cfg == nil || cfg.Target == "" || cfg.Timestamp == "" || cfg.Horizon <= 0 {
		return nil, fmt.Errorf("%w: forecast job is missing timestamp/target/horizon", domain.ErrInvalidInput)
	}

	if cfg.Model == "gbm" || cfg.Model == "chronos" {
		return nil, fmt.Errorf("%w: model %q needs the Python trainer (TRAINING_BACKEND=local)", domain.ErrInvalidInput, cfg.Model)
	}

	set, err := buildSeries(rows, cfg)
	if err != nil {
		return nil, err
	}

	progress(onUpdate, 15)

	m := seasonFor(cfg.Frequency, cfg.SeasonLength)
	h := cfg.Horizon
	windows := max(1, cfg.BacktestWindows)
	metric := cfg.Metric

	if metric == "" {
		metric = "mase"
	}

	models := simModels
	if cfg.Model == "seasonal_naive" {
		models = simModels[:1]
	}

	type key struct {
		model string
		w     int
	}

	agg := map[key][]scores{}
	baseAgg := map[int][]scores{}
	usedSeries := 0

	var plot *domain.BacktestPlot

	firstOrigin := map[int]string{}

	for _, id := range set.ids {
		pts := set.series[id]
		y := values(pts)
		minTrain := max(2*m, 8)

		if len(y) < minTrain+h*windows {
			continue
		}

		usedSeries++

		for w := 1; w <= windows; w++ {
			cut := len(y) - h*(windows-w+1)
			train, actual := y[:cut], y[cut:cut+h]

			if _, ok := firstOrigin[w]; !ok {
				firstOrigin[w] = pts[cut-1].t.Format("2006-01-02T15:04:05")
			}

			bmean, bsig := simModels[0].fit(train, m)(h)
			baseAgg[w] = append(baseAgg[w], scoreForecast(train, actual, bmean, bsig, m))

			for _, md := range models {
				mean, sig := md.fit(train, m)(h)
				agg[key{md.name, w}] = append(agg[key{md.name, w}], scoreForecast(train, actual, mean, sig, m))
			}
		}

		progress(onUpdate, 15+int(60*float64(usedSeries)/float64(len(set.ids))))
	}

	if usedSeries == 0 {
		return nil, fmt.Errorf(
			"%w: no series is long enough for horizon %d × %d backtest windows (needs ≥ %d points)",
			domain.ErrInvalidInput, h, windows, max(2*m, 8)+h*windows)
	}

	mean := func(ss []scores) scores {
		var o scores

		for _, s := range ss {
			o.mase += s.mase
			o.smape += s.smape
			o.wql += s.wql
		}

		n := float64(len(ss))

		return scores{o.mase / n, o.smape / n, o.wql / n}
	}

	board := make([]domain.TabularCandidate, 0, len(models))
	bestName, bestScore := "", math.Inf(1)

	overall := map[string]scores{}

	for _, md := range models {
		var all []scores
		for w := 1; w <= windows; w++ {
			all = append(all, agg[key{md.name, w}]...)
		}

		s := mean(all)
		overall[md.name] = s
		board = append(board, domain.TabularCandidate{
			Model: md.name, Score: round4(metricOf(s, metric)), Metric: metric, Baseline: md.name == "seasonal_naive",
		})

		if v := metricOf(s, metric); v < bestScore {
			bestName, bestScore = md.name, v
		}
	}

	for i := range board {
		board[i].Chosen = board[i].Model == bestName
	}

	var bt []domain.BacktestWindow

	for w := 1; w <= windows; w++ {
		bs := mean(baseAgg[w])

		for _, md := range models {
			s := mean(agg[key{md.name, w}])
			bt = append(bt, domain.BacktestWindow{
				Window: w, Origin: firstOrigin[w], Model: md.name,
				MASE: round4(s.mase), SMAPE: round4(s.smape), WQL: round4(s.wql), SeasonalNaiveMASE: round4(bs.mase),
			})
		}
	}

	// Plot: last window of the first usable series for the chosen model.
	for _, id := range set.ids {
		y := values(set.series[id])
		if len(y) < max(2*m, 8)+h*windows {
			continue
		}

		cut := len(y) - h
		train := y[:cut]
		pm, ps := modelByName(bestName).fit(train, m)(h)
		sn, _ := simModels[0].fit(train, m)(h)
		pl := &domain.BacktestPlot{SeasonalNaive: sn, Actual: y[cut:], Predicted: pm}

		for i := range h {
			pl.Timestamps = append(pl.Timestamps, set.series[id][cut+i].t.Format("2006-01-02T15:04:05"))
			pl.Lower = append(pl.Lower, pm[i]-z90*ps[i])
			pl.Upper = append(pl.Upper, pm[i]+z90*ps[i])
		}

		plot = pl

		break
	}

	baseVal := metricOf(overall["seasonal_naive"], metric)

	return &domain.TrainingMetrics{
		TrainExamples:  len(rows),
		Epochs:         1,
		FinalLoss:      round4(bestScore),
		Primary:        metric,
		PrimaryVal:     round4(bestScore),
		HigherIsBetter: false,
		Baselines: map[string]float64{
			"seasonal_naive_" + metric: round4(baseVal),
		},
		Improvement:  round4(baseVal - bestScore),
		TableBackend: "simulation-stats",
		Leaderboard:  board,
		Backtest:     bt,
		BacktestPlot: plot,
		ChosenModel:  bestName,
		SeasonLength: m,
	}, nil
}

// ---------- serving ----------.

// forecastFromRows refits the chosen model on the stored history (plus any
// caller history) and forecasts h steps with 80% intervals.
func forecastFromRows(
	job *domain.TrainingJob, rows []map[string]interface{}, req domain.ForecastRequest,
) (domain.ForecastResult, error) {
	cfg := job.Forecast
	if cfg == nil || job.Metrics == nil {
		return domain.ForecastResult{}, fmt.Errorf("%w: not a trained forecasting model", domain.ErrInvalidInput)
	}

	all := rows

	if len(req.History) > 0 {
		// Caller history replaces stored observations of the same period.
		override := map[string]bool{}

		for _, r := range req.History {
			if ts, ok := domain.ParseTableTime(str(r[cfg.Timestamp])); ok {
				override[historyKey(cfg, r, ts)] = true
			}
		}

		all = make([]map[string]interface{}, 0, len(rows)+len(req.History))

		for _, r := range rows {
			if ts, ok := domain.ParseTableTime(str(r[cfg.Timestamp])); ok && override[historyKey(cfg, r, ts)] {
				continue
			}

			all = append(all, r)
		}

		all = append(all, req.History...)
	}

	set, err := buildSeries(all, cfg)
	if err != nil {
		return domain.ForecastResult{}, err
	}

	id := req.ItemID

	switch {
	case id == "" && len(set.ids) == 1:
		id = set.ids[0]
	case id == "":
		return domain.ForecastResult{}, fmt.Errorf(
			"%w: this model has %d series; pass \"item_id\" (e.g. %q)", domain.ErrInvalidInput, len(set.ids), set.ids[0])
	}

	pts, ok := set.series[id]
	if !ok {
		return domain.ForecastResult{}, fmt.Errorf("%w: unknown item_id %q", domain.ErrInvalidInput, id)
	}

	md := modelByName(job.Metrics.ChosenModel)
	if md == nil {
		md = &simModels[0]
	}

	m := seasonFor(cfg.Frequency, cfg.SeasonLength)
	mean, sig := md.fit(values(pts), m)(req.Horizon)
	last := pts[len(pts)-1].t
	res := domain.ForecastResult{
		ItemID: id, Model: md.name, Metric: job.Metrics.Primary, MetricValue: job.Metrics.PrimaryVal,
		Baselines: job.Metrics.Baselines,
	}

	for i := range req.Horizon {
		res.Forecast = append(res.Forecast, domain.ForecastPoint{
			Timestamp: stepTime(last, cfg.Frequency, i+1).Format("2006-01-02T15:04:05"),
			Value:     round4(mean[i]),
			Lower:     round4(mean[i] - z90*sig[i]),
			Upper:     round4(mean[i] + z90*sig[i]),
		})
	}

	return res, nil
}

func historyKey(cfg *domain.ForecastConfig, r map[string]interface{}, ts time.Time) string {
	id := ""
	if cfg.ItemID != "" {
		id = str(r[cfg.ItemID])
	}

	return id + "|" + alignTime(ts, cfg.Frequency).Format(time.RFC3339)
}
