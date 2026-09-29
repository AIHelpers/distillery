package simulation

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"distillery/internal/domain"
)

// This file holds the pieces shared by the simulation backend's tabular
// trainer and serving engine: feature-schema construction, the distance-
// weighted k-nearest-neighbour model, splits and metrics. The simulation
// backend is a deliberately small, dependency-free stand-in for the GBM
// trainer (trainer/tasks/tabular.py): its numbers are REAL holdout scores of
// the very model that is served, never synthetic.

const (
	maxKNNRows     = 5000
	maxEvalRows    = 600
	maxCategories  = 64
	maxZ           = 3.0
	maxExplain     = 5
	simHoldoutFrac = 0.2
)

func round4(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}

	return math.Round(f*10000) / 10000
}

func str(v interface{}) string {
	if v == nil {
		return ""
	}

	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}

	return strings.TrimSpace(fmt.Sprintf("%v", v))
}

// buildSchema derives the deployed feature schema from the training rows and
// the job's mapping (feature columns = cfg.Columns minus target/exclusions).
func buildSchema(rows []map[string]interface{}, cfg *domain.TabularConfig, classes []string) *domain.FeatureSchema {
	excluded := map[string]bool{cfg.Target: true}
	for _, c := range cfg.ExcludeColumns {
		excluded[c] = true
	}

	names := make([]string, 0, len(cfg.Columns))

	for n, t := range cfg.Columns {
		if excluded[n] || t == domain.ColID || t == domain.ColIgnored || t == domain.ColText {
			continue
		}

		names = append(names, n)
	}

	sort.Strings(names)

	feats := make([]domain.TableFeature, 0, len(names))

	for _, name := range names {
		f := domain.TableFeature{Name: name}

		missing := false
		counts := map[string]int{}

		var lo, hi *float64

		for _, r := range rows {
			s := str(r[name])
			if s == "" {
				missing = true

				continue
			}

			counts[s]++

			if v, ok := domain.TableToFloat(s); ok {
				if lo == nil || v < *lo {
					x := v
					lo = &x
				}

				if hi == nil || v > *hi {
					x := v
					hi = &x
				}
			}
		}

		f.Optional = missing

		switch cfg.Columns[name] {
		case domain.ColNumeric:
			f.Type, f.Min, f.Max = domain.FeatureNumeric, lo, hi
		case domain.ColDatetime:
			f.Type = domain.FeatureDatetime
		case domain.ColCategorical, domain.ColText, domain.ColID, domain.ColIgnored:
			f.Type = domain.FeatureCategorical
			f.Categories, f.OpenVocabulary = topCategories(counts)
		}

		feats = append(feats, f)
	}

	return &domain.FeatureSchema{Features: feats, Target: cfg.Target, Task: cfg.Task, Classes: classes}
}

// topCategories returns up to maxCategories most frequent values (sorted by
// frequency, then name) and whether the vocabulary was truncated.
func topCategories(counts map[string]int) (cats []string, truncated bool) {
	for k := range counts {
		cats = append(cats, k)
	}

	sort.Slice(cats, func(i, j int) bool {
		if counts[cats[i]] != counts[cats[j]] {
			return counts[cats[i]] > counts[cats[j]]
		}

		return cats[i] < cats[j]
	})

	if len(cats) > maxCategories {
		return cats[:maxCategories], true
	}

	return cats, false
}

// ---------- k-nearest-neighbour model ----------.

type knnFeature struct {
	name string
	typ  domain.TableFeatureType
	mean float64
	std  float64
	cats map[string]int
}

type knnModel struct {
	feats      []knnFeature
	X          [][]float64
	yClass     []int
	yVal       []float64
	classes    []string
	regression bool
	weights    []float64
}

var nan = math.NaN()

// buildKNN encodes rows for scoring. weights (by feature name) may be nil
// (uniform). Rows without a usable target are dropped; rows beyond
// maxKNNRows are stride-sampled.
func buildKNN( //nolint:gocognit,cyclop // sequential pipeline; splitting adds indirection without clarity
	rows []map[string]interface{}, schema *domain.FeatureSchema, weights map[string]float64,
) *knnModel {
	m := &knnModel{regression: schema.Task == domain.TabularRegression, classes: schema.Classes}

	classIdx := map[string]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}

	stride := 1
	if len(rows) > maxKNNRows {
		stride = len(rows) / maxKNNRows
	}

	var kept []map[string]interface{}

	for i := 0; i < len(rows) && len(kept) < maxKNNRows; i += stride {
		t := str(rows[i][schema.Target])
		if t == "" {
			continue
		}

		if m.regression {
			v, ok := domain.TableToFloat(t)
			if !ok {
				continue
			}

			m.yVal = append(m.yVal, v)
		} else {
			ci, ok := classIdx[t]
			if !ok {
				continue
			}

			m.yClass = append(m.yClass, ci)
		}

		kept = append(kept, rows[i])
	}

	for _, f := range schema.Features {
		kf := knnFeature{name: f.Name, typ: f.Type, std: 1}

		if f.Type == domain.FeatureCategorical {
			kf.cats = map[string]int{}
			for i, c := range f.Categories {
				kf.cats[strings.ToLower(c)] = i
			}
		}

		m.feats = append(m.feats, kf)
	}

	m.X = make([][]float64, len(kept))

	for i, r := range kept {
		m.X[i] = m.encodeMap(r)
	}

	for fi := range m.feats {
		if m.feats[fi].typ != domain.FeatureNumeric {
			continue
		}

		var sum float64

		n := 0

		for _, x := range m.X {
			if !math.IsNaN(x[fi]) {
				sum += x[fi]
				n++
			}
		}

		if n == 0 {
			continue
		}

		mean := sum / float64(n)

		var sq float64

		for _, x := range m.X {
			if !math.IsNaN(x[fi]) {
				sq += (x[fi] - mean) * (x[fi] - mean)
			}
		}

		m.feats[fi].mean = mean
		if n > 1 && sq > 0 {
			m.feats[fi].std = math.Sqrt(sq / float64(n-1))
		}
	}

	m.setWeights(weights)

	return m
}

func (m *knnModel) setWeights(weights map[string]float64) {
	m.weights = make([]float64, len(m.feats))

	var sum float64

	for i, f := range m.feats {
		w := 1.0

		if weights != nil {
			w = 0.05
			if v, ok := weights[f.name]; ok && v > 0 {
				w = v
			}
		}

		if f.typ == domain.FeatureDatetime {
			w = 0
		}

		m.weights[i] = w
		sum += w
	}

	if sum > 0 {
		for i := range m.weights {
			m.weights[i] /= sum
		}
	}
}

// encodeMap encodes a raw row (string values from the table).
func (m *knnModel) encodeMap(r map[string]interface{}) []float64 {
	x := make([]float64, len(m.feats))

	for i, f := range m.feats {
		s := str(r[f.name])
		x[i] = m.encodeValue(f, s)
	}

	return x
}

func (m *knnModel) encodeValue(f knnFeature, s string) float64 {
	if s == "" {
		return nan
	}

	switch f.typ {
	case domain.FeatureNumeric:
		if v, ok := domain.TableToFloat(s); ok {
			return v
		}

		return nan
	case domain.FeatureCategorical:
		if i, ok := f.cats[strings.ToLower(s)]; ok {
			return float64(i)
		}

		return float64(len(f.cats)) // unseen -> "other".
	case domain.FeatureDatetime:
	}

	return nan
}

// encodeValidated encodes an API input row.
func (m *knnModel) encodeValidated(v *domain.ValidatedRow) []float64 {
	x := make([]float64, len(m.feats))

	for i, f := range m.feats {
		x[i] = nan

		switch f.typ {
		case domain.FeatureNumeric:
			if val, ok := v.Numerics[f.name]; ok {
				x[i] = val
			}
		case domain.FeatureCategorical:
			if c, ok := v.Cats[f.name]; ok {
				x[i] = m.encodeValue(f, c)
			}
		case domain.FeatureDatetime:
		}
	}

	return x
}

func (m *knnModel) distance(a, b []float64, skip int) float64 {
	var total float64

	for i, f := range m.feats {
		w := m.weights[i]
		if w == 0 || i == skip {
			continue
		}

		switch {
		case math.IsNaN(a[i]) != math.IsNaN(b[i]):
			total += w
		case math.IsNaN(a[i]):
		case f.typ == domain.FeatureNumeric:
			z := math.Max(-maxZ, math.Min(maxZ, (a[i]-b[i])/f.std))
			total += w * z * z
		case a[i] != b[i]:
			total += w
		}
	}

	return math.Sqrt(total)
}

type nb struct {
	dist float64
	idx  int
}

// neighbours returns the k nearest training rows (excluding index `exclude`
// for leave-one-out style evaluation).
func (m *knnModel) neighbours(x []float64, k, skip, exclude int) []nb {
	all := make([]nb, 0, len(m.X))

	for i, row := range m.X {
		if i == exclude {
			continue
		}

		all = append(all, nb{dist: m.distance(x, row, skip), idx: i})
	}

	sort.SliceStable(all, func(i, j int) bool { return all[i].dist < all[j].dist })

	if k > len(all) {
		k = len(all)
	}

	return all[:k]
}

// score returns class probabilities (classification) or the weighted mean
// (regression) for an encoded row.
func (m *knnModel) score(x []float64, k, skip int) (probs []float64, value float64) {
	ns := m.neighbours(x, k, skip, -1)

	var wsum float64

	if m.regression {
		var vsum float64

		for _, n := range ns {
			w := 1 / (n.dist*n.dist + 1e-6)
			wsum += w
			vsum += w * m.yVal[n.idx]
		}

		if wsum == 0 {
			return nil, 0
		}

		return nil, vsum / wsum
	}

	probs = make([]float64, len(m.classes))

	for _, n := range ns {
		w := 1 / (n.dist*n.dist + 1e-6)
		wsum += w
		probs[m.yClass[n.idx]] += w
	}

	if wsum > 0 {
		for i := range probs {
			probs[i] /= wsum
		}
	}

	return probs, 0
}

func argmax(p []float64) int {
	best := 0

	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}

	return best
}

// ---------- splits ----------.

// lcg is a tiny deterministic generator (reproducible splits without pulling
// math/rand's global state).
type lcg uint64

func (l *lcg) next() uint64 {
	*l = *l*6364136223846793005 + 1442695040888963407

	return uint64(*l >> 11)
}

func shuffled(n int, seed uint64) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}

	g := lcg(seed)

	for i := n - 1; i > 0; i-- {
		j := int(g.next() % uint64(i+1)) //nolint:gosec // i+1 is a positive slice length
		idx[i], idx[j] = idx[j], idx[i]
	}

	return idx
}

func fnv(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}

	return h
}

// splitIndices divides row indices into train / holdout following the job's
// split strategy. It returns the strategy actually used.
//
//nolint:cyclop,gocognit // sequential pipeline; splitting adds indirection without clarity
func splitIndices(rows []map[string]interface{}, cfg *domain.TabularConfig) (train, hold []int, used domain.SplitStrategy) {
	n := len(rows)
	used = cfg.SplitStrategy

	if used == "" {
		used = domain.SplitRandom
	}

	order := shuffled(n, 42)
	cut := int(float64(n) * (1 - simHoldoutFrac))

	switch used {
	case domain.SplitTime:
		col := ""

		for name, t := range cfg.Columns {
			if t == domain.ColDatetime && name != cfg.Target && (col == "" || name < col) {
				col = name
			}
		}

		if col != "" {
			order = make([]int, n)
			for i := range order {
				order[i] = i
			}

			sort.SliceStable(order, func(a, b int) bool { return str(rows[order[a]][col]) < str(rows[order[b]][col]) })
		}
	case domain.SplitGroup:
		g := cfg.GroupColumn

		for _, i := range order {
			if fnv(str(rows[i][g]))%5 == 0 {
				hold = append(hold, i)
			} else {
				train = append(train, i)
			}
		}

		if len(hold) > 0 && len(train) > 0 {
			return train, hold, used
		}
	case domain.SplitStratified:
		seen := map[string]int{}

		for _, i := range order {
			c := str(rows[i][cfg.Target])
			seen[c]++

			if seen[c]%5 == 0 {
				hold = append(hold, i)
			} else {
				train = append(train, i)
			}
		}

		if len(hold) > 0 && len(train) > 0 {
			return train, hold, used
		}
	case domain.SplitRandom:
	}

	return order[:cut], order[cut:], used
}

// ---------- metrics ----------.

func rocAUC(pos []bool, s []float64) float64 {
	type p struct {
		s   float64
		pos bool
	}

	ps := make([]p, len(s))
	nPos := 0

	for i := range s {
		ps[i] = p{s[i], pos[i]}
		if pos[i] {
			nPos++
		}
	}

	nNeg := len(ps) - nPos
	if nPos == 0 || nNeg == 0 {
		return 0.5
	}

	sort.Slice(ps, func(i, j int) bool { return ps[i].s < ps[j].s })

	var rankSum float64

	for i := 0; i < len(ps); {
		j := i
		for j+1 < len(ps) && ps[j+1].s == ps[i].s {
			j++
		}

		avg := float64(i+j+2) / 2

		for k := i; k <= j; k++ {
			if ps[k].pos {
				rankSum += avg
			}
		}

		i = j + 1
	}

	return (rankSum - float64(nPos)*float64(nPos+1)/2) / (float64(nPos) * float64(nNeg))
}

func averagePrecision(pos []bool, s []float64) float64 {
	idx := make([]int, len(s))
	nPos := 0

	for i := range idx {
		idx[i] = i

		if pos[i] {
			nPos++
		}
	}

	if nPos == 0 {
		return 0
	}

	sort.SliceStable(idx, func(a, b int) bool { return s[idx[a]] > s[idx[b]] })

	tp, ap := 0.0, 0.0

	for r, i := range idx {
		if pos[i] {
			tp++
			ap += tp / float64(r+1)
		}
	}

	return ap / float64(nPos)
}

// classMetrics scores class-probability predictions on the holdout.
func classMetrics(y []int, probs [][]float64, nClasses int) map[string]float64 {
	out := map[string]float64{}
	n := float64(len(y))

	if n == 0 {
		return out
	}

	correct := 0
	ll := 0.0
	f1s := make([]float64, nClasses)
	tp := make([]float64, nClasses)
	fp := make([]float64, nClasses)
	fn := make([]float64, nClasses)

	for i, yi := range y {
		pi := argmax(probs[i])
		if pi == yi {
			correct++
			tp[yi]++
		} else {
			fp[pi]++
			fn[yi]++
		}

		ll -= math.Log(math.Max(1e-9, probs[i][yi]))
	}

	present := 0

	for c := range nClasses {
		if tp[c]+fn[c] == 0 {
			continue
		}

		present++

		if d := 2*tp[c] + fp[c] + fn[c]; d > 0 {
			f1s[c] = 2 * tp[c] / d
		}
	}

	var f1sum float64
	for _, f := range f1s {
		f1sum += f
	}

	out["accuracy"] = float64(correct) / n
	out["logloss"] = ll / n
	out["f1"] = f1sum / math.Max(1, float64(present))

	// AUC / PR-AUC: binary uses the positive class (index 1), multiclass is a
	// macro one-vs-rest average over classes present in the holdout.
	var aucSum, apSum float64

	cnt := 0

	for c := range nClasses {
		if nClasses == 2 && c == 0 {
			continue
		}

		pos := make([]bool, len(y))
		sc := make([]float64, len(y))
		anyPos := false

		for i, yi := range y {
			pos[i] = yi == c
			sc[i] = probs[i][c]
			anyPos = anyPos || pos[i]
		}

		if !anyPos {
			continue
		}

		aucSum += rocAUC(pos, sc)
		apSum += averagePrecision(pos, sc)
		cnt++
	}

	if cnt > 0 {
		out["roc_auc"] = aucSum / float64(cnt)
		out["pr_auc"] = apSum / float64(cnt)
	} else {
		out["roc_auc"], out["pr_auc"] = 0.5, 0
	}

	return out
}

func regMetrics(y, pred []float64) map[string]float64 {
	out := map[string]float64{}
	n := float64(len(y))

	if n == 0 {
		return out
	}

	var mean float64
	for _, v := range y {
		mean += v
	}

	mean /= n

	var se, ae, st float64

	for i := range y {
		d := y[i] - pred[i]
		se += d * d
		ae += math.Abs(d)
		st += (y[i] - mean) * (y[i] - mean)
	}

	out["rmse"] = math.Sqrt(se / n)
	out["mae"] = ae / n

	if st > 0 {
		out["r2"] = 1 - se/st
	}

	return out
}

func higherIsBetter(metric string) bool {
	switch metric {
	case "rmse", "mae", "logloss":
		return false
	}

	return true
}
