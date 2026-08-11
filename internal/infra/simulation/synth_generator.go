package simulation

import (
	"fmt"
	"math/rand"

	"distillery/internal/domain"
)

// SyntheticGenerator implements domain.SyntheticGenerator. In production this
// would call a frontier model (e.g. the Claude API) to bootstrap/augment
// examples from a handful of seeds; here it produces clearly-labeled
// template-based variations so the full pipeline can be exercised offline.
type SyntheticGenerator struct {
	rand *rand.Rand
}

func NewSyntheticGenerator() *SyntheticGenerator {
	return &SyntheticGenerator{rand: rand.New(rand.NewSource(42))}
}

var variationPrefixes = []string{
	"", "Please note: ", "FYI - ", "Update: ", "Re: ", "Regarding this: ",
}

func (g *SyntheticGenerator) Generate(task *domain.Task, seed []*domain.Example, count int) []*domain.Example {
	if len(seed) == 0 || count <= 0 {
		return nil
	}
	out := make([]*domain.Example, 0, count)
	for i := 0; i < count; i++ {
		base := seed[g.rand.Intn(len(seed))]
		prefix := variationPrefixes[g.rand.Intn(len(variationPrefixes))]
		out = append(out, &domain.Example{
			TaskID: task.ID,
			Input:  fmt.Sprintf("%s%s", prefix, base.Input),
			Output: base.Output,
			Source: domain.SourceSynthetic,
		})
	}
	return out
}
