package usecase

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

// IDGenerator generates unique, prefixed IDs for new entities.
type IDGenerator interface {
	NewID(prefix string) string
}

type RandomIDGenerator struct{}

func NewRandomIDGenerator() *RandomIDGenerator { return &RandomIDGenerator{} }

func (g *RandomIDGenerator) NewID(prefix string) string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%s_%s", prefix, hex.EncodeToString(b))
}
