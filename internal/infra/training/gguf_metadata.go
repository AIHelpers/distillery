package training

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// GGUFMagic is the 4-byte GGUF file magic header.
var GGUFMagic = []byte{'G', 'G', 'U', 'F'}

// Sentinel errors for GGUF parsing failures. Each is wrapped with the
// specific detail (file name, offending value, limit) at its call site.
var (
	errGGUFTypeOutOfRange   = errors.New("GGUF value type out of range")
	errGGUFTruncated        = errors.New("GGUF metadata is truncated")
	errGGUFStringLen        = errors.New("GGUF string length is implausible")
	errGGUFBadMagic         = errors.New("not a valid GGUF (bad magic)")
	errGGUFKVCount          = errors.New("GGUF metadata count is implausible")
	errGGUFArrayElemType    = errors.New("GGUF array element type unsupported")
	errGGUFArrayTooLarge    = errors.New("GGUF array value too large")
	errGGUFValueType        = errors.New("GGUF value type unsupported")
	errGGUFMetadataTooLarge = errors.New("GGUF metadata section too large")

	// ErrIncompleteGGUF is returned by ValidateGGUFCompleteness when a GGUF
	// has architecture/weights metadata but no tokenizer keys, meaning
	// llama.cpp / HomeBred-LLM would be unable to tokenize prompts for it.
	ErrIncompleteGGUF = errors.New("incomplete GGUF export")
)

// GGUFValueType enum per the GGUF spec.
//
//	0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32,
//	6 float32, 7 bool, 8 string, 9 array, 10 uint64, 11 int64,
//	12 float64
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// ggufValueSizes maps GGUF value types to their fixed byte width (excluding
// string and array which have variable lengths).
var ggufValueSizes = map[byte]int{
	ggufTypeUint8:   1,
	ggufTypeInt8:    1,
	ggufTypeUint16:  2,
	ggufTypeInt16:   2,
	ggufTypeUint32:  4,
	ggufTypeInt32:   4,
	ggufTypeFloat32: 4,
	ggufTypeBool:    1,
	ggufTypeUint64:  8,
	ggufTypeInt64:   8,
	ggufTypeFloat64: 8,
}

// ggufType narrows a raw u32 GGUF type tag read from the file to the byte
// range ggufValueSizes and the switch statements below key on. A value
// outside 0-255 is not a truncation risk here — it's proof the file is
// corrupt or hostile, so it's rejected rather than silently wrapped.
func ggufType(name string, raw uint32) (byte, error) {
	if raw > math.MaxUint8 {
		return 0, fmt.Errorf("%s: %w: %d", name, errGGUFTypeOutOfRange, raw)
	}

	return byte(raw), nil
}

// ggufStringLenLimit caps a single metadata string length to guard against
// corrupt/truncated files claiming absurd lengths. 64 MB is generous for
// real GGUF metadata keys/values.
const ggufStringLenLimit = 64 * 1024 * 1024

// ggufMaxKVCount caps the number of metadata entries a GGUF may declare.
// Real GGUFs have tens to low thousands of keys; 1M is a sanity guard
// against corrupt/truncated files.
const ggufMaxKVCount = 1_000_000

// ggufMaxKVBytes caps the cumulative metadata KV section size. Tokenizer
// arrays (token list) can be large (~1MB for 128K vocab), so allow up to 8GB.
const ggufMaxKVBytes = 8 * 1024 * 1024 * 1024

// GGUFMetadataKeys walks the GGUF metadata KV section and returns all key
// names. It only parses the header + metadata KV section — tensor data is
// never touched, so it's cheap even for multi-GB model files.
//
// Returns an error if the file is not a valid GGUF, is truncated, or the
// metadata section is malformed.
func GGUFMetadataKeys(ggufPath string) ([]string, error) {
	f, err := os.Open(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("opening GGUF file: %w", err)
	}
	defer f.Close()

	return ggufMetadataKeysReader(f, filepath.Base(ggufPath))
}

// ggufReader wraps an io.Reader with the little-endian primitive readers the
// GGUF metadata format needs, tagging every error with the file name.
type ggufReader struct {
	r    io.Reader
	name string
}

// readExact reads exactly n bytes from the reader.
func (g *ggufReader) readExact(n int) ([]byte, error) {
	buf := make([]byte, n)

	_, err := io.ReadFull(g.r, buf)
	if err != nil {
		if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%s: %w", g.name, errGGUFTruncated)
		}

		return nil, fmt.Errorf("%s: reading GGUF: %w", g.name, err)
	}

	return buf, nil
}

// readU64 reads a u64 (GGUF stores counts as little-endian uint64).
func (g *ggufReader) readU64() (uint64, error) {
	buf, err := g.readExact(8)
	if err != nil {
		return 0, err
	}

	return binary.LittleEndian.Uint64(buf), nil
}

// readU32 reads a u32 (GGUF stores value types as little-endian uint32).
func (g *ggufReader) readU32() (uint32, error) {
	buf, err := g.readExact(4)
	if err != nil {
		return 0, err
	}

	return binary.LittleEndian.Uint32(buf), nil
}

// readStr reads a length-prefixed string.
func (g *ggufReader) readStr() (string, error) {
	n, err := g.readU64()
	if err != nil {
		return "", err
	}

	if n > ggufStringLenLimit {
		return "", fmt.Errorf("%s: %w: %d", g.name, errGGUFStringLen, n)
	}

	buf, err := g.readExact(int(n))
	if err != nil {
		return "", err
	}

	return string(buf), nil
}

// readHeader validates the GGUF magic/version and returns the declared
// metadata KV count (n_kv), having already validated it against
// ggufMaxKVCount.
func (g *ggufReader) readHeader() (nKV uint64, err error) {
	magic, err := g.readExact(4)
	if err != nil {
		return 0, err
	}

	if string(magic) != "GGUF" {
		return 0, fmt.Errorf("%s: %w: %q", g.name, errGGUFBadMagic, magic)
	}

	// Version (u32) — we don't need it, just validate it's readable.
	_, err = g.readExact(4)
	if err != nil {
		return 0, err
	}

	// n_tensors (u64) — metadata only; we don't walk tensor info.
	_, err = g.readU64()
	if err != nil {
		return 0, err
	}

	nKV, err = g.readU64()
	if err != nil {
		return 0, err
	}

	if nKV > ggufMaxKVCount {
		return 0, fmt.Errorf("%s: %w: %d", g.name, errGGUFKVCount, nKV)
	}

	return nKV, nil
}

// skipArrayPayload discards a raw fixed-width array payload of the given
// total byte size in bounded chunks, without buffering it all in memory.
func (g *ggufReader) skipArrayPayload(total uint64) error {
	remaining := total
	for remaining > 0 {
		chunk := remaining
		if chunk > 4*1024*1024 {
			chunk = 4 * 1024 * 1024
		}

		_, err := g.readExact(int(chunk))
		if err != nil {
			return err
		}

		remaining -= chunk
	}

	return nil
}

// readArrayValue reads a GGUF array value ([elem_type:u32][n_elem:u64]
// [elems...]), discarding the elements themselves, and returns its total
// encoded byte size (for the caller's cumulative-size accounting).
func (g *ggufReader) readArrayValue() (size uint64, err error) {
	elemTypeVal, err := g.readU32()
	if err != nil {
		return 0, err
	}

	elemType, err := ggufType(g.name, elemTypeVal)
	if err != nil {
		return 0, err
	}

	nElem, err := g.readU64()
	if err != nil {
		return 0, err
	}

	size = 1 + 8

	if elemType == ggufTypeString {
		// Array of strings.
		for range nElem {
			s, err := g.readStr()
			if err != nil {
				return 0, err
			}

			size += uint64(len(s)) + 8
		}

		return size, nil
	}

	elemSize, ok := ggufValueSizes[elemType]
	if !ok {
		return 0, fmt.Errorf("%s: %w: %d", g.name, errGGUFArrayElemType, elemType)
	}

	total := nElem * uint64(elemSize)
	if total > ggufMaxKVBytes {
		return 0, fmt.Errorf("%s: %w (%d bytes)", g.name, errGGUFArrayTooLarge, total)
	}

	if total > 0 {
		err = g.skipArrayPayload(total)
		if err != nil {
			return 0, err
		}
	}

	return size + total, nil
}

// readValue reads (or, for arrays, discards) a single GGUF metadata value of
// the given type and returns its total encoded byte size.
func (g *ggufReader) readValue(vType byte) (size uint64, err error) {
	switch vType {
	case ggufTypeString:
		val, err := g.readStr()
		if err != nil {
			return 0, err
		}

		return uint64(len(val)) + 8, nil

	case ggufTypeArray:
		return g.readArrayValue()

	default:
		fixedSize, ok := ggufValueSizes[vType]
		if !ok {
			return 0, fmt.Errorf("%s: %w: %d", g.name, errGGUFValueType, vType)
		}

		_, err = g.readExact(fixedSize)
		if err != nil {
			return 0, err
		}

		return uint64(fixedSize), nil
	}
}

// readEntry reads one metadata KV entry (key + typed value) and returns the
// key name plus the entry's total encoded byte size (key + value).
func (g *ggufReader) readEntry() (key string, size uint64, err error) {
	key, err = g.readStr()
	if err != nil {
		return "", 0, err
	}

	size = uint64(len(key)) + 8 // key bytes + length prefix.

	// Value type (u32 — per the GGUF spec, value types are stored as 4-byte
	// uint32, matching what llama.cpp and the python gguf writer emit).
	// Reading only 1 byte here desyncs the parser by 3 bytes and mis-reads
	// the next string length (e.g. a bogus 83886080).
	vTypeVal, err := g.readU32()
	if err != nil {
		return "", 0, err
	}

	vType, err := ggufType(g.name, vTypeVal)
	if err != nil {
		return "", 0, err
	}

	valSize, err := g.readValue(vType)
	if err != nil {
		return "", 0, err
	}

	return key, size + valSize, nil
}

// ggufMetadataKeysReader is the io.Reader-based implementation underlying
// GGUFMetadataKeys.
func ggufMetadataKeysReader(r io.Reader, name string) ([]string, error) {
	g := &ggufReader{r: r, name: name}

	nKV, err := g.readHeader()
	if err != nil {
		return nil, err
	}

	// --- Metadata KV section ---.
	keys := make([]string, 0, min(nKV, 1024))
	cumulativeBytes := uint64(0)

	for range nKV {
		key, entrySize, err := g.readEntry()
		if err != nil {
			return nil, err
		}

		keys = append(keys, key)
		cumulativeBytes += entrySize

		if cumulativeBytes > ggufMaxKVBytes {
			return nil, fmt.Errorf("%s: %w (%d bytes)", name, errGGUFMetadataTooLarge, cumulativeBytes)
		}
	}

	return keys, nil
}

// GGUFHasTokenizerKeys reports whether the metadata keys include at least
// one tokenizer key (tokenizer.ggml.* or the broader tokenizer.* namespace).
func GGUFHasTokenizerKeys(keys []string) bool {
	for _, k := range keys {
		if strings.HasPrefix(k, "tokenizer.ggml.") || strings.HasPrefix(k, "tokenizer.") {
			return true
		}
	}

	return false
}

// ValidateGGUFCompleteness validates that the GGUF file at path is:
//  1. A valid GGUF (magic + header parseable), and
//  2. Contains at least one tokenizer metadata key.
//
// A GGUF with architecture/weights metadata but zero tokenizer keys is an
// incomplete export — llama.cpp / HomeBred-LLM cannot tokenize prompts and
// the file is unloadable at inference time. This check must run on both
// freshly-converted files and cached files before serving them.
func ValidateGGUFCompleteness(ggufPath string) error {
	keys, err := GGUFMetadataKeys(ggufPath)
	if err != nil {
		return err
	}

	if !GGUFHasTokenizerKeys(keys) {
		return fmt.Errorf(
			"%s: %w: it has %d metadata keys for "+
				"architecture/weights but zero tokenizer keys. "+
				"llama.cpp / HomeBred-LLM cannot tokenize prompts without "+
				"tokenizer metadata, so the file would be unloadable at "+
				"inference time. Ensure the training job saved its tokenizer "+
				"(tokenizer.json / tokenizer.model) alongside the model "+
				"weights, then re-run the export",
			filepath.Base(ggufPath),
			ErrIncompleteGGUF,
			len(keys),
		)
	}

	return nil
}
