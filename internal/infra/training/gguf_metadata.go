package training

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// GGUF_MAGIC is the 4-byte GGUF file magic header.
var GGUF_MAGIC = []byte{'G', 'G', 'U', 'F'}

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

// ggufMetadataKeysReader is the io.Reader-based implementation shared by
// GGUFMetadataKeys and the tests (which feed in-memory buffers).
func ggufMetadataKeysReader(r io.Reader, name string) ([]string, error) {
	// Helper: read exactly n bytes from the reader.
	readExact := func(n int) ([]byte, error) {
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, fmt.Errorf("%s: GGUF metadata is truncated", name)
			}

			return nil, fmt.Errorf("%s: reading GGUF: %w", name, err)
		}

		return buf, nil
	}

	// Read a u64 (GGUF stores counts as little-endian uint64).
	readU64 := func() (uint64, error) {
		buf, err := readExact(8)
		if err != nil {
			return 0, err
		}

		return binary.LittleEndian.Uint64(buf), nil
	}

	// Read a u32 (GGUF stores value types as little-endian uint32).
	readU32 := func() (uint32, error) {
		buf, err := readExact(4)
		if err != nil {
			return 0, err
		}

		return binary.LittleEndian.Uint32(buf), nil
	}

	// Read a length-prefixed string.
	readStr := func() (string, error) {
		n, err := readU64()
		if err != nil {
			return "", err
		}

		if n > ggufStringLenLimit {
			return "", fmt.Errorf("%s: GGUF string length %d is implausible", name, n)
		}

		buf, err := readExact(int(n))
		if err != nil {
			return "", err
		}

		return string(buf), nil
	}

	// --- Header ---.
	magic, err := readExact(4)
	if err != nil {
		return nil, err
	}

	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("%s: not a valid GGUF (bad magic %q)", name, magic)
	}

	// Version (u32) — we don't need it, just validate it's readable.
	if _, err := readExact(4); err != nil {
		return nil, err
	}

	// n_tensors (u64) — metadata only; we don't walk tensor info.
	if _, err := readU64(); err != nil {
		return nil, err
	}

	// n_kv (u64).
	nKV, err := readU64()
	if err != nil {
		return nil, err
	}

	if nKV > ggufMaxKVCount {
		return nil, fmt.Errorf("%s: GGUF metadata count %d is implausible", name, nKV)
	}

	// --- Metadata KV section ---.
	keys := make([]string, 0, min(nKV, 1024))
	cumulativeBytes := uint64(0)

	for range nKV {
		key, err := readStr()
		if err != nil {
			return nil, err
		}

		keys = append(keys, key)
		cumulativeBytes += uint64(len(key)) + 8 // key bytes + length prefix.

		// Value type (u32 — per the GGUF spec, value types are stored as
		// 4-byte uint32, matching what llama.cpp and the python gguf writer
		// emit). Reading only 1 byte here desyncs the parser by 3 bytes and
		// mis-reads the next string length (e.g. a bogus 83886080).
		vTypeVal, err := readU32()
		if err != nil {
			return nil, err
		}

		vType := byte(vTypeVal)

		switch vType {
		case ggufTypeString:
			val, err := readStr()
			if err != nil {
				return nil, err
			}

			cumulativeBytes += uint64(len(val)) + 8

		case ggufTypeArray:
			// Array: [elem_type:u32][n_elem:u64][elems...].
			elemTypeVal, err := readU32()
			if err != nil {
				return nil, err
			}

			elemType := byte(elemTypeVal)

			nElem, err := readU64()
			if err != nil {
				return nil, err
			}

			cumulativeBytes += 1 + 8

			if elemType == ggufTypeString {
				// Array of strings.
				for range nElem {
					s, err := readStr()
					if err != nil {
						return nil, err
					}

					cumulativeBytes += uint64(len(s)) + 8
				}
			} else {
				elemSize, ok := ggufValueSizes[elemType]
				if !ok {
					return nil, fmt.Errorf("%s: GGUF array element type %d unsupported", name, elemType)
				}

				total := nElem * uint64(elemSize)
				if total > ggufMaxKVBytes {
					return nil, fmt.Errorf("%s: GGUF array value too large (%d bytes)", name, total)
				}

				if total > 0 {
					// Skip over the raw array payload in bounded chunks.
					remaining := total
					for remaining > 0 {
						chunk := remaining
						if chunk > 4*1024*1024 {
							chunk = 4 * 1024 * 1024
						}

						if _, err := readExact(int(chunk)); err != nil {
							return nil, err
						}

						remaining -= chunk
					}
				}

				cumulativeBytes += total
			}

		default:
			size, ok := ggufValueSizes[vType]
			if !ok {
				return nil, fmt.Errorf("%s: GGUF value type %d unsupported", name, vType)
			}

			if _, err := readExact(size); err != nil {
				return nil, err
			}

			cumulativeBytes += uint64(size)
		}

		if cumulativeBytes > ggufMaxKVBytes {
			return nil, fmt.Errorf("%s: GGUF metadata section too large (%d bytes)", name, cumulativeBytes)
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
			"%s is an incomplete GGUF export: it has %d metadata keys for "+
				"architecture/weights but zero tokenizer keys. "+
				"llama.cpp / HomeBred-LLM cannot tokenize prompts without "+
				"tokenizer metadata, so the file would be unloadable at "+
				"inference time. Ensure the training job saved its tokenizer "+
				"(tokenizer.json / tokenizer.model) alongside the model "+
				"weights, then re-run the export.",
			filepath.Base(ggufPath),
			len(keys),
		)
	}

	return nil
}
