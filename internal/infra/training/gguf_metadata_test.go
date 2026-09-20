package training

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

// ggufTestKey describes a single metadata KV entry for building test fixtures.
type ggufTestKey struct {
	key   string
	value uint64 // simple integer value (or 0 for string values).
	isStr bool   // if true, value is stored as a string.
	str   string // string value (used when isStr=true).
}

// buildGGUFFixture constructs a minimal GGUF file in memory with the given
// metadata keys. Tensor count is always 0 (we never look at tensor data).
//
// Layout (per the GGUF spec, matching what the python gguf writer and
// llama.cpp emit):
//
//	[magic:4][version:u32][n_tensors:u64][n_kv:u64][kv pairs...]
//
// Each KV pair is: [key_len:u64][key][value_type:u32][value...]
// Value types are 4-byte uint32 (NOT 1 byte).
func buildGGUFFixture(keys []ggufTestKey) []byte {
	buf := new(bytes.Buffer)

	// Magic + version (v3).
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))

	// n_tensors = 0, n_kv = len(keys).
	binary.Write(buf, binary.LittleEndian, uint64(0))
	binary.Write(buf, binary.LittleEndian, uint64(len(keys)))

	// Value type is a u32 per the GGUF spec.
	const (
		ggufTypeString = 8
		ggufTypeUint64 = 10
	)

	// KV pairs.
	for _, k := range keys {
		// key: [len:u64][bytes]
		binary.Write(buf, binary.LittleEndian, uint64(len(k.key)))
		buf.WriteString(k.key)

		if k.isStr {
			// value type 8 = string.
			binary.Write(buf, binary.LittleEndian, uint32(ggufTypeString))
			binary.Write(buf, binary.LittleEndian, uint64(len(k.str)))
			buf.WriteString(k.str)
		} else {
			// value type 10 = uint64.
			binary.Write(buf, binary.LittleEndian, uint32(ggufTypeUint64))
			binary.Write(buf, binary.LittleEndian, k.value)
		}
	}

	return buf.Bytes()
}

// completeGQUFKeys returns representative keys for a *complete* GGUF —
// architecture/weights metadata plus tokenizer keys.
func completeGQUFKeys() []ggufTestKey {
	return []ggufTestKey{
		{key: "general.architecture", isStr: true, str: "qwen3"},
		{key: "general.name", isStr: true, str: "test-model"},
		{key: "qwen3.block_count", value: 32},
		{key: "qwen3.context_length", value: 4096},
		{key: "tokenizer.ggml.model", isStr: true, str: "qwen3"},
		{key: "tokenizer.ggml.tokens", value: 151936},
		{key: "tokenizer.ggml.bos_token_id", value: 151643},
		{key: "tokenizer.ggml.eos_token_id", value: 151645},
	}
}

// incompleteGQUFKeys returns keys for an *incomplete* GGUF — architecture and
// weights metadata but NO tokenizer keys (the bug being fixed).
func incompleteGQUFKeys() []ggufTestKey {
	return []ggufTestKey{
		{key: "general.architecture", isStr: true, str: "qwen3"},
		{key: "general.name", isStr: true, str: "test-model"},
		{key: "qwen3.block_count", value: 32},
		{key: "qwen3.context_length", value: 4096},
	}
}

func TestGGUFMetadataKeys_Complete(t *testing.T) {
	t.Parallel()

	data := buildGGUFFixture(completeGQUFKeys())

	keys, err := ggufMetadataKeysReader(bytes.NewReader(data), "test-complete.gguf")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(keys) != 8 {
		t.Errorf("expected 8 metadata keys, got %d", len(keys))
	}

	wantKeys := []string{
		"general.architecture",
		"general.name",
		"qwen3.block_count",
		"qwen3.context_length",
		"tokenizer.ggml.model",
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.bos_token_id",
		"tokenizer.ggml.eos_token_id",
	}
	for i, want := range wantKeys {
		if keys[i] != want {
			t.Errorf("key[%d] = %q, want %q", i, keys[i], want)
		}
	}

	if !GGUFHasTokenizerKeys(keys) {
		t.Error("expected tokenizer keys to be detected")
	}
}

// TestGGUFMetadataKeys_U32TypeLayoutRoundTrip verifies the parser reads value
// types as 4-byte uint32 — the on-disk layout the python gguf writer and
// llama.cpp emit. Before this fix the parser read a single byte, desyncing by
// 3 bytes and mis-reading the next string length (e.g. bogus 83886080).
func TestGGUFMetadataKeys_U32TypeLayoutRoundTrip(t *testing.T) {
	t.Parallel()

	// Hand-build with the exact python gguf writer layout:
	//   key: [len:u64][bytes]
	//   value_type: u32
	//   value: u32 for uint32 value (matches writer's _simple_value_packing)
	buf := new(bytes.Buffer)

	// Header: magic + version(3) + n_tensors(0) + n_kv(1).
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(0))
	binary.Write(buf, binary.LittleEndian, uint64(1))

	// KV: key "general.architecture", value_type=8(STRING), value "qwen3".
	key := "general.architecture"
	binary.Write(buf, binary.LittleEndian, uint64(len(key)))
	buf.WriteString(key)
	binary.Write(buf, binary.LittleEndian, uint32(8))
	binary.Write(buf, binary.LittleEndian, uint64(len("qwen3")))
	buf.WriteString("qwen3")

	keys, err := ggufMetadataKeysReader(bytes.NewReader(buf.Bytes()), "u32-roundtrip.gguf")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(keys) != 1 || keys[0] != "general.architecture" {
		t.Errorf("expected [general.architecture], got %v", keys)
	}
}

func TestGGUFMetadataKeys_Incomplete(t *testing.T) {
	t.Parallel()

	data := buildGGUFFixture(incompleteGQUFKeys())

	keys, err := ggufMetadataKeysReader(bytes.NewReader(data), "test-incomplete.gguf")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(keys) != 4 {
		t.Errorf("expected 4 metadata keys, got %d", len(keys))
	}

	if GGUFHasTokenizerKeys(keys) {
		t.Error("expected no tokenizer keys to be detected")
	}
}

func TestGGUFHasTokenizerKeys(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		keys []string
		want bool
	}{
		{"empty", nil, false},
		{"only arch", []string{"general.architecture", "general.name"}, false},
		{"ggml prefix", []string{"tokenizer.ggml.model", "tokenizer.ggml.tokens"}, true},
		{"generic prefix", []string{"tokenizer.model", "tokenizer.vocab"}, true},
		{"mixed", []string{"general.architecture", "tokenizer.ggml.bos_token_id"}, true},
		{"looks-like but different", []string{"tokenizer_ggml_model"}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := GGUFHasTokenizerKeys(tc.keys)
			if got != tc.want {
				t.Errorf("GGUFHasTokenizerKeys(%v) = %v, want %v", tc.keys, got, tc.want)
			}
		})
	}
}

func TestGGUFMetadataKeys_BadMagic(t *testing.T) {
	t.Parallel()

	data := []byte("NOTG")

	_, err := ggufMetadataKeysReader(bytes.NewReader(data), "bad.gguf")
	if err == nil {
		t.Fatal("expected error for bad magic")
	}

	if !strings.Contains(err.Error(), "not a valid GGUF") {
		t.Errorf("expected 'not a valid GGUF' error, got: %v", err)
	}
}

func TestGGUFMetadataKeys_Truncated(t *testing.T) {
	t.Parallel()

	// A complete fixture truncated mid-KV-section.
	full := buildGGUFFixture(completeGQUFKeys())
	truncated := full[:len(full)-5] // chop off part of the last KV value.

	_, err := ggufMetadataKeysReader(bytes.NewReader(truncated), "trunc.gguf")
	if err == nil {
		t.Fatal("expected error for truncated GGUF")
	}

	if !strings.Contains(err.Error(), "truncated") {
		t.Errorf("expected 'truncated' error, got: %v", err)
	}
}

func TestValidateGGUFCompleteness_CompleteFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := dir + "/complete.gguf"

	data := buildGGUFFixture(completeGQUFKeys())
	err := writeFile(path, data)
	if err != nil {
		t.Fatalf("failed to write fixture: %v", err)
	}

	err := ValidateGGUFCompleteness(path)
	if err != nil {
		t.Errorf("expected complete GGUF to pass validation, got: %v", err)
	}
}

func TestValidateGGUFCompleteness_IncompleteFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := dir + "/incomplete.gguf"

	data := buildGGUFFixture(incompleteGQUFKeys())
	if err := writeFile(path, data); err != nil {
		t.Fatalf("failed to write fixture: %v", err)
	}

	err := ValidateGGUFCompleteness(path)
	if err == nil {
		t.Fatal("expected incomplete GGUF to be rejected")
	}

	if !strings.Contains(err.Error(), "zero tokenizer keys") {
		t.Errorf("expected error to mention zero tokenizer keys, got: %v", err)
	}

	if !strings.Contains(err.Error(), "incomplete GGUF export") {
		t.Errorf("expected error to mention 'incomplete GGUF export', got: %v", err)
	}
}

func TestValidateGGUFCompleteness_NotGGUF(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := dir + "/notgguf.bin"

	// 4-byte "GGUF" magic but truncated header.
	if err := writeFile(path, []byte("GGUF\x03")); err != nil {
		t.Fatalf("failed to write fixture: %v", err)
	}

	err := ValidateGGUFCompleteness(path)
	if err == nil {
		t.Fatal("expected error for non-GGUF file")
	}

	if !strings.Contains(err.Error(), "truncated") {
		t.Errorf("expected truncation error, got: %v", err)
	}
}

func TestValidateGGUFCompleteness_MissingFile(t *testing.T) {
	t.Parallel()

	err := ValidateGGUFCompleteness(t.TempDir() + "/nonexistent.gguf")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// writeFile is a tiny helper that writes data to path (tests only).
func writeFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}
