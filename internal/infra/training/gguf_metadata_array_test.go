package training

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// TestGGUFMetadataKeys_ArrayLayoutRoundTrip verifies ARRAY value parsing with
// the real on-disk layout the python gguf writer and llama.cpp emit:
//
//	[value_type:u32][elem_type:u32][n_elem:u64][elements...]
//
// Real GGUFs store tokenizer.ggml.tokens / merges as ARRAY[STRING] and
// tokenizer.ggml.scores / token_type as ARRAY[FLOAT32] / ARRAY[INT32]. The
// pre-fix parser read value/elem types as a single byte, desyncing by 3 bytes
// and mis-reading the following string/array lengths (e.g. bogus 83886080).
func TestGGUFMetadataKeys_ArrayLayoutRoundTrip(t *testing.T) {
	t.Parallel()

	buf := new(bytes.Buffer)

	// Header: magic + version(3) + n_tensors(0) + n_kv(3).
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(0))
	binary.Write(buf, binary.LittleEndian, uint64(3))

	writeKV := func(key string, vtype uint32) {
		binary.Write(buf, binary.LittleEndian, uint64(len(key)))
		buf.WriteString(key)
		binary.Write(buf, binary.LittleEndian, vtype)
	}

	// 1) tokenizer.ggml.model = STRING("qwen3").
	writeKV("tokenizer.ggml.model", 8)
	binary.Write(buf, binary.LittleEndian, uint64(len("qwen3")))
	buf.WriteString("qwen3")

	// 2) tokenizer.ggml.merges = ARRAY[STRING] of 2.
	writeKV("tokenizer.ggml.merges", 9)
	binary.Write(buf, binary.LittleEndian, uint32(8)) // elem string.
	binary.Write(buf, binary.LittleEndian, uint64(2))

	for _, s := range []string{"a b", "c d"} {
		binary.Write(buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}

	// 3) tokenizer.ggml.scores = ARRAY[FLOAT32] of 2.
	writeKV("tokenizer.ggml.scores", 9)
	binary.Write(buf, binary.LittleEndian, uint32(6)) // elem float32.
	binary.Write(buf, binary.LittleEndian, uint64(2))

	var f float32 = 0.5
	binary.Write(buf, binary.LittleEndian, f)
	binary.Write(buf, binary.LittleEndian, f)

	keys, err := ggufMetadataKeysReader(bytes.NewReader(buf.Bytes()), "array-roundtrip.gguf")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	want := []string{"tokenizer.ggml.model", "tokenizer.ggml.merges", "tokenizer.ggml.scores"}
	if len(keys) != len(want) {
		t.Fatalf("expected %d keys, got %d: %v", len(want), len(keys), keys)
	}

	for i, w := range want {
		if keys[i] != w {
			t.Errorf("key[%d] = %q, want %q", i, keys[i], w)
		}
	}

	if !GGUFHasTokenizerKeys(keys) {
		t.Error("expected tokenizer keys to be detected")
	}
}
