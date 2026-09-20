"""Regression test for the missing attention head_dim metadata (Qwen3).

Real ``Qwen3-0.6B`` decouples the per-head attention dimension from the
``hidden_size // num_attention_heads`` default via an explicit ``head_dim``
field in ``config.json``:

    "hidden_size": 1024,
    "num_attention_heads": 16,
    "head_dim": 128

``128 × 16 = 2048`` — exactly the width llama.cpp finds on ``attn_q.weight``.
The original converter never wrote ``key_length``/``value_length`` metadata,
so llama.cpp fell back to its default (``n_embd / n_head = 1024 / 16 = 64``)
and rejected the correctly-shaped tensors (``check_tensor_dims`` error).

The fix belongs in *metadata*, not in the tensors — the tensors were trained
correctly. These tests lock in:

1. ``key_length`` / ``value_length`` come from the *source config's*
   ``head_dim`` (128 in this scenario), not from a hardcoded fallback.
2. The fallback (``hidden_size // num_attention_heads``) fires when the
   config has no explicit ``head_dim``.
3. The hard gate raises when the metadata implied by the config does not
   match the actual ``blk.0.attn_q.weight`` tensor shape.
4. The tokenizer's control-type special tokens (e.g. ``</s>``) get
   ``TokenType.CONTROL``, fixing llama.cpp's "control-looking token was
   not control-type" warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gguf")

from gguf.constants import TokenType  # noqa: E402

from trainer.gguf import GGUFConversionError, GGUFConverter  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _write_qwen3_shaped_config(
    model_dir: Path,
    *,
    hidden_size: int = 1024,
    n_heads: int = 16,
    head_dim: int | None = 128,
    n_layers: int = 1,
    intermediate_size: int = 4096,
) -> None:
    """Write a config.json shaped like real Qwen3-0.6B.

    When ``head_dim is None`` the key is omitted entirely — mimicking an
    architecture that doesn't decouple head_dim (so the converter must
    fall back to hidden_size // num_attention_heads).
    """
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": hidden_size,
        "num_attention_heads": n_heads,
        "num_hidden_layers": n_layers,
        "num_key_value_heads": 8,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 4096,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
    }
    if head_dim is not None:
        config["head_dim"] = head_dim
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )


def _write_tokenizer_files(
    model_dir: Path,
    *,
    n_base: int = 10,
    eos_name: str = "</s>",
    eos_id: int | None = None,
) -> None:
    """Write minimal BPE tokenizer files (vocab.json + added_tokens.json)."""
    vocab = {f"tok_{i}": i for i in range(n_base)}
    (model_dir / "vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8"
    )

    added: dict[str, int] = {}
    n = n_base
    for name in ("<|im_start|>", "<|im_end|>", eos_name):
        added[name] = n
        n += 1
    (model_dir / "added_tokens.json").write_text(
        json.dumps(added), encoding="utf-8"
    )

    (model_dir / "merges.txt").write_text(
        "tok_0 tok_1\n", encoding="utf-8"
    )

    # tokenizer.json for SpecialVocab EOS lookup.
    full_vocab = dict(vocab)
    full_vocab.update(added)
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": full_vocab},
                "decoder": {"type": "ByteLevel"},
            }
        ),
        encoding="utf-8",
    )

    # tokenizer_config.json with EOS as a string (dict form).
    if eos_id is None:
        eos_id = n_base + 2  # 10 + im_start + im_end => </s> is id 12.
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": {"content": eos_name, "id": eos_id},
            }
        ),
        encoding="utf-8",
    )


def _write_safetensors(
    model_dir: Path,
    *,
    q_out: int = 2048,
    embd_rows: int = 13,
    hidden_size: int = 1024,
) -> None:
    """Write a small but *valid* safetensors file with the key Qwen3 tensors.

    ``model.embed_tokens.weight`` is needed for the embedding-row scan
    (embd_rows); ``model.layers.0.self_attn.q_proj.weight``'s output
    dimension (``q_out``) drives the attention head_dim hard gate.
    """
    import numpy as np
    from safetensors.numpy import save_file

    tensors = {
        "model.embed_tokens.weight": np.zeros(
            (embd_rows, hidden_size), dtype=np.float32
        ),
        "model.layers.0.self_attn.q_proj.weight": np.zeros(
            (q_out, hidden_size), dtype=np.float32
        ),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))


class _FakeWriter:
    """Stand-in for GGUFWriter capturing the KV calls made."""

    def __init__(self) -> None:
        self.kvs: dict[str, object] = {}
        self.calls: list[str] = []

    def _record(self, name: str, *args) -> None:
        self.calls.append(name)

    def add_tokenizer_model(self, model: str) -> None:
        self._record("add_tokenizer_model")

    def add_token_list(self, tokens) -> None:
        self._record("add_token_list")

    def add_token_scores(self, scores) -> None:
        self._record("add_token_scores")

    def add_token_types(self, types) -> None:
        self._record("add_token_types")
        self.kvs["token_types"] = list(types)

    # SpecialVocab.add_to_gguf calls these on the writer.
    def add_token_merges(self, merges) -> None:
        self._record("add_token_merges")

    def add_token_id(self, *args, **kwargs) -> None:
        self._record("add_token_id")

    def add_bos_token_id(self, *args, **kwargs) -> None:
        self._record("add_bos_token_id")

    def add_eos_token_id(self, *args, **kwargs) -> None:
        self._record("add_eos_token_id")

    def add_unk_token_id(self, *args, **kwargs) -> None:
        self._record("add_unk_token_id")

    def add_sep_token_id(self, *args, **kwargs) -> None:
        self._record("add_sep_token_id")

    def add_pad_token_id(self, *args, **kwargs) -> None:
        self._record("add_pad_token_id")

    def add_cls_token_id(self, *args, **kwargs) -> None:
        self._record("add_cls_token_id")

    def add_mask_token_id(self, *args, **kwargs) -> None:
        self._record("add_mask_token_id")

    def add_chat_template(self, template) -> None:
        self._record("add_chat_template")


def _make_converter(tmp_path: Path) -> GGUFConverter:
    return GGUFConverter(
        base_model="test/base",
        job_dir=tmp_path,
        output_name="test",
    )


def _capture_head_dim(tmp_path: Path, *, head_dim: int | None, q_out: int) -> dict[str, int]:
    """Run the conversion and capture the attention/rope dimension metadata.

    Returns ``{"key_length", "value_length", "rope_dimension_count"}`` — the
    values the converter passes to the GGUF writer. ``rope_dimension_count``
    is included because n_rot must equal n_embd_head: if the converter writes
    key_length=128 (from config head_dim) but leaves n_rot at the stale
    hidden_size//n_heads=64, llama.cpp's ``GGML_ASSERT(n_embd_head == n_rot)``
    aborts at load time.
    """
    import gguf

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen3_shaped_config(model_dir, head_dim=head_dim)
    _write_tokenizer_files(model_dir)
    _write_safetensors(model_dir, q_out=q_out)

    captured: dict[str, int] = {}

    original_add_key_length = gguf.GGUFWriter.add_key_length
    original_add_value_length = gguf.GGUFWriter.add_value_length
    original_add_rope_dimension_count = gguf.GGUFWriter.add_rope_dimension_count

    def spy_key_length(self, v, *args, **kwargs):
        captured["key_length"] = int(v)
        return original_add_key_length(self, v, *args, **kwargs)

    def spy_value_length(self, v, *args, **kwargs):
        captured["value_length"] = int(v)
        return original_add_value_length(self, v, *args, **kwargs)

    def spy_rope_dimension_count(self, v, *args, **kwargs):
        captured["rope_dimension_count"] = int(v)
        return original_add_rope_dimension_count(self, v, *args, **kwargs)

    gguf.GGUFWriter.add_key_length = spy_key_length
    gguf.GGUFWriter.add_value_length = spy_value_length
    gguf.GGUFWriter.add_rope_dimension_count = spy_rope_dimension_count
    try:
        converter = _make_converter(tmp_path)
        converter._convert_dir_pure_python(model_dir)
    finally:
        gguf.GGUFWriter.add_key_length = original_add_key_length
        gguf.GGUFWriter.add_value_length = original_add_value_length
        gguf.GGUFWriter.add_rope_dimension_count = original_add_rope_dimension_count

    return captured


# --------------------------------------------------------------------------
# 1. key_length / value_length from source config
# --------------------------------------------------------------------------
class TestHeadDimMetadata:
    def test_key_length_from_config_head_dim(self, tmp_path: Path) -> None:
        """key_length/value_length must come from config.json's head_dim (128)."""
        captured = _capture_head_dim(
            tmp_path, head_dim=128, q_out=2048  # 128 x 16 = 2048 = tensor width
        )
        assert captured.get("key_length") == 128, (
            f"key_length should be 128 (from config head_dim), got {captured.get('key_length')}"
        )
        assert captured.get("value_length") == 128, (
            f"value_length should be 128 (from config head_dim), got {captured.get('value_length')}"
        )

    def test_key_length_fallback_no_head_dim(self, tmp_path: Path) -> None:
        """Without head_dim, fall back to hidden_size // num_heads."""
        captured = _capture_head_dim(
            tmp_path, head_dim=None, q_out=1024  # 1024 // 16 = 64, matched by tensor
        )
        assert captured.get("key_length") == 64, (
            f"key_length fallback should be hidden_size//n_heads = 64, "
            f"got {captured.get('key_length')}"
        )
        assert captured.get("value_length") == 64, (
            f"value_length fallback should be 64, got {captured.get('value_length')}"
        )

    def test_rope_dimension_count_equals_head_dim(self, tmp_path: Path) -> None:
        """rope.dimension_count (n_rot) must equal head_dim (n_embd_head).

        The head_dim fix adds key_length/value_length=128, but n_rot must
        follow: llama.cpp asserts ``GGML_ASSERT(n_embd_head == n_rot)`` at
        model load. If the converter writes rope.dimension_count as the
        stale hidden_size//n_heads (=64 for Qwen3-0.6B) while n_embd_head
        is now 128, the model aborts before it can even load. This test
        locks n_rot to head_dim for the head_dim=128 case, and to the
        hidden_size//n_heads fallback when head_dim is absent.
        """
        captured = _capture_head_dim(
            tmp_path, head_dim=128, q_out=2048  # 128 x 16 = 2048 = tensor width
        )
        assert captured.get("key_length") == 128, (
            f"key_length should be 128 (from config head_dim), got {captured.get('key_length')}"
        )
        assert captured.get("rope_dimension_count") == 128, (
            f"rope.dimension_count (n_rot) should equal head_dim (n_embd_head) "
            f"= 128, got {captured.get('rope_dimension_count')}. llama.cpp "
            "asserts n_embd_head == n_rot at load — a stale hidden_size//"
            "n_heads=64 here makes the GGUF unloadable."
        )


# --------------------------------------------------------------------------
# 2. Hard gate: metadata implied dim must match actual tensor shape
# --------------------------------------------------------------------------
class TestHardGateAttentionDim:
    def test_hard_gate_raises_on_mismatch(self, tmp_path: Path) -> None:
        """A config whose head_dim does NOT match the q_proj tensor must raise.

        Config says head_dim=64 (implies 64x16=1024 output) but the actual
        ``blk.0.attn_q.weight`` has 2048 output rows. The converter must
        refuse to write a file llama.cpp would reject.
        """
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_qwen3_shaped_config(model_dir, head_dim=64)  # implies 64x16=1024
        _write_tokenizer_files(model_dir)
        _write_safetensors(model_dir, q_out=2048)  # actual 2048 -> mismatch

        converter = _make_converter(tmp_path)
        with pytest.raises(
            GGUFConversionError, match="key_length|value_length|head_dim"
        ):
            converter._convert_dir_pure_python(model_dir)

    def test_hard_gate_not_triggered_when_match(self, tmp_path: Path) -> None:
        """When head_dim matches the actual tensor, no error from the gate."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_qwen3_shaped_config(model_dir, head_dim=128)  # implies 128x16=2048
        _write_tokenizer_files(model_dir)
        _write_safetensors(model_dir, q_out=2048)  # actual matches

        converter = _make_converter(tmp_path)
        # Full conversion should complete (the file lands in tmp_path/gguf).
        try:
            converter._convert_dir_pure_python(model_dir)
        except Exception as exc:  # noqa: BLE001
            assert "GGUF metadata implies" not in str(exc), (
                f"hard gate should not raise in the matching case, got: {exc}"
            )


# --------------------------------------------------------------------------
# 3. Tokenizer control-type fix (EOS token must be TokenType.CONTROL)
# --------------------------------------------------------------------------
class TestTokenizerControlType:
    def test_eos_token_gets_control_type(self, tmp_path: Path) -> None:
        """The EOS token ('</s>') must carry TokenType.CONTROL in token_types.

        This fixes llama.cpp's warning:
            "control-looking token: 128247 '</s>' was not control-type;
            this is probably a bug in the model"
        """
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_qwen3_shaped_config(model_dir)
        _write_tokenizer_files(model_dir, eos_name="</s>")
        _write_safetensors(model_dir)

        writer = _FakeWriter()
        converter = _make_converter(tmp_path)
        converter._add_tokenizer(writer, model_dir, embd_rows=13)

        types = writer.kvs.get("token_types")
        assert types is not None, "token_types were never written"
        assert len(types) == 13

        # The EOS token </s> is the 3rd added token (position 12 = n_base+2).
        eos_type = types[12]
        assert eos_type == int(TokenType.CONTROL), (
            f"EOS token '</s>' has token type {eos_type}, "
            f"expected {int(TokenType.CONTROL)} (TokenType.CONTROL). "
            "llama.cpp would warn: 'control-looking token ... was not "
            "control-type'."
        )

    def test_normal_tokens_get_normal_type(self, tmp_path: Path) -> None:
        """Base vocab tokens must remain TokenType.NORMAL."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_qwen3_shaped_config(model_dir)
        _write_tokenizer_files(model_dir)
        _write_safetensors(model_dir)

        writer = _FakeWriter()
        converter = _make_converter(tmp_path)
        converter._add_tokenizer(writer, model_dir, embd_rows=13)

        types = writer.kvs.get("token_types")
        assert types is not None
        for i in range(10):  # base BPE tokens are indices 0..9
            assert types[i] == int(TokenType.NORMAL), (
                f"base token {i} has type {types[i]}, expected NORMAL"
            )