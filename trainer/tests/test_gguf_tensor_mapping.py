"""Regression test for the GGUF tensor-name-mapping bug.

TensorNameMap.get_name() only strips the ".weight"/".bias" suffix when you
pass try_suffixes explicitly -- its internal lookup table is keyed WITHOUT
that suffix. Calling it with no try_suffixes (as trainer/gguf.py originally
did) means every real tensor name misses the lookup and returns None; the
`or name` fallback in the calling code then silently substituted the raw
HuggingFace tensor name, producing a GGUF file that looked complete (right
header, right tensor count, valid tokenizer metadata) but was unloadable by
llama.cpp, which requires exact llama.cpp-style tensor names
(blk.N.attn_q.weight, token_embd.weight, etc).

This test locks in the fix: every tensor name a real HF safetensors
checkpoint would produce must resolve to its llama.cpp name, for EVERY
architecture the app supports -- not just Qwen3. Future architecture
additions should fail CI immediately the same way this bug should have been
caught before ever reaching a user.
"""
from __future__ import annotations

import pytest

gguf = pytest.importorskip("gguf")

from gguf.constants import MODEL_ARCH  # noqa: E402
from gguf.tensor_mapping import TensorNameMap  # noqa: E402


# Every architecture trainer/gguf.py _ARCH_MAP can emit, together with a
# realistic set of HF safetensors tensor names that a real checkpoint of
# that architecture contains. Keep this in sync with _ARCH_MAP when new
# architectures are added. (MistralForCausalLM/MixtralForCausalLM map to
# LLAMA in the app, so they share the LLAMA candidate set.)
#
# GEMMA/GEMMA2 intentionally do NOT list "lm_head.weight": these models tie
# embeddings and have no separate output projection tensor in a real
# checkpoint, so TensorNameMap correctly returns None for it. The list below
# mirrors what llama.cpp's own converter expects for each arch.
_SUPPORTED_ARCHS = {
    "LLAMA": [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
    "QWEN2": [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
    "QWEN3": [
        # Qwen3 is the architecture that originally shipped broken
        # (task-1-v1-task-1-q4_k_m.gguf).
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
    # GEMMA / GEMMA2 tie embeddings -- no lm_head tensor in a real
    # checkpoint (the app's writer only sees what safetensors contains).
    "GEMMA": [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
    "GEMMA2": [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        # Gemma2-specific norms:
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
    ],
    "PHI3": [
        # Phi-3 merges query/key/value into a single qkv_proj and gate/up
        # into one gate_up_proj -- the tensor names differ from Llama.
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
}


# Spot-check assertions: the HF tensor name must resolve to this exact
# llama.cpp name. Verified directly against the installed gguf package.
_SPOT_CHECKS = {
    "LLAMA": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
        ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
        ("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"),
        ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
        ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
        ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
    ],
    "QWEN2": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
    ],
    "QWEN3": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("model.layers.0.self_attn.q_norm.weight", "blk.0.attn_q_norm.weight"),
        ("model.layers.0.self_attn.k_norm.weight", "blk.0.attn_k_norm.weight"),
    ],
    "GEMMA": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
    ],
    "GEMMA2": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.layers.0.pre_feedforward_layernorm.weight", "blk.0.ffn_norm.weight"),
        ("model.layers.0.post_feedforward_layernorm.weight", "blk.0.post_ffw_norm.weight"),
    ],
    "PHI3": [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.layers.0.self_attn.qkv_proj.weight", "blk.0.attn_qkv.weight"),
        ("model.layers.0.mlp.gate_up_proj.weight", "blk.0.ffn_up.weight"),
    ],
}


@pytest.mark.parametrize("arch_name", sorted(_SUPPORTED_ARCHS))
def test_get_name_without_try_suffixes_always_misses(arch_name: str) -> None:
    """Documents the bug: this is what the old code path did, for every arch."""
    tensor_map = TensorNameMap(MODEL_ARCH[arch_name], 32)
    for name in _SUPPORTED_ARCHS[arch_name]:
        assert tensor_map.get_name(name) is None, (
            f"{name!r} unexpectedly resolved without try_suffixes -- if this "
            "assertion starts failing, gguf's TensorNameMap behavior has "
            "changed and the accompanying code comment in trainer/gguf.py "
            "may be out of date."
        )


@pytest.mark.parametrize("arch_name", sorted(_SUPPORTED_ARCHS))
def test_get_name_with_try_suffixes_resolves_every_tensor(arch_name: str) -> None:
    """This is the fix: every real tensor name must resolve to a llama.cpp name."""
    tensor_map = TensorNameMap(MODEL_ARCH[arch_name], 32)
    resolved = {
        name: tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
        for name in _SUPPORTED_ARCHS[arch_name]
    }

    unresolved = [name for name, mapped in resolved.items() if mapped is None]
    assert not unresolved, (
        f"[{arch_name}] tensor names failed to resolve: {unresolved}"
    )

    # Spot-check a few concrete mappings so a regression is caught even if
    # every name "resolves" to something wrong.
    for src, expected in _SPOT_CHECKS.get(arch_name, []):
        assert resolved[src] == expected, (
            f"[{arch_name}] {src!r} mapped to {resolved[src]!r}, "
            f"expected {expected!r}"
        )

    # None of the resolved names should still look like a HuggingFace name --
    # this is the actual symptom that made GGUF files unloadable.
    for mapped in resolved.values():
        assert not mapped.startswith("model."), (
            f"[{arch_name}] still HF-shaped: {mapped!r}"
        )
        assert "self_attn" not in mapped and "mlp." not in mapped, (
            f"[{arch_name}] still HF-shaped: {mapped!r}"
        )


def test_hard_gate_produces_token_embd_and_blk0() -> None:
    """The conversion hard gate requires token_embd + at least one blk.0 tensor."""
    for arch_name, names in _SUPPORTED_ARCHS.items():
        tensor_map = TensorNameMap(MODEL_ARCH[arch_name], 32)
        mapped = {
            tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            for name in names
            if tensor_map.get_name(name, try_suffixes=(".weight", ".bias")) is not None
        }
        assert "token_embd.weight" in mapped, (
            f"[{arch_name}] hard gate would fail: no token_embd.weight"
        )
        assert any(n.startswith("blk.0.") for n in mapped), (
            f"[{arch_name}] hard gate would fail: no blk.0.* tensors"
        )