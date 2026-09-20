"""Regression test for the GGUF tokenizer vocab / embedding-row mismatch.

The Qwen2/2.5/3 pattern: a checkpoint's embedding matrix
(``model.embed_tokens.weight``) is padded to a hardware-aligned row count
(e.g. 151936) while the tokenizer's *defined* BPE vocab + special tokens only
cover ~151643 of those rows (the rest are reserved/placeholder slots).
llama.cpp refuses to load a GGUF where ``tokenizer.ggml.tokens`` disagrees
with ``token_embd.weight``'s row count (``check_tensor_dims``).

The converter bug this test locks in: the token list was built from the
tokenizer's defined vocab only, while ``token_embd.weight`` was written from
the safetensors shape — the two were never reconciled. The fix pads the
*token list* (never trims the embedding matrix) to match the embedding
row count, using the source tokenizer's own ``<|extra_N|>`` placeholders
when they exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gguf")

from trainer.gguf import GGUFConversionError, GGUFConverter  # noqa: E402


def _write_tokenizer_files(
    model_dir: Path,
    n_extra: int = 0,
    n_base: int = 10,
) -> None:
    """Write the tokenizer files a Qwen-style model directory would have.

    The vocab.json (which the pure-Python BpeVocab reads) deliberately does
    NOT include the reserved ``<|extra_N|>`` tokens — that's the bug
    condition we're reproducing: the *real* tokenizer.json includes them
    but BpeVocab (reading vocab.json + added_tokens.json) drops them.

    Args:
        model_dir: directory to write files into.
        n_extra: number of ``<|extra_N|>`` reserved slots in tokenizer.json.
        n_base: number of real BPE tokens in vocab.json.
    """
    # vocab.json: only real BPE tokens, sequential IDs 0..n_base-1.
    vocab = {f"tok_{i}": i for i in range(n_base)}
    (model_dir / "vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8"
    )

    # added_tokens.json: sequential IDs starting at n_base (BpeVocab's
    # __init__ validates this exact pattern).
    added: dict[str, int] = {}
    for i, name in enumerate(("<|endoftext|>", "<|im_start|>", "<|im_end|>")):
        added[name] = n_base + i
    (model_dir / "added_tokens.json").write_text(
        json.dumps(added), encoding="utf-8"
    )

    # merges.txt: required for BpeVocab's constructor to pick the BPE path.
    (model_dir / "merges.txt").write_text(
        "tok_0 tok_1\n", encoding="utf-8"
    )

    # tokenizer.json: the *real* HF fast tokenizer, whose model.vocab
    # includes the reserved placeholder tokens the converter must recover.
    full_vocab = dict(vocab)
    full_vocab.update(dict(added))
    next_id = n_base + len(added)
    for i in range(n_extra):
        full_vocab[f"<|extra_{i}|>"] = next_id + i

    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": full_vocab},
                "decoder": {"type": "ByteLevel"},
            }
        ),
        encoding="utf-8",
    )


class _FakeWriter:
    """Stand-in for GGUFWriter capturing the arrays passed to it.

    Implements the same methods GGUFWriter exposes that ``SpecialVocab``
    and ``_add_tokenizer`` call, so the real ``_add_tokenizer`` can run
    against this stand-in without needing a real output file.
    """

    def __init__(self) -> None:
        self.token_list: list = []
        self.token_scores: list[float] = []
        self.token_types: list[int] = []
        self.tokenizer_model: str | None = None
        self.calls: list[str] = []  # record every add_* call for debugging.

    def _record(self, name: str) -> None:
        self.calls.append(name)

    def add_tokenizer_model(self, model: str) -> None:
        self.tokenizer_model = model
        self._record("add_tokenizer_model")

    def add_token_list(self, tokens) -> None:
        self.token_list = list(tokens)
        self._record("add_token_list")

    def add_token_scores(self, scores) -> None:
        self.token_scores = list(scores)
        self._record("add_token_scores")

    def add_token_types(self, types) -> None:
        self.token_types = list(types)
        self._record("add_token_types")

    # ---- SpecialVocab.add_to_gguf calls these on the writer ---- #
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


def test_pad_token_list_to_embedding_rows(tmp_path: Path) -> None:
    """The token list must be padded to match the embedding row count.

    Scenario: 10 base BPE tokens + 3 added specials = 13 defined tokens,
    but the embedding matrix is padded to 16 rows (13 + 3 reserved
    placeholder slots). The tokenizer.json defines the 3 reserved
    ``<|extra_0|>`` … ``<|extra_2|>`` names; the converter must recover
    and use them rather than inventing new ones.
    """
    n_extra = 3
    embd_rows = 16  # 13 defined + 3 reserved.

    _write_tokenizer_files(tmp_path, n_extra=n_extra)

    converter = GGUFConverter(
        base_model="test/base",
        job_dir=tmp_path,
        output_name="test",
    )

    writer = _FakeWriter()
    converter._add_tokenizer(writer, tmp_path, embd_rows=embd_rows)

    # The padded token list must exactly match the embedding row count —
    # this is the *relationship* that llama.cpp requires (check_tensor_dims).
    assert len(writer.token_list) == embd_rows, (
        f"tokenizer.ggml.tokens has {len(writer.token_list)} entries but "
        f"token_embd.weight has {embd_rows} rows — llama.cpp would refuse "
        "to load this GGUF."
    )
    assert len(writer.token_scores) == embd_rows
    assert len(writer.token_types) == embd_rows

    # The padding slots must be the source tokenizer's own reserved
    # <|extra_N|> tokens, not invented names.
    padded = [t.decode("utf-8") if isinstance(t, bytes) else str(t)
              for t in writer.token_list]
    extra_slots = [t for t in padded if t.startswith("<|extra_")]
    assert len(extra_slots) == n_extra, (
        f"expected {n_extra} <|extra_N|> padding slots, got {len(extra_slots)}: "
        f"{padded}"
    )
    assert extra_slots == [f"<|extra_{i}|>" for i in range(n_extra)], (
        f"padding tokens should reuse the source tokenizer's own reserved "
        f"names, got {extra_slots}"
    )


def test_padding_uses_unused_token_type(tmp_path: Path) -> None:
    """Padding slots must carry the UNUSED token type (gguf's TokenType.UNUSED)."""
    n_extra = 3
    embd_rows = 16

    _write_tokenizer_files(tmp_path, n_extra=n_extra)

    converter = GGUFConverter(
        base_model="test/base",
        job_dir=tmp_path,
        output_name="test",
    )

    writer = _FakeWriter()
    converter._add_tokenizer(writer, tmp_path, embd_rows=embd_rows)

    # 13 defined tokens fill indices 0..12; padding slots are 13..15.
    n_defined = 13
    token_types = writer.token_types
    assert len(token_types) == embd_rows

    unused_type = 5  # gguf.constants.TokenType.UNUSED
    for i in range(n_defined, embd_rows):
        assert token_types[i] == unused_type, (
            f"padding slot {i} has token type {token_types[i]}, "
            f"expected {unused_type} (TokenType.UNUSED)"
        )


def test_padding_generates_names_when_source_has_none(tmp_path: Path) -> None:
    """If the source tokenizer defines no <|extra_N|>, generate them.

    Scenario: 13 defined tokens, embedding has 16 rows, but tokenizer.json
    has NO reserved placeholder names. The converter must generate
    ``<|extra_13|>``, ``<|extra_14|>``, ``<|extra_15|>`` following the Qwen
    convention (N counting up from the current token count).
    """
    embd_rows = 16

    _write_tokenizer_files(tmp_path, n_extra=0)

    converter = GGUFConverter(
        base_model="test/base",
        job_dir=tmp_path,
        output_name="test",
    )

    writer = _FakeWriter()
    converter._add_tokenizer(writer, tmp_path, embd_rows=embd_rows)

    assert len(writer.token_list) == embd_rows
    padded = [t.decode("utf-8") if isinstance(t, bytes) else str(t)
              for t in writer.token_list]

    # Generated placeholders start at the current token count (13).
    expected_generated = [f"<|extra_{13 + i}|>" for i in range(3)]
    extra_slots = [t for t in padded if t.startswith("<|extra_")]
    assert extra_slots == expected_generated, (
        f"generated placeholder names should follow <<|extra_N|>> with N "
        f"starting at the current token count; got {extra_slots}"
    )


def test_hard_gate_raises_when_tokens_exceed_embd(tmp_path: Path) -> None:
    """Converter must refuse when the tokenizer defines MORE tokens than embd rows.

    Scenario: 12 base BPE tokens + 3 added = 15 defined tokens, but the
    embedding matrix has only 14 rows. Padding can't help; raising is
    correct (the source files are inconsistent).
    """
    embd_rows = 14  # fewer rows than 15 defined tokens.

    _write_tokenizer_files(tmp_path, n_extra=0, n_base=12)

    converter = GGUFConverter(
        base_model="test/base",
        job_dir=tmp_path,
        output_name="test",
    )

    writer = _FakeWriter()
    with pytest.raises(GGUFConversionError, match="tokenizer.ggml.tokens has 15 entries"):
        converter._add_tokenizer(writer, tmp_path, embd_rows=embd_rows)