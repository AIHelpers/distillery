#!/usr/bin/env python3
"""Repair a broken Qwen3 GGUF in place.

The bug: ``trainer/gguf.py`` (pre-fix) wrote ``qwen3.rope.dimension_count``
from the stale ``hidden_size // num_attention_heads`` assumption (64 for
Qwen3-0.6B) while ``qwen3.attention.key_length`` / ``value_length`` were
correctly 128. llama.cpp enforces ``GGML_ASSERT(n_embd_head == n_rot)`` at
load; the mismatch calls native ``abort()``, which is **not** a recoverable
exception — the entire consumer process (e.g. HomeBred-LLM) hard-crashes
with exit code ``0xC0000409`` and no error dialog.

This script rewrites the metadata so ``{arch}.rope.dimension_count`` equals
``{arch}.attention.key_length``. Both values are GGUF ``u32`` (4 bytes), so
the value is overwritten **in place** — no offsets change, tensor data is
byte-identical, and the file keeps its original size. (The metadata section
contains only the KV pairs + tensor-infos header; the weights follow after.)

Usage::

    python scripts/repair_qwen3_rope.py path/to/model.gguf [--dry-run]

Exit codes:
    0  repaired / verified OK / nothing to do
    1  usage error or unsupported file (not GGUF / no Qwen3 keys)
    2  file was broken but repair failed

Example::

    python scripts/repair_qwen3_rope.py task-1-v1-task-1-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF value types we may encounter while walking the KV section.
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Fixed sizes for non-string / non-array scalar types.
_SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(
            f"unexpected end of file: wanted {n} bytes, got {len(data)}"
        )
    return data


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_value(f, ftype: int) -> None:
    """Skip past a KV value of the given GGUF type."""
    if ftype in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[ftype], 1)
        return
    if ftype == GGUF_TYPE_STRING:
        n = _read_u64(f)
        f.seek(n, 1)
        return
    if ftype == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        n_elem = _read_u64(f)
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(n_elem):
                _skip_value(f, GGUF_TYPE_STRING)
        else:
            size = _SCALAR_SIZES.get(elem_type)
            if size is None:
                raise ValueError(f"unsupported GGUF array element type {elem_type}")
            f.seek(size * n_elem, 1)
        return
    raise ValueError(f"unsupported GGUF value type {ftype}")


def _read_value(f, ftype: int):
    """Read a scalar KV value; returns an int for integral types."""
    if ftype == GGUF_TYPE_UINT32:
        return _read_u32(f)
    if ftype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64):
        return _read_u64(f)
    if ftype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return struct.unpack("<H", _read_exact(f, 2))[0]
    if ftype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return _read_exact(f, 1)[0]
    # Others — skip and return None (we only read integral types).
    _skip_value(f, ftype)
    return None


def analyze(path: Path):
    """Scan a GGUF, return (arch, key_length, value_length, dimension_count).

    ``arch`` is the arch key prefix (e.g. ``"qwen3"``). Keys are matched as
    ``{arch}.attention.key_length`` / ``{arch}.attention.value_length`` /
    ``{arch}.rope.dimension_count``.

    Returns ``None`` if the file isn't a recognized Qwen3-family GGUF.
    """
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file (bad magic)")

        _version = _read_u32(f)
        _n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        fields: dict[str, object] = {}
        positions: dict[str, tuple[int, object]] = {}
        for _ in range(n_kv):
            key = _read_str(f)
            ftype = _read_u32(f)
            # Record the byte offset of the value payload (4-byte ftype was
            # just consumed, so this points at the value).
            value_offset = f.tell()
            value = _read_value(f, ftype)
            if ftype == GGUF_TYPE_STRING:
                fields[key] = value  # value was a str for strings
            elif value is not None:
                fields[key] = value
            positions[key] = (value_offset, ftype)

    # Match Qwen3-family arch keys: "qwen3", "qwen2", "qwen2moe", etc.
    arch_key = None
    for key in fields:
        if (
            key.endswith(".attention.key_length")
            or key.endswith(".attention.value_length")
            or key.endswith(".rope.dimension_count")
        ):
            prefix, _, _ = key.rpartition(".")
            arch_key = prefix.rsplit(".", 1)[0] if "." in prefix else None
            break

    if arch_key is None:
        return None

    key_length = fields.get(f"{arch_key}.attention.key_length")
    value_length = fields.get(f"{arch_key}.attention.value_length")
    dimension_count = fields.get(f"{arch_key}.rope.dimension_count")

    return {
        "arch": arch_key,
        "key_length": key_length,
        "value_length": value_length,
        "dimension_count": dimension_count,
        "dimension_count_offset": positions.get(
            f"{arch_key}.rope.dimension_count", (None, None)
        )[0],
        "dimension_count_type": positions.get(
            f"{arch_key}.rope.dimension_count", (None, None)
        )[1],
    }


def repair(path: Path, *, dry_run: bool = False) -> int:
    """Rewrite dimension_count in place to match key_length.

    Returns 0 on success (including "nothing to fix" and repaired),
    1 if the file is not a Qwen GGUF, 2 on failure.
    """
    try:
        info = analyze(path)
    except (ValueError, OSError, struct.error) as exc:
        print(f"repair: {exc}", file=sys.stderr)
        return 1

    if info is None:
        print(
            f"repair: {path} has no Qwen3-style attention/rope metadata — "
            "nothing to do.",
            file=sys.stderr,
        )
        return 1

    arch = info["arch"]
    key_length = info["key_length"]
    dim_count = info["dimension_count"]
    dim_offset = info["dimension_count_offset"]
    dim_type = info["dimension_count_type"]
    value_length = info["value_length"]

    if key_length is None or value_length is None:
        print(
            f"repair: {path} has rope.dimension_count but no "
            f"{arch}.attention.key_length/value_length — cannot determine "
            "the correct value; skipping.",
            file=sys.stderr,
        )
        return 1

    if dim_count is None:
        print(
            f"repair: {path} has no {arch}.rope.dimension_count — nothing "
            "to fix.",
            file=sys.stderr,
        )
        return 0

    if dim_count == key_length:
        print(
            f"repair: {path} OK — {arch}.rope.dimension_count "
            f"({dim_count}) already equals {arch}.attention.key_length "
            f"({key_length}). No changes."
        )
        return 0

    if dim_type != GGUF_TYPE_UINT32:
        print(
            f"repair: {path} {arch}.rope.dimension_count is type "
            f"{dim_type}, not u32 — cannot safely patch in place; skipping.",
            file=sys.stderr,
        )
        return 1

    if dim_offset is None:
        print(
            f"repair: {path} {arch}.rope.dimension_count value offset not "
            "found — skipping.",
            file=sys.stderr,
        )
        return 1

    print(
        f"repair: {path}: {arch}.rope.dimension_count = {dim_count}, "
        f"{arch}.attention.key_length = {key_length} — "
        f"{'would set' if dry_run else 'setting'} to {key_length}.",
    )

    if dry_run:
        return 0

    with open(path, "r+b") as f:
        f.seek(dim_offset)
        f.write(struct.pack("<I", key_length))
        f.flush()
        import os
        os.fsync(f.fileno())

    print(
        f"repair: patched {path} in place — {arch}.rope.dimension_count "
        f"is now {key_length} (was {dim_count}). Tensor data unchanged."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair a broken Qwen3 GGUF: set arch.rope.dimension_count "
        "to match arch.attention.key_length (in place, metadata only)."
    )
    parser.add_argument(
        "gguf_path", nargs="+", help="GGUF file(s) to repair."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying any file.",
    )
    args = parser.parse_args(argv)

    rc = 0
    for path_str in args.gguf_path:
        path = Path(path_str)
        if not path.is_file():
            print(f"repair: file not found: {path}", file=sys.stderr)
            rc = 1
            continue
        r = repair(path, dry_run=args.dry_run)
        if r != 0:
            rc = r
    return rc


if __name__ == "__main__":
    sys.exit(main())