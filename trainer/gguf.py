#!/usr/bin/env python3
"""GGUF conversion for Distillery trained models.

Given a job directory containing a trained LoRA adapter (or a merged model),
this module:

  1. merges the LoRA adapter into the base model weights (when adapter present),
  2. writes the merged PyTorch model + tokenizer to a temp directory,
  3. converts the merged HuggingFace model to GGUF via the pure-Python
     ``gguf`` package (or llama.cpp's ``convert_hf_to_gguf`` as fallback),
  4. quantizes the GGUF inline during conversion (e.g. ``q4_k_m``) using the
     pure-Python ``gguf.quants.quantize`` function — no separate read-back
     pass needed. Falls back to llama.cpp's ``quantize`` binary if needed.

The result is one ``.gguf`` file placed at ``job-dir/gguf/<name>.gguf``
plus a small ``gguf-manifest.json`` describing the conversion.

Production-ready features:

  * **Pure-Python GGUF** — uses the ``gguf`` package (``pip install gguf``)
    which requires no native compilation. Falls back to llama-cpp-python
    if available.
  * **Inline quantization** — quantization happens during the initial
    tensor write, avoiding a fragile read-back-and-requantize pass that
    fails on some tensor dtypes.
  * **Caching** — if the requested quantization already exists on disk,
    it is served without re-running the pipeline (unless ``--force``).
  * **Concurrency lock** — a lock file prevents two conversions for the
    same job from running simultaneously.
  * **Dependency check** — fails fast with actionable messages if
    GGUF tooling is missing.
  * **Post-conversion verification** — the GGUF is parsed to ensure it is
    loadable before being served.
  * **Progress events** — JSON lines are emitted to stdout (and stderr for
    informational messages) so Go can report conversion progress.

Dependencies (install with ``pip install gguf`` — pure Python, no compilation):

    gguf>=0.10.0
    numpy>=1.17.0

Optional (for adapter merging + base model download):

    torch>=2.1.0
    transformers>=4.40.0
    peft>=0.9.0

Optional fallback (native compilation required):

    llama-cpp-python>=0.2.5
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# fcntl is POSIX-only; on Windows we degrade to a no-op lock.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows.
    fcntl = None  # type: ignore[assignment]

# llama.cpp's converter is shipped inside llama-cpp-python (optional fallback).
try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - only needed for smoke verify.
    Llama = None  # type: ignore[assignment]

VALID_QUANTIZATIONS = {
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1",
    "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k", "q8_0", "f16", "f32",
}

# Default timeout for the whole conversion (seconds). Large base models can
# take a while to download + merge + convert, so default to 2 hours.
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60

# Map our quantization names to gguf.GGMLQuantizationType enum values.
_QUANT_TYPE_MAP = {
    "q2_k": "Q2_K",
    "q3_k_s": "Q3_K",
    "q3_k_m": "Q3_K",
    "q3_k_l": "Q3_K",
    "q4_0": "Q4_0",
    "q4_1": "Q4_1",
    "q4_k_s": "Q4_K",
    "q4_k_m": "Q4_K",
    "q5_0": "Q5_0",
    "q5_1": "Q5_1",
    "q5_k_s": "Q5_K",
    "q5_k_m": "Q5_K",
    "q6_k": "Q6_K",
    "q8_0": "Q8_0",
    "f16": "F16",
    "f32": "F32",
}

# Map HF architecture names to gguf MODEL_ARCH enum member names (uppercase).
_ARCH_MAP = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "QWEN2",
    "Qwen3ForCausalLM": "QWEN3",
    "MistralForCausalLM": "LLAMA",
    "MixtralForCausalLM": "LLAMA",
    "GemmaForCausalLM": "GEMMA",
    "Gemma2ForCausalLM": "GEMMA2",
    "Phi3ForCausalLM": "PHI3",
    "PhiForCausalLM": "PHI3",
}

# Default GGUF format version to emit. We use GGUF v3 by default so that all
# architectures (including qwen3, which is only expressible in v3) export
# correctly out of the box. gguf >= 0.17 already writes v3 by default.
#
# NOTE: GGUF v3 requires a llama.cpp backend built from commit b2588 or newer
# (e.g. LLamaSharp >= 0.28.6). Older consumers (e.g. HomeBred-LLM built with
# LLamaSharp 0.27.0) can only read up to GGUF v2 and will fail to load v3
# files; pass --gguf-version 2 in that case.
DEFAULT_GGUF_VERSION = 3

# Architectures that are only expressible in GGUF v3. These require a backend
# built from llama.cpp >= b2588 (e.g. LLamaSharp >= 0.28.6) and fail to load in
# the older consumer (which only understands GGUF v2).
_REQUIRES_GGUF_V3 = {"QWEN3"}


class GGUFConversionError(RuntimeError):
    """Raised when GGUF conversion or quantization fails."""


# Windows error-code → actionable message map for common storage failures.
# WinError 112 is ERROR_DISK_FULL ("Недостаточно места на диске").
_WINERROR_MESSAGES = {
    112: (
        "disk is full (WinError 112) — free up space on the drive and retry. "
        "The model + quantization needs several GB of free space in "
        "job-dir and the temp dir."
    ),
    28: (
        "disk is full (errno 28 ENOSPC) — free up space on the drive and retry. "
        "The model + quantization needs several GB of free space in "
        "job-dir and the temp dir."
    ),
}


def _format_any_exc(exc: BaseException) -> str:
    """Format an exception for display, unwrapping empty storage errors.

    On Windows, a disk-full (WinError 112) or I/O failure produces an OSError
    whose str() is EMPTY — so ``f"{exc}"`` renders as a trailing colon with
    nothing after it ("failed to download base model '...':"). Worse, torch /
    transformers often WRAP the underlying OSError in a RuntimeError whose own
    str() is also empty, so ``isinstance(exc, OSError)`` is False.

    This unwraps the cause/context chain and returns an actionable message:
      1. If the top exception has a non-empty str(), use it.
      2. Otherwise walk ``__cause__``/``__context__`` for an OSError and use
         ``_format_storage_error`` on it (maps WinError 112 etc.).
      3. Otherwise return a generic note that the error message was empty.
    """
    msg = str(exc)
    if msg:
        return msg

    # Walk the exception chain looking for a storage OSError.
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, OSError):
            storage = _format_storage_error(cursor)
            if storage:
                return storage
        # Surface the innermost exception type so we can classify it.
        cursor = cursor.__cause__ or cursor.__context__

    # A MemoryError on Windows has an EMPTY str() (like WinError 112).
    # Distinguish it from disk-full so the user doesn't free up disk
    # space on the wrong resource. RAM exhaustion during
    # AutoModelForCausalLM.from_pretrained() with torch_dtype=float32 is
    # the common cause: the full model is materialized in RAM, then
    # save_pretrained() copies it again.
    cls_name = type(exc).__name__
    if cls_name == "MemoryError" or "MemoryError" in cls_name:
        return (
            "out of memory (MemoryError — error message empty on Windows). "
            "The conversion loads the full base model into RAM (float32 ≈ "
            "4 bytes/parameter) BEFORE saving it, so a 0.6B-parameter model "
            "needs ≈ 2.4 GiB just for weights, plus ~2 GiB of workspace. "
            "Close other applications / add RAM, or ensure the job runs on "
            "a machine with ≥ 8 GiB free RAM. "
            "Lowering the base-model size also helps."
        )

    return (
        f"{cls_name} (error message empty — this is almost always a "
        f"disk-full / I/O failure on Windows). Free up space on the "
        "job-dir drive and retry."
    )


def _format_storage_error(exc: OSError) -> str:
    """Return a human-readable, actionable message for a storage OSError.

    Handles both Windows (WinError) and POSIX (errno) disk-full conditions,
    plus generic I/O failures. Falls back to the raw message otherwise.
    """
    if exc.winerror is not None and exc.winerror in _WINERROR_MESSAGES:
        return _WINERROR_MESSAGES[exc.winerror]
    if exc.errno is not None and exc.errno in _WINERROR_MESSAGES:
        return _WINERROR_MESSAGES[exc.errno]

    # Generic I/O — include the path if present for context.
    path = getattr(exc, "filename", None)
    if path:
        return f"I/O error while writing to {path}: {exc}"

    return str(exc)


def _wrap_storage_errors(fn):
    """Decorator that converts OSError/IOError into GGUFConversionError.

    This ensures WinError 112 (disk full) and similar storage failures
    surface as a clear, actionable GGUFConversionError instead of a raw
    OS-level error bubbling up as "unexpected error".
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OSError as exc:
            raise GGUFConversionError(
                f"conversion failed: {_format_storage_error(exc)}"
            ) from exc

    return wrapper


# Heuristic minimum free bytes needed for a GGUF conversion. A 7B-parameter
# model in q4_k_m is roughly 4 GB; we allow headroom for the merged HF model
# in the temp dir plus the final GGUF copy. This is a fail-fast sanity check —
# the actual requirement depends on the base model size.
MIN_FREE_BYTES_FOR_GGUF = 2 * 1024 * 1024 * 1024  # 2 GiB


def _check_free_space(path: Path) -> None:
    """Fail fast with an actionable message if the drive has too little space.

    Called at the start of the pipeline so we don't spend 30+ minutes
    downloading/merging/converting only to hit WinError 112 at the end.
    """
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        # Can't determine free space (e.g. network path) — don't block.
        return

    if usage.free < MIN_FREE_BYTES_FOR_GGUF:
        free_gb = usage.free / (1024 ** 3)
        raise GGUFConversionError(
            f"not enough free disk space: only {free_gb:.1f} GiB available on "
            f"{path} (need at least {MIN_FREE_BYTES_FOR_GGUF / (1024 ** 3):.0f} GiB). "
            "The model + quantization needs several GB of free space. "
            "Free up space and retry."
        )


def _deps_ok() -> tuple[bool, str]:
    """Return (ok, message) describing whether GGUF tooling is present."""
    # Primary: pure-Python gguf package (no compilation needed).
    try:
        import gguf  # noqa: F401
        return True, "gguf package found"
    except ImportError:
        pass

    # Fallback: llama-cpp-python ships the converter too.
    try:
        import convert_hf_to_gguf  # noqa: F401
        return True, "llama-cpp-python converter found"
    except ImportError:
        pass

    try:
        from llama_cpp import convert_hf_to_gguf  # type: ignore[import-not-found]
        return True, "llama-cpp-python converter found (llama_cpp module)"
    except ImportError:
        pass

    return False, (
        "GGUF converter not installed. "
        "Install: pip install gguf  (pure Python, no compilation needed)"
    )


def _find_quantize_bin() -> Optional[str]:
    """Find the llama.cpp quantize binary (fallback for native quantization)."""
    candidates = [
        "quantize",
        "llama-quantize",
    ]

    # Search in python site-packages (llama-cpp-python ships binaries).
    try:
        import llama_cpp

        pkg_dir = Path(llama_cpp.__file__).parent
        for name in ("quantize", "llama-quantize", "quantize.exe"):
            candidate = pkg_dir / "bin" / name
            if candidate.exists():
                return str(candidate)
            candidate = pkg_dir / name
            if candidate.exists():
                return str(candidate)
    except ImportError:
        pass

    for name in candidates:
        path = shutil.which(name)
        if path:
            return path

    return None


def _patch_writer_chunked_write(writer) -> None:
    """Monkey-patch GGUFWriter to use chunked file writes on Windows.

    The stock implementation calls numpy.tofile() in two places:
    1. add_tensor() — writes tensor data to a SpooledTemporaryFile when
       use_temp_file=True (happens during the "loading_safetensors" step).
    2. write_tensor_data() — writes tensor data to the final output file
       during the "writing_weights" step.

    On Windows, numpy.tofile() can fail for large arrays with
    'N requested and 0 written' because the OS limits single write
    operations. Writing in 4MB chunks avoids this limitation.
    """
    import numpy as np
    import tempfile
    from gguf import GGUFEndian, WriterState

    CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks

    def _chunked_tofile(fp, tensor):
        """Write tensor data to fp in chunks instead of tensor.tofile(fp)."""
        total_bytes = tensor.nbytes
        written = 0
        flat = tensor.reshape(-1).view(np.uint8)  # treat as raw bytes
        while written < total_bytes:
            end = min(written + CHUNK_SIZE, total_bytes)
            fp.write(flat[written:end].tobytes())
            written = end

    # --- Patch add_tensor to chunk the temp-file write ---
    original_add_tensor = writer.add_tensor

    def chunked_add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        if tensor_endianess is None:
            tensor_endianess = GGUFEndian.BIG if sys.byteorder == 'big' else GGUFEndian.LITTLE

        if tensor_endianess != writer.endianess:
            tensor = tensor.byteswap(inplace=False)

        if writer.use_temp_file and writer.temp_file is None:
            fp = tempfile.SpooledTemporaryFile(mode="w+b", max_size=256 * 1024 * 1024)
            fp.seek(0)
            writer.temp_file = fp

        shape = raw_shape if raw_shape is not None else tensor.shape
        writer.add_tensor_info(name, shape, tensor.dtype, tensor.nbytes, raw_dtype=raw_dtype)

        if writer.temp_file is None:
            writer.tensors[-1][name].tensor = tensor
            return

        # Use chunked write instead of tensor.tofile(writer.temp_file).
        _chunked_tofile(writer.temp_file, tensor)
        writer.write_padding(writer.temp_file, tensor.nbytes)

    writer.add_tensor = chunked_add_tensor

    # --- Patch write_tensor_data to chunk the output-file write ---
    def chunked_write_tensor_data(tensor, tensor_endianess=None):
        if writer.state is not WriterState.TI_DATA and writer.state is not WriterState.WEIGHTS:
            raise ValueError(f'Expected output file to contain tensor info or weights, got {writer.state}')
        assert writer.fout is not None

        if tensor_endianess is None:
            tensor_endianess = GGUFEndian.BIG if sys.byteorder == 'big' else GGUFEndian.LITTLE

        if tensor_endianess != writer.endianess:
            tensor = tensor.byteswap(inplace=False)

        file_id = -1
        for i, tensors in enumerate(writer.tensors):
            if len(tensors) > 0:
                file_id = i
                break

        fout = writer.fout[file_id]

        first_tensor_name = next(iter(writer.tensors[file_id]))
        ti = writer.tensors[file_id].pop(first_tensor_name)
        assert ti.nbytes == tensor.nbytes

        writer.write_padding(fout, fout.tell())

        # Use chunked write instead of tensor.tofile(fout).
        _chunked_tofile(fout, tensor)

        writer.write_padding(fout, tensor.nbytes)
        writer.state = WriterState.WEIGHTS

    writer.write_tensor_data = chunked_write_tensor_data


def _assert_arch_gguf_compat(arch_str: str, arch_enum_name: str, version: int) -> None:
    """Fail fast if the architecture cannot be expressed in GGUF ``version``.

    Some architectures (e.g. qwen3) are only representable in GGUF v3. Writing
    them to a v2 file would produce an unloadable GGUF that fails only at load
    time in the consumer (e.g. HomeBred-LLM's "unsupported GGUF format v2"),
    so we raise an actionable error up front instead.
    """
    if arch_enum_name in _REQUIRES_GGUF_V3 and version < 3:
        raise GGUFConversionError(
            f"architecture {arch_str!r} ({arch_enum_name}) requires GGUF v3, "
            f"but GGUF v{version} was requested. "
            "GGUF v3 needs a llama.cpp backend built from commit b2588 or "
            "newer (e.g. LLamaSharp >= 0.28.6). Choose --gguf-version 3 if "
            "your HomeBred-LLM/llama.cpp build supports it, or use a base "
            "model whose architecture is GGUF v2 compatible (llama, mistral, "
            "gemma2, phi3, qwen2)."
        )


def _force_gguf_version(version: int) -> None:
    """Force the pure-Python gguf writer to emit a specific GGUF version.

    ``gguf >= 0.17`` writes GGUF v3 by default (``constants.GGUF_VERSION == 3``),
    which older llama.cpp backends — including the one bundled with LLamaSharp
    0.27.0 used by HomeBred-LLM — cannot read ("unsupported GGUF format v3").

    The writer reads ``GGUF_VERSION`` lazily from its module globals inside
    ``write_header_to_file``, so patching both ``gguf.constants`` and the
    ``gguf.gguf_writer`` module attribute before converting is sufficient to
    change the emitted version. We patch both (plus ``gguf`` package-level) to
    be robust to how the installed gguf version binds the constant.
    """
    import gguf
    import gguf.constants as gguf_constants

    gguf_constants.GGUF_VERSION = int(version)
    setattr(gguf, "GGUF_VERSION", int(version))

    # The writer module may have already bound the name via
    # `from .constants import GGUF_VERSION` — patch it directly too.
    gguf_writer = sys.modules.get("gguf.gguf_writer")
    if gguf_writer is not None:
        setattr(gguf_writer, "GGUF_VERSION", int(version))


class GGUFConverter:
    """Converts a trained adapter (or merged HF model) to GGUF.

    The converter uses the pure-Python ``gguf`` package as the primary
    conversion/quantization engine (no native compilation required).
    If ``gguf`` is not available, it falls back to llama-cpp-python's
    ``convert_hf_to_gguf`` CLI and ``quantize`` binary.

    Three paths are supported:

    * **Adapter + base model** — the LoRA adapter is merged into the base
      weights with `peft` and then converted.
    * **Merged HF model dir** — a directory containing a full HF model is
      converted directly (no merging step).
    * **Base model only** — downloads the base model from HuggingFace and
      converts it (fallback when no trained weights exist).

    Production features: caching, concurrency lock, dependency preflight,
    and post-conversion verification.
    """

    def __init__(
        self,
        base_model: str,
        job_dir: Path,
        output_name: str,
        quantization: str = "q4_k_m",
        cache_dir: Optional[Path] = None,
        force: bool = False,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        gguf_version: int = DEFAULT_GGUF_VERSION,
    ) -> None:
        self.base_model = base_model
        self.job_dir = job_dir
        self.output_name = output_name
        self.quantization = quantization.strip().lower()
        self.cache_dir = cache_dir
        self.force = force
        self.timeout_seconds = timeout_seconds
        self.gguf_version = int(gguf_version)

        if self.quantization not in VALID_QUANTIZATIONS:
            raise GGUFConversionError(
                f"unsupported quantization {self.quantization!r}; "
                f"choose one of {sorted(VALID_QUANTIZATIONS)}"
            )
        if self.gguf_version not in (2, 3):
            raise GGUFConversionError(
                f"unsupported GGUF version {self.gguf_version!r}; choose 2 or 3"
            )

        self.out_dir = job_dir / "gguf"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    def convert_adapter(self, adapter_dir: Path) -> Path:
        """Merge the adapter into the base model and produce the GGUF."""
        return self._run_pipeline("adapter", adapter_dir)

    def convert_merged(self, model_dir: Path) -> Path:
        """Convert an already-merged HF model directory to GGUF."""
        return self._run_pipeline("merged", model_dir)

    def convert_base_model(self) -> Path:
        """Download the base model from HuggingFace and convert it to GGUF.

        This is the fallback path used when no trained adapter or merged
        model exists on disk (e.g. the simulation training backend, which
        produces no real weights). It produces a real, full-size GGUF of
        the base model rather than fabricating a worthless pseudo-random
        file — so the user gets a loadable llama.cpp/HomeBred-LLM model
        that, while not fine-tuned, is genuinely usable at inference.
        """
        return self._run_pipeline("base", self.out_dir)

    @_wrap_storage_errors
    def _run_pipeline(self, kind: str, source_dir: Path) -> Path:
        """Shared pipeline with caching + locking."""
        # Preflight dependency check (fail fast before loading heavy deps).
        ok, msg = _deps_ok()
        if not ok:
            raise GGUFConversionError(msg)

        # Preflight disk-space check — fail fast instead of running the whole
        # merge+convert only to hit WinError 112 (disk full) at the end.
        try:
            usage = shutil.disk_usage(str(self.job_dir))
            free_gb = usage.free / (1024 ** 3)
            self._log_event(
                "disk_space_check",
                free_bytes=usage.free,
                free_gb=round(free_gb, 2),
                path=str(self.job_dir),
            )
        except OSError:
            pass  # can't determine free space — don't block conversion.
        _check_free_space(self.job_dir)

        final = self.out_dir / f"{self.output_name}-{self.quantization}.gguf"

        # Cache hit: if the requested file already exists and we're not
        # forcing a rebuild, serve it directly.
        if not self.force and final.exists() and final.stat().st_size > 0:
            self._log_event("cache_hit", file=final.name, quantization=self.quantization)
            return final

        # Concurrency lock: one conversion per job at a time.
        lock_path = self.out_dir / ".gguf.lock"
        lock_acquired = self._acquire_lock(lock_path)
        if not lock_acquired:
            # Another conversion is in-flight — wait for it (with timeout).
            self._log_event(
                "lock_wait",
                message=f"waiting for another GGUF conversion of {self.output_name}",
            )
            self._wait_for_lock(lock_path)

            # After waiting, re-check cache.
            if not self.force and final.exists() and final.stat().st_size > 0:
                self._log_event("cache_hit_after_wait", file=final.name, quantization=self.quantization)
                return final

        try:
            self._log_event("conversion_started", quantization=self.quantization, kind=kind)

            # Use a temp dir without auto-cleanup context manager — on
            # Windows, lingering memmap/torch handles can block deletion
            # inside __exit__, raising WinError 32. We clean up manually
            # with gc + retry instead.
            #
            # IMPORTANT: create the temp dir ON THE SAME DRIVE as job_dir.
            # The merged HF model + intermediate GGUF can grow to several
            # GB; if we used tempfile's default TEMP dir (often a different
            # drive), the disk-space preflight above (which checks job_dir)
            # would be meaningless and the actual write could fail with
            # WinError 112 (disk full) — an OSError whose str() is EMPTY on
            # Windows, surfacing as "failed to download base model: " with
            # nothing after the colon.
            tmp = tempfile.mkdtemp(prefix="distillery-gguf-", dir=str(self.job_dir))
            try:
                convert_dir = None
                if kind == "adapter":
                    merged_dir = Path(tmp) / "merged"
                    self._merge_adapter(source_dir, merged_dir)
                    convert_dir = merged_dir
                elif kind == "base":
                    # No trained weights — download the base model from
                    # HuggingFace into a temp dir and convert that directly.
                    convert_dir = Path(tmp) / "base"
                    self._download_base_model(convert_dir)
                else:
                    convert_dir = source_dir

                gguf_path = self._convert_dir(convert_dir)

                # The pure-Python converter quantizes inline (single-pass),
                # so the output is already at the requested quantization.
                # The llama.cpp fallback produces f16; only then do we need
                # a separate quantization pass.
                if (
                    self.quantization
                    and self.quantization not in ("f16", "f32")
                    and "-f16" in gguf_path.name
                ):
                    gguf_path = self._quantize(gguf_path)

                self._emit(gguf_path, final)
                self._verify(final)
            finally:
                # Force garbage collection to release any lingering file
                # handles (memmap, torch tensors, etc.) before cleanup.
                import gc
                gc.collect()

                # Best-effort cleanup — on Windows, some handles may still
                # be held by the OS for a moment. Retry a few times.
                _safe_rmtree(Path(tmp))
        finally:
            self._release_lock(lock_path)

        return final

    # ------------------------------------------------------------------ #
    # Locking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _acquire_lock(lock_path: Path) -> bool:
        """Try to acquire an exclusive lock; returns True if acquired."""
        if fcntl is None:
            # Non-POSIX platform (Windows) — degrade to no locking.
            return True

        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Store the fd so we can release later.
                _gc_lock_fds[lock_path] = fd
                return True
            except BlockingIOError:
                os.close(fd)
                return False
        except (OSError, ValueError):
            # Non-POSIX platform (Windows) or missing fcntl — degrade to no locking.
            return True

    @staticmethod
    def _wait_for_lock(lock_path: Path, timeout: int = 2 * 60 * 60) -> None:
        """Block until the lock is released or the timeout expires."""
        if fcntl is None:
            # Non-POSIX platform (Windows) — locking is a no-op.
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fd = os.open(str(lock_path), os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Acquired — release immediately (we only need to know
                    # the other process finished).
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                    return
                except BlockingIOError:
                    os.close(fd)
            except (OSError, ValueError):
                # No lock file or non-POSIX — treat as released.
                return

            time.sleep(5)

        raise GGUFConversionError(
            f"timed out waiting for another GGUF conversion ({timeout}s)"
        )

    @staticmethod
    def _release_lock(lock_path: Path) -> None:
        """Release the lock for the given path (if held)."""
        fd = _gc_lock_fds.pop(lock_path, None)
        if fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except (OSError, ValueError):
                pass
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _log_event(event_type: str, **fields: object) -> None:
        """Emit a JSON event line to stdout for Go to tail."""
        rec = {"type": event_type, "timestamp": time.time()}
        rec.update(fields)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    # ------------------------------------------------------------------ #
    # Merging
    # ------------------------------------------------------------------ #
    def _merge_adapter(self, adapter_dir: Path, merged_dir: Path) -> None:
        if not adapter_dir.is_dir():
            raise GGUFConversionError(f"adapter dir not found: {adapter_dir}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log_event(
            "merge_started",
            adapter=adapter_dir.name,
            base_model=self.base_model,
            device=device,
        )

        try:
            # Load in float16 on CUDA for speed; float32 on CPU. Keeps
            # weights on CPU in float32 because LoRA merge_and_unload()
            # needs full precision. low_cpu_mem_usage=True allocates tensors
            # per-shard on the meta device instead of all at once — reducing
            # peak RAM from ~2× the model size to ~1.2×.
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to(device)

            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()

            merged_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(merged_dir))

            # Load the tokenizer from the **base model repo**, NOT from
            # adapter_dir (which only contains LoRA weights — no tokenizer
            # files). For BPE tokenizers (Qwen3, GPT-2, etc.) this guarantees
            # the full tokenizer is present, including the merge rules that
            # llama.cpp / HomeBred-LLM require to load the GGUF ("Cannot find
            # Llama BPE tokenizer" occurs when merges.txt / vocab.json are
            # missing).
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(merged_dir))

            # Belt-and-braces: some tokenizer classes' save_pretrained() omit
            # the BPE artifacts (merges.txt / vocab.json) even though the
            # tokenizer.json references them. Explicitly copy every tokenizer
            # artifact from the base HF cache into the merged dir so both the
            # pure-Python and llama.cpp converters always find a complete
            # tokenizer.
            self._copy_tokenizer_artifacts(self.base_model, merged_dir)

            self._log_event("merge_completed", merged_dir=str(merged_dir))
        except GGUFConversionError:
            raise
        except OSError as exc:
            # On Windows, disk-full (WinError 112) and similar storage
            # failures produce an OSError whose str() is EMPTY — the user
            # would see "merge failed:" with nothing after. _format_storage_error
            # converts it into the actionable "disk is full — free up space"
            # message.
            raise GGUFConversionError(
                f"merge failed: {_format_storage_error(exc)}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise GGUFConversionError(f"merge failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Tokenizer artifact copying
    # ------------------------------------------------------------------ #
    def _copy_tokenizer_artifacts(self, repo_id: str, dest_dir: Path) -> None:
        """Copy every tokenizer artifact for ``repo_id`` into ``dest_dir``.

        ``tokenizer.save_pretrained()`` writes ``tokenizer.json`` but — for
        BPE tokenizers (Qwen3, GPT-2, etc.) — does **not** always write the
        ``vocab.json`` / ``merges.txt`` files that llama.cpp's converter
        requires. The GGUF converters (``gguf`` and llama.cpp's
        ``convert_hf_to_gguf``) read these files directly from the model
        directory; if they are absent, llama.cpp raises
        "Cannot find Llama BPE tokenizer" and the whole conversion fails.

        This helper downloads only the tokenizer artifact files from the
        HuggingFace repo and copies them into ``dest_dir``, guaranteeing
        both converters find a complete tokenizer regardless of which
        tokenizer class the merge used.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Files the converters may need. ``special_tokens_map.json`` /
        # ``added_tokens.json`` are optional but harmless; the critical
        # ones are tokenizer.json (fast) and vocab.json + merges.txt (BPE).
        allowed_patterns = (
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "preprocessor_config.json",
        )

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            # huggingface_hub is a transitive dep of transformers — if it is
            # somehow unavailable, we can't pull artifacts. The tokenizer we
            # saved via save_pretrained() above is best-effort; log a warning
            # and proceed (the pure-Python path may still succeed on
            # tokenizer.json alone).
            self._log_event(
                "tokenizer_artifacts_skipped",
                reason="huggingface_hub not available",
            )
            return

        try:
            snapshot_dir = snapshot_download(
                repo_id,
                allow_patterns=allowed_patterns,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_dir=None,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_event(
                "tokenizer_artifacts_skipped",
                reason=f"snapshot download failed: {exc}",
            )
            return

        snapshot = Path(snapshot_dir)
        copied = []
        for name in allowed_patterns:
            src = snapshot / name
            if src.is_file():
                shutil.copy2(src, dest_dir / name)
                copied.append(name)

        if copied:
            self._log_event("tokenizer_artifacts_copied", files=copied)
        else:
            self._log_event(
                "tokenizer_artifacts_skipped",
                reason="no tokenizer artifact files found in repo snapshot",
            )

    # ------------------------------------------------------------------ #
    # Base-model download (no trained weights fallback)
    # ------------------------------------------------------------------ #
    def _download_base_model(self, dest_dir: Path) -> None:
        """Download the base model + tokenizer from HuggingFace into dest_dir.

        Used when no trained adapter/merged model exists on disk: rather
        than fabricating a fake GGUF, we produce a real, full-size GGUF of
        the base model itself. The result is not fine-tuned, but it is a
        genuine, loadable llama.cpp/HomeBred-LLM model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log_event(
            "base_model_download_started",
            base_model=self.base_model,
            device=device,
        )

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)

            # low_cpu_mem_usage=True keeps weights on the meta device and
            # allocates them per-shard during from_pretrained — cutting peak
            # RAM from ~2× the model size down to ~1.2×. float16 also halves
            # the resident weight size vs float32. The GGUF converter reads
            # from the saved safetensors regardless of this dtype (and
            # quantizes anyway in the q4_k_m path), so there's no precision
            # loss in the pipeline.
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
            model.save_pretrained(str(dest_dir))

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(dest_dir))

            # Same BPE-artifact guarantee as the adapter path: llama.cpp's
            # converter needs merges.txt / vocab.json for BPE tokenizers
            # (Qwen3, GPT-2, etc.). save_pretrained() alone may omit them.
            self._copy_tokenizer_artifacts(self.base_model, dest_dir)

            self._log_event("base_model_download_completed", dest_dir=str(dest_dir))
        except OSError as exc:
            # On Windows, disk-full (WinError 112) and similar storage
            # failures produce an OSError whose str() is EMPTY — the user
            # sees "failed to download base model '...':" with nothing after
            # the colon. _format_storage_error converts it into the
            # actionable "disk is full — free up space" message.
            raise GGUFConversionError(
                f"failed to download base model {self.base_model!r}: "
                f"{_format_storage_error(exc)}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            # NOT bare {exc}: torch/transformers wrap the storage OSError
            # (e.g. WinError 112) in a RuntimeError whose own str() is
            # empty — the user would see "failed to download base model
            # '...':" with nothing after. _format_any_exc unwraps the
            # cause/context chain to surface the real actionable message.
            raise GGUFConversionError(
                f"failed to download base model {self.base_model!r}: "
                f"{_format_any_exc(exc)}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def _convert_dir(self, model_dir: Path) -> Path:
        """Convert a HuggingFace model directory to GGUF.

        Uses the pure-Python ``gguf`` package (``pip install gguf``) as the
        primary path — no native compilation required. Falls back to
        llama-cpp-python's ``convert_hf_to_gguf`` CLI if ``gguf`` is not
        available.
        """
        # Try the pure-Python gguf package first.
        pp_output = None
        try:
            pp_output = self._convert_dir_pure_python(model_dir)
        except ImportError:
            pass

        if pp_output is not None:
            # Verify the pure-Python output is loadable. If it fails, fall
            # back to the llama.cpp converter rather than serving a broken
            # file.
            try:
                self._verify_loadable(pp_output)
                return pp_output
            except GGUFConversionError as exc:
                self._log_event(
                    "pure_python_verify_failed",
                    message=f"pure-Python GGUF failed load verification: {exc}",
                )
                try:
                    pp_output.unlink()
                except OSError:
                    pass

        # Fallback: llama-cpp-python's convert_hf_to_gguf CLI.
        return self._convert_dir_llamacpp(model_dir)

    def _convert_dir_pure_python(self, model_dir: Path) -> Path:
        """Convert using the pure-Python gguf package — no compilation needed.

        When a quantization is requested (e.g. q4_k_m), the quantization
        happens **inline** during tensor writing — we call
        ``gguf.quants.quantize`` on each weight tensor (which is a clean
        float32 numpy array from safetensors) and write the quantized bytes
        directly. This avoids the fragile two-pass approach of writing f16
        first and then reading it back with GGUFReader (which fails on
        certain tensor dtypes).
        """
        import numpy as np
        from gguf import (
            GGUFWriter,
            SafetensorsLocal,
            TensorNameMap,
            MODEL_ARCH,
            GGMLQuantizationType,
        )

        # Determine the output type: quantized or f16/f32.
        is_quantized = self.quantization not in ("f16", "f32")
        quant_type = None
        pp_quantize = None
        if is_quantized:
            quant_type_str = _QUANT_TYPE_MAP.get(self.quantization)
            if quant_type_str is None:
                raise GGUFConversionError(
                    f"unsupported quantization {self.quantization!r} for pure-Python path"
                )
            quant_type = GGMLQuantizationType[quant_type_str]
            try:
                from gguf.quants import quantize as pp_quantize
            except ImportError:
                # No quantize function available — fall through to llamacpp.
                raise

        # Output file: quantized suffix or f16 suffix.
        suffix = self.quantization if is_quantized else "f16"
        outfile = self.out_dir / f"{self.output_name}-{suffix}.gguf"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        self._log_event(
            "convert_hf_to_gguf_started",
            model_dir=str(model_dir),
            outfile=outfile.name,
            mode="pure-python",
            quantization=self.quantization,
            inline_quantize=is_quantized,
        )

        # Detect architecture from config.json.
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise GGUFConversionError(f"no config.json in {model_dir}")

        with open(config_path) as f:
            hparams = json.load(f)

        arch_str = hparams.get("architectures", ["LlamaForCausalLM"])
        if isinstance(arch_str, list):
            arch_str = arch_str[0] if arch_str else "LlamaForCausalLM"

        arch_enum_name = _ARCH_MAP.get(arch_str, "LLAMA")

        try:
            arch = MODEL_ARCH[arch_enum_name]
        except KeyError:
            arch = MODEL_ARCH.LLAMA

        # The writer expects the lowercase arch name string (e.g. "llama").
        from gguf import MODEL_ARCH_NAMES
        arch_name = MODEL_ARCH_NAMES.get(arch, "llama")

        # Compatibility guard: some architectures are only expressible in
        # GGUF v3. Producing a v2 file for them would silently fail at load
        # time in every consumer, so fail fast with an actionable message.
        _assert_arch_gguf_compat(arch_str, arch_enum_name, self.gguf_version)

        # Force the requested GGUF version before the writer emits the header.
        _force_gguf_version(self.gguf_version)

        # Find safetensors files.
        safetensors_files = sorted(model_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise GGUFConversionError(
                f"no .safetensors files in {model_dir}; "
                "ensure the model was saved in safetensors format"
            )

        # GGUFWriter.__init__ already adds general.architecture — do NOT
        # call add_architecture() again (would duplicate the key).
        writer = GGUFWriter(str(outfile), arch_name, use_temp_file=True)

        # Monkey-patch write_tensor_data to use chunked writes instead of
        # numpy.tofile(). On Windows, tofile() can fail for large arrays
        # with "N requested and 0 written" because it issues a single
        # write() call that the OS can't handle in one go. Writing in
        # smaller chunks avoids this limitation.
        _patch_writer_chunked_write(writer)

        # Add model metadata from config.
        n_layers = hparams.get("num_hidden_layers", 32)
        n_heads = hparams.get("num_attention_heads", 32)
        hidden_size = hparams.get("hidden_size", 4096)

        writer.add_name(self.output_name)
        writer.add_context_length(hparams.get("max_position_embeddings", 4096))
        writer.add_embedding_length(hidden_size)
        writer.add_block_count(n_layers)
        writer.add_head_count(n_heads)
        writer.add_head_count_kv(
            hparams.get("num_key_value_heads", n_heads)
        )

        # Attention head dimension: Qwen3 (and some other architectures)
        # decouple this from hidden_size // num_attention_heads via an
        # explicit head_dim field in config.json. Without
        # key_length/value_length metadata, llama.cpp falls back to the
        # default n_embd/n_head and rejects the (correctly-shaped)
        # tensors as a shape mismatch (check_tensor_dims).
        #
        # Fallback for architectures that don't specify head_dim
        # explicitly: hidden_size // num_attention_heads is correct for
        # those (the default llama.cpp uses).
        head_dim = hparams.get("head_dim") or (hidden_size // n_heads)
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)

        writer.add_feed_forward_length(hparams.get("intermediate_size", 11008))
        # n_rot must equal n_embd_head (which the key_length/value_length
        # above now set to head_dim). Writing hidden_size // n_heads here
        # instead of head_dim leaves n_rot=64 while n_embd_head=128 for
        # Qwen3-0.6B — llama.cpp asserts n_embd_head == n_rot and aborts.
        # The fallback (hidden_size // n_heads) is already folded into
        # head_dim when the config has no explicit head_dim.
        writer.add_rope_dimension_count(head_dim)
        writer.add_rope_freq_base(hparams.get("rope_theta", 10000.0))
        writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-6))

        # ---- First pass (metadata-only): collect the embedding row count
        # ---- without reading any tensor data. The embedding matrix
        # ---- (token_embd.weight) is the ground-truth vocab size the source
        # ---- checkpoint was padded to; the tokenizer's defined BPE vocab may
        # ---- be shorter (Qwen2/2.5/3 reserve ~293 rows for hardware
        # ---- alignment). The token list must be padded to match the embedding
        # ---- row count — llama.cpp refuses to load a GGUF where
        # ---- tokenizer.ggml.tokens and token_embd.weight disagree.
        embd_rows: Optional[int] = None

        # Scan safetensors headers only (SafetensorsLocal.__init__ reads just
        # the JSON metadata, not tensor data — cheap) to find
        # model.embed_tokens.weight's first dimension.
        for st_file in safetensors_files:
            reader = SafetensorsLocal(st_file)
            for name, tensor in reader.tensors.items():
                if name == "model.embed_tokens.weight":
                    embd_rows = int(tensor.shape[0])
                    break
            if embd_rows is not None:
                break

        if embd_rows is None:
            raise GGUFConversionError(
                "safetensors checkpoint has no 'model.embed_tokens.weight' "
                "tensor — cannot determine the embedding row count needed "
                "to reconcile tokenizer.ggml.tokens against token_embd.weight."
            )

        # Build tensor name map (used by the first-pass loop below).
        tensor_map = TensorNameMap(arch, n_layers)

        # Add tokenizer metadata — hard gate. A GGUF with zero tokenizer
        # keys is an incomplete export: llama.cpp / HomeBred-LLM cannot
        # tokenize prompts and the file is unloadable at inference time.
        # Pass embd_rows so _add_tokenizer can pad the token list to match
        # the embedding matrix (the Qwen-style reserved-slot pattern).
        self._add_tokenizer(writer, model_dir, embd_rows)

        # Safetensors dtype string → numpy dtype.
        # NOTE: BF16 is 2 bytes/element — we can't map it directly to np.float32
        # (4 bytes) because np.frombuffer would misread the raw bytes. Instead we
        # handle BF16 specially below by reading as uint16 and upcasting.
        st_dtype_map = {
            "F64": np.float64,
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,  # special-cased below
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
            "BOOL": np.bool_,
        }

        # ---- First pass: collect tensor info ----
        # For quantized output, we need to test-quantize each tensor to
        # determine if it can be quantized (some tensors like embeddings
        # may need to stay in f32). We collect the data type for each.
        tensor_infos = []  # list of (gguf_name, shape, np_dtype, nbytes, raw_dtype)
        tensor_data_map = {}  # gguf_name → numpy array

        # Actual (logical, pre-quantization) output dimension of
        # blk.0.attn_q.weight — used by the hard gate below that verifies
        # the attention metadata (key_length/value_length x head_count)
        # matches the tensor shape we're about to write.
        attn_q_out_dim_actual: Optional[int] = None

        for st_file in safetensors_files:
            self._log_event("loading_safetensors", file=st_file.name)
            reader = SafetensorsLocal(st_file)
            st_file_path = str(st_file)

            for name, tensor in reader.tensors.items():
                # BUG FIX: TensorNameMap stores its lookup table with the
                # ".weight"/".bias" suffix already stripped (e.g. the key for
                # "model.layers.0.self_attn.q_proj.weight" is stored as
                # "model.layers.{bid}.self_attn.q_proj", not the full name
                # with suffix). get_name() only strips a suffix itself when
                # you pass try_suffixes — called with no arguments (as this
                # line previously was), it does a raw dict lookup on the
                # *full* key, which is never present, so it returns None for
                # every single tensor. Combined with the "or name" fallback
                # below, that meant EVERY tensor silently fell through to its
                # original HuggingFace name (verified directly against the
                # installed gguf package: TensorNameMap(...).get_name(
                # "model.embed_tokens.weight") returns None, while
                # get_name(..., try_suffixes=(".weight", ".bias")) correctly
                # returns "token_embd.weight"). The resulting GGUF looked
                # complete (right size, right tensor count, valid header) but
                # was unloadable by llama.cpp, which requires exact
                # llama.cpp-style names (blk.N.attn_q.weight, etc).
                gguf_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))

                # Record the actual output dimension of blk.0.attn_q.weight
                # (from its source HF q_proj tensor shape) — used by the hard
                # gate below. Note: safetensors shapes are (out_features,
                # in_features), so the first shape element is the output dim.
                if gguf_name == "blk.0.attn_q.weight":
                    attn_q_out_dim_actual = int(tensor.shape[0])

                if gguf_name is None:
                    # A tensor this architecture's TensorNameMap doesn't
                    # recognize (e.g. a rotary-embedding buffer some HF
                    # checkpoints include, or a future model detail this
                    # mapping table doesn't cover yet). Do NOT fall back to
                    # the raw HuggingFace name — writing it through
                    # unmapped is exactly how the original bug produced
                    # unloadable GGUFs. llama.cpp doesn't need tensors it
                    # doesn't recognize by name, so skip and log instead.
                    self._log_event(
                        "tensor_skipped_unmapped",
                        tensor=name,
                        reason="no llama.cpp tensor-name mapping for this architecture",
                    )
                    continue

                # Read raw bytes directly from the file (NOT via np.memmap,
                # which keeps a file handle open on Windows and prevents
                # tempfile.TemporaryDirectory from deleting the temp dir).
                # Using a regular open() ensures the handle is closed
                # immediately after reading.
                np_dtype = st_dtype_map.get(tensor.dtype, np.float32)
                offset = tensor.data_range.offset
                size = tensor.data_range.size
                with open(st_file_path, "rb") as f:
                    f.seek(offset)
                    raw_bytes = f.read(size)

                # BF16: read as uint16, then upcast to float32 by shifting
                # the 16-bit mantissa into the upper 16 bits of a float32.
                if tensor.dtype == "BF16":
                    data_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
                    data = (data_u16.astype(np.uint32) << 16).view(np.float32).reshape(tensor.shape).copy()
                else:
                    data = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(tensor.shape).copy()
                del raw_bytes  # free the raw bytes immediately

                # Convert to float32 for GGUF output.
                data_f32 = data.astype(np.float32) if data.dtype != np.float32 else data

                if is_quantized and pp_quantize is not None:
                    # Try to quantize this tensor inline.
                    try:
                        quantized = pp_quantize(data_f32, quant_type)
                        # quantized is a uint8 numpy array with a *byte shape*
                        # (different from the original float32 shape). We must
                        # pass raw_dtype=quant_type so the writer knows it's
                        # quantized data, and the byte shape so it can compute
                        # the logical shape via quant_shape_from_byte_shape.
                        tensor_infos.append(
                            (gguf_name, quantized.shape, np.uint8,
                             quantized.nbytes, quant_type)
                        )
                        tensor_data_map[gguf_name] = quantized
                    except Exception:  # noqa: BLE001
                        # Can't quantize this tensor (e.g. it's a 1D norm
                        # weight) — keep it in f32.
                        tensor_infos.append(
                            (gguf_name, tensor.shape, np.float32,
                             data_f32.nbytes, GGMLQuantizationType.F32)
                        )
                        tensor_data_map[gguf_name] = data_f32
                else:
                    # No quantization — write as f32.
                    tensor_infos.append(
                        (gguf_name, tensor.shape, np.float32,
                         data_f32.nbytes, GGMLQuantizationType.F32)
                    )
                    tensor_data_map[gguf_name] = data_f32

        # Hard gate: a GGUF missing the embedding/output tensors or any
        # per-block tensors is unloadable by llama.cpp no matter how
        # complete its metadata looks (this is exactly how the
        # get_name()-without-try_suffixes bug above shipped broken files —
        # the header, tensor count and tokenizer metadata all looked fine).
        # Fail loudly here instead of writing a file that only fails at
        # load time on the consumer's machine.
        mapped_names = {info[0] for info in tensor_infos}
        if "token_embd.weight" not in mapped_names:
            raise GGUFConversionError(
                "tensor-name mapping produced no 'token_embd.weight' — the "
                "embedding tensor didn't resolve to a llama.cpp-style name. "
                "This GGUF would be unloadable; refusing to write it. "
                f"(architecture={arch_name!r})"
            )
        if not any(n.startswith("blk.0.") for n in mapped_names):
            raise GGUFConversionError(
                "tensor-name mapping produced no 'blk.0.*' tensors — none of "
                "this model's per-layer weights resolved to llama.cpp-style "
                "names. This GGUF would be unloadable; refusing to write it. "
                f"(architecture={arch_name!r})"
            )

        # Hard gate: the actual attention-query output dimension (from the
        # source tensor shapes) must equal what the GGUF metadata implies
        # (head_count x head_dim). This would have caught the Qwen3 issue
        # BEFORE writing any file: without explicit key_length/value_length
        # metadata, llama.cpp computes head_dim as hidden_size/n_head (the
        # fallback) and rejects the correctly-shaped tensors as a shape
        # mismatch (check_tensor_dims). The tensors are right; the metadata
        # was the bug. Check it for ANY architecture, not just Qwen3.
        expected_q_out = head_dim * n_heads
        if attn_q_out_dim_actual is not None and expected_q_out != attn_q_out_dim_actual:
            raise GGUFConversionError(
                f"blk.0.attn_q.weight will be written with output dimension "
                f"{attn_q_out_dim_actual}, but the GGUF metadata implies "
                f"{expected_q_out} (head_count={n_heads} x head_dim={head_dim}). "
                "llama.cpp will reject this. Check whether this architecture "
                "needs explicit key_length/value_length metadata (e.g. Qwen3's "
                "head_dim is NOT hidden_size/num_heads). "
                f"(architecture={arch_name!r})"
            )
        # Add the file type metadata so llama.cpp knows the quantization type.
        # Without this, llama.cpp may default to F32 and reject quantized tensors.
        # The enum was renamed from GGUFFileType to LlamaFileType in gguf 0.10+.
        try:
            from gguf import GGUFFileType as _FileType
        except ImportError:
            from gguf import LlamaFileType as _FileType
        _FILE_TYPE_MAP = {
            "q2_k": _FileType.MOSTLY_Q2_K,
            "q3_k_s": _FileType.MOSTLY_Q3_K_S,
            "q3_k_m": _FileType.MOSTLY_Q3_K_M,
            "q3_k_l": _FileType.MOSTLY_Q3_K_L,
            "q4_0": _FileType.MOSTLY_Q4_0,
            "q4_1": _FileType.MOSTLY_Q4_1,
            "q4_k_s": _FileType.MOSTLY_Q4_K_S,
            "q4_k_m": _FileType.MOSTLY_Q4_K_M,
            "q5_0": _FileType.MOSTLY_Q5_0,
            "q5_1": _FileType.MOSTLY_Q5_1,
            "q5_k_s": _FileType.MOSTLY_Q5_K_S,
            "q5_k_m": _FileType.MOSTLY_Q5_K_M,
            "q6_k": _FileType.MOSTLY_Q6_K,
            "q8_0": _FileType.MOSTLY_Q8_0,
            "f16": _FileType.MOSTLY_F16,
            "f32": _FileType.ALL_F32,
        }
        ft = _FILE_TYPE_MAP.get(self.quantization, _FileType.ALL_F32)
        writer.add_file_type(ft)

        # Add tensor info to writer.
        # add_tensor_info signature:
        #   (name, tensor_shape, tensor_dtype: np.dtype, tensor_nbytes,
        #    raw_dtype: GGMLQuantizationType | None = None)
        # When raw_dtype is set and tensor_dtype is uint8, the writer knows
        # the data is quantized and calls quant_shape_from_byte_shape to get
        # the logical shape. When raw_dtype is None, it infers the GGUF dtype
        # from the numpy dtype (must be F16/F32/F64/I8/I16/I32/I64).
        for name, shape, np_dtype, nbytes, raw_dtype in tensor_infos:
            writer.add_tensor_info(name, shape, np_dtype, nbytes,
                                   raw_dtype=raw_dtype)

        # Write header + KV + tensor info.
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        # ---- Second pass: write tensor data ----
        for name, data in tensor_data_map.items():
            writer.write_tensor_data(data)

        writer.close()

        if not outfile.exists():
            raise GGUFConversionError("GGUF writer produced no output file")

        self._log_event(
            "convert_hf_to_gguf_completed",
            outfile=outfile.name,
            size_bytes=outfile.stat().st_size,
            mode="pure-python",
            quantization=self.quantization,
            inline_quantize=is_quantized,
        )

        # If we quantized inline, also emit a quantize_completed event
        # so the Go progress tracker advances to ~90%.
        if is_quantized:
            self._log_event(
                "quantize_completed",
                dest=outfile.name,
                size_bytes=outfile.stat().st_size,
                mode="pure-python-inline",
            )

        return outfile

    def _add_tokenizer(self, writer, model_dir: Path, embd_rows: int) -> None:
        """Add tokenizer metadata to the GGUF writer.

        Dispatches on the tokenizer layout found in ``model_dir``:

        * ``vocab.json`` + ``merges.txt`` (BPE; Qwen3, GPT-2, ...) → ``BpeVocab``
        * ``tokenizer.model`` (SentencePiece; Llama, Mistral, ...) → ``SentencePieceVocab``
        * ``tokenizer.json`` only (HuggingFace fast) → ``LlamaHfVocab``

        Raises GGUFConversionError if the tokenizer files are missing or
        unreadable — a GGUF with zero tokenizer keys is unloadable by
        llama.cpp/HomeBred-LLM and must not be served as a "successful"
        export. The BPE path is critical: hard-coding ``LlamaHfVocab`` for
        Qwen3's generic BPE (non-Llama-3-shaped decoder) makes llama.cpp
        raise "Cannot find Llama BPE tokenizer".
        """
        from gguf import BpeVocab, LlamaHfVocab, SentencePieceVocab, SpecialVocab

        # Preflight: the source must contain the tokenizer files. A GGUF
        # without tokenizer metadata is an incomplete export — llama.cpp
        # cannot tokenize prompts. Fail with an actionable message instead
        # of silently producing a useless file.
        tokenizer_file = model_dir / "tokenizer.json"
        vocab_json = model_dir / "vocab.json"
        merges_txt = model_dir / "merges.txt"
        tok_vocab = model_dir / "tokenizer.model"

        if (
            not tokenizer_file.exists()
            and not vocab_json.exists()
            and not tok_vocab.exists()
        ):
            raise GGUFConversionError(
                f"cannot add tokenizer metadata: no tokenizer.json, vocab.json, "
                f"or tokenizer.model found in {model_dir}. "
                "A GGUF export without tokenizer metadata is unloadable by "
                "llama.cpp / HomeBred-LLM (no tokens to map prompt text). "
                "Ensure the training job saved its tokenizer alongside the "
                "model weights, then retry the export."
            )

        # ---- Choose the tokenizer backend based on what's on disk. ----
        if vocab_json.exists() and merges_txt.exists():
            # Generic BPE (Qwen3, GPT-2, ...). LlamaHfVocab rejects these
            # ("Cannot find Llama BPE tokenizer") unless they are the exact
            # Llama-3-shaped BPE, so use BpeVocab which reads vocab.json +
            # merges.txt directly.
            vocab = BpeVocab(model_dir)
            tokenizer_model = getattr(vocab, "tokenizer_model", None) or "gpt2"
        elif tok_vocab.exists():
            # SentencePiece (Llama, Mistral, Gemma, ...).
            vocab = SentencePieceVocab(model_dir)
            tokenizer_model = getattr(vocab, "tokenizer_model", None) or "llama"
        else:
            # HuggingFace fast tokenizer (tokenizer.json). LlamaHfVocab only
            # accepts the Llama-3-shaped BPE layout; anything else raises
            # "Cannot find Llama BPE tokenizer". If the layout doesn't match
            # and we have vocab.json+merges.txt, we'd already be in the BPE
            # branch above — so reaching here means a real Llama-3 / Llama2
            # fast tokenizer.
            try:
                vocab = LlamaHfVocab(model_dir)
            except FileNotFoundError:
                raise GGUFConversionError(
                    f"{model_dir.name} tokenizer.json is not a supported BPE "
                    "layout (LlamaHfVocab rejected it: 'Cannot find Llama BPE "
                    "tokenizer'). Ensure vocab.json + merges.txt are present "
                    "for BPE models, or tokenizer.model for SentencePiece."
                ) from None
            except TypeError:
                raise GGUFConversionError(
                    f"{model_dir.name} tokenizer.json is Llama 3 BPE — this "
                    "build cannot convert it (use a tokenizer.model-based "
                    "model or BPE with vocab.json + merges.txt)."
                ) from None
            tokenizer_model = getattr(vocab, "tokenizer_model", None) or "llama"

        # NOTE: vocab.all_tokens() is a GENERATOR — materialize it to a list
        # once. Every consumer below (n_vocab, empty check, iteration) must
        # reuse this single list; calling all_tokens() again would create a
        # fresh exhausted/positioned generator and len() on it would raise
        # "object of type 'generator' has no len()".
        raw_tokens = list(vocab.all_tokens())

        special_vocab = SpecialVocab(model_dir, load_merges=True,
                                     n_vocab=len(raw_tokens))
        if not raw_tokens:
            raise GGUFConversionError(
                "tokenizer metadata produced an empty token list — refusing "
                "to write a GGUF with zero tokens (unloadable by llama.cpp). "
                "Ensure the tokenizer files are valid."
            )

        # Manually add token list + scores + types to the writer.
        #
        # NOTE: raw_tokens entries are (text, score, TokenType) triples
        # for BpeVocab / SentencePieceVocab, but plain bytes for
        # LlamaHfVocab. GGUFWriter.add_token_list only accepts the token
        # text (bytes or str) — we unpack the tuple element here.
        tokens = []
        token_scores: list[float] = []
        for item in raw_tokens:
            if isinstance(item, tuple):
                text = item[0]
                score = float(item[1])
            else:
                text = item
                score = 0.0
            tokens.append(text)
            token_scores.append(score)

        # Token types: built from the vocab's per-token type when available.
        # Initialize BEFORE the padding block below so padding can append
        # alongside tokens/scores.
        #
        # NOTE: BpeVocab / SentencePieceVocab all_tokens() already yields
        # (bytes, score, TokenType) triples where the vocab computed a
        # correct type — e.g. TokenType.CONTROL for added special tokens
        # like </s> / <|endoftext|>, TokenType.UNKNOWN / CONTROL / BYTE for
        # the SentencePiece classification, etc. The previous code gated
        # this on hasattr(vocab, "get_token_type"), which is False for both
        # BpeVocab and SentencePieceVocab, so those vocabs landed in the
        # else and EVERY token got type 0 — an invalid TokenType (NORMAL=1),
        # and specifically NOT TokenType.CONTROL for EOS. That produced
        # llama.cpp's warning: "control-looking token: 128247 '</s>' was not
        # control-type; this is probably a bug in the model".
        #
        # Correct logic: prefer the per-token TokenType from the all_tokens()
        # tuple when present; fall back to the vocab's get_token_type(i) only
        # for LlamaHfVocab (whose all_tokens() yields plain bytes); default
        # to TokenType.NORMAL otherwise.
        from gguf.constants import TokenType as _TokenType

        token_types: list[int] = []
        for i, item in enumerate(raw_tokens):
            if isinstance(item, tuple) and len(item) >= 3 and item[2] is not None:
                token_types.append(int(item[2]))
            elif hasattr(vocab, "get_token_type"):
                try:
                    token_types.append(int(vocab.get_token_type(i) or _TokenType.NORMAL))
                except Exception:  # noqa: BLE001
                    token_types.append(int(_TokenType.NORMAL))
            else:
                token_types.append(int(_TokenType.NORMAL))

        # ---- Vocab padding for the reserved-slot pattern (Qwen2/2.5/3). ----
        #
        # Real Qwen checkpoints train an embedding matrix padded to a
        # hardware-aligned row count (e.g. 151936) that includes
        # reserved/placeholder rows beyond the tokenizer's *defined*
        # vocabulary (e.g. 151643 = real BPE tokens + special tokens).
        # llama.cpp refuses to load a GGUF where
        # tokenizer.ggml.tokens differs in length from token_embd.weight's
        # row count (check_tensor_dims), so we pad the token list to match
        # the embedding matrix. The embedding matrix is the source of
        # truth — we never trim it (that would shift every embedding row
        # and produce numerically wrong output).
        if len(tokens) != embd_rows:
            if len(tokens) > embd_rows:
                raise GGUFConversionError(
                    f"tokenizer.ggml.tokens has {len(tokens)} entries but "
                    f"token_embd.weight has only {embd_rows} rows. The "
                    "embedding matrix is the ground truth; the tokenizer "
                    "defines MORE tokens than the model can embed. This "
                    "is abnormal — verify the source model files."
                )

            n_pad = embd_rows - len(tokens)

            # First try to recover the source tokenizer's own reserved
            # placeholder tokens (e.g. <|extra_0|> … <|extra_N|> — the Qwen
            # convention). These live in tokenizer.json's model vocab but
            # are dropped by the BpeVocab path above (which only reads
            # vocab.json + added_tokens.json). Reuse them if they exist;
            # otherwise generate placeholder names following the same
            # convention.
            reserved_names: list[str] = []
            tokenizer_json_path = model_dir / "tokenizer.json"
            if tokenizer_json_path.exists():
                try:
                    with open(tokenizer_json_path, encoding="utf-8") as f:
                        tj = json.load(f)
                    tj_vocab = (tj.get("model") or {}).get("vocab") or {}
                    extra_ids: dict[int, str] = {}
                    for tok, tok_id in tj_vocab.items():
                        m = re.match(r"^<\|extra_(\d+)\|>$", tok)
                        if m:
                            extra_ids[int(m.group(1))] = tok
                    # Sort by ID and take the lowest n_pad slots.
                    for idx in sorted(extra_ids):
                        if len(reserved_names) >= n_pad:
                            break
                        reserved_names.append(extra_ids[idx])
                except (OSError, ValueError, TypeError):
                    reserved_names = []  # fall through to generated names.

            # Generate placeholder names if the source had none (or not
            # enough). Use the Qwen convention <|extra_N|> where N starts
            # at the current token count (these are *new* reserved slots,
            # distinct from any real <|extra_N|> the source defined).
            while len(reserved_names) < n_pad:
                reserved_names.append(f"<|extra_{len(tokens) + len(reserved_names)}|>")

            from gguf.constants import TokenType
            pad_token_type = int(TokenType.UNUSED)
            for name in reserved_names:
                tokens.append(name.encode("utf-8"))
                token_scores.append(0.0)
                token_types.append(pad_token_type)

        writer.add_tokenizer_model(tokenizer_model)
        writer.add_token_list(tokens)
        # Scores: use per-token scores when available; default 0.0.
        # NOTE: pass plain Python lists to add_token_scores/add_token_types —
        # GGUFWriter._pack_val requires a collections.abc.Sequence, and
        # numpy ndarrays are NOT registered as Sequences (raises
        # "Invalid GGUF metadata array, expecting sequence").
        if not token_scores:
            token_scores = [0.0] * len(tokens)
        writer.add_token_scores(token_scores)
        # Token types were built above (including padding) — write them.
        writer.add_token_types(token_types)

        # Hard gate: tokenizer.ggml.tokens length MUST equal
        # token_embd.weight's row count or llama.cpp refuses the model.
        # (This is the sibling of the tensor-name hard gate above.)
        if len(tokens) != embd_rows:
            raise GGUFConversionError(
                f"tokenizer.ggml.tokens has {len(tokens)} entries but "
                f"token_embd.weight has {embd_rows} rows. These must match "
                "exactly or llama.cpp will refuse to load the model. "
                f"(architecture={getattr(vocab, 'tokenizer_model', 'unknown')!r})"
            )

        # Add special tokens (BOS, EOS, etc.).
        special_vocab.add_to_gguf(writer)

    def _convert_dir_llamacpp(self, model_dir: Path) -> Path:
        """Fallback: convert using llama-cpp-python's convert_hf_to_gguf CLI."""
        try:
            import convert_hf_to_gguf  # from llama-cpp-python
        except ImportError:
            try:
                from llama_cpp import convert_hf_to_gguf  # type: ignore[import-not-found]
            except ImportError:
                raise GGUFConversionError(
                    "GGUF converter not installed. "
                    "Install: pip install gguf  (pure Python, no compilation)"
                ) from None

        # Architecture compatibility guard: some architectures (e.g. qwen3) are
        # only expressible in GGUF v3. llama.cpp's converter would silently
        # produce a v3 file that the older HomeBred-LLM backend can't read, so
        # fail fast with an actionable message instead.
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    arch_str = json.load(f).get("architectures", ["LlamaForCausalLM"])
                if isinstance(arch_str, list):
                    arch_str = arch_str[0] if arch_str else "LlamaForCausalLM"
                _assert_arch_gguf_compat(
                    arch_str, _ARCH_MAP.get(arch_str, "LLAMA"), self.gguf_version
                )
            except (OSError, ValueError):
                pass  # best-effort — the pure-Python path already validates.

        # llama.cpp's converter can quantize directly via --outtype.
        outtype = self.quantization if self.quantization in ("f16", "f32") else "f16"
        outfile = self.out_dir / f"{self.output_name}-{outtype}.gguf"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        self._log_event(
            "convert_hf_to_gguf_started",
            model_dir=str(model_dir),
            outfile=outfile.name,
            mode="llamacpp",
        )

        args = [
            str(model_dir),
            "--outfile", str(outfile),
            "--outtype", outtype,
        ]
        if self.cache_dir:
            args += ["--model-cache-dir", str(self.cache_dir)]

        argv_json = json.dumps(["convert"] + args)
        script = (
            "import json, sys; "
            "sys.argv = json.loads(sys.argv[1]); "
            "try:\n"
            "    import convert_hf_to_gguf\n"
            "except ImportError:\n"
            "    from llama_cpp import convert_hf_to_gguf\n"
            "convert_hf_to_gguf.main()"
        )

        full_cmd = [sys.executable, "-c", script, argv_json]

        self._log_event("convert_command", command=" ".join(full_cmd))

        try:
            subprocess.run(
                full_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise GGUFConversionError(
                f"convert_hf_to_gguf timed out after {self.timeout_seconds}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise GGUFConversionError(
                f"convert_hf_to_gguf failed (rc={exc.returncode}): "
                f"{exc.stderr[-2000:] if exc.stderr else 'no stderr'}"
            ) from exc

        if not outfile.exists():
            raise GGUFConversionError("convert_hf_to_gguf produced no output file")

        self._log_event(
            "convert_hf_to_gguf_completed",
            outfile=outfile.name,
            size_bytes=outfile.stat().st_size,
            mode="llamacpp",
        )

        return outfile

    # ------------------------------------------------------------------ #
    # Quantization (standalone — only used for llama.cpp f16 fallback)
    # ------------------------------------------------------------------ #
    def _quantize(self, src_gguf: Path) -> Path:
        """Quantize the f16 GGUF to the requested quantization scheme.

        This is only called when the converter produced an f16 GGUF (i.e.
        the llama.cpp fallback path). The pure-Python converter quantizes
        inline during the initial conversion and never reaches here.

        Uses llama.cpp's ``quantize`` CLI binary.
        """
        dest = src_gguf.parent / f"{src_gguf.stem}-{self.quantization}.gguf"

        self._log_event("quantize_started", source=src_gguf.name, quantization=self.quantization)

        return self._quantize_llamacpp(src_gguf, dest)

    def _quantize_llamacpp(self, src_gguf: Path, dest: Path) -> Path:
        """Quantize using llama.cpp's quantize binary."""
        quantize = _find_quantize_bin()
        if quantize is None:
            raise GGUFConversionError(
                "GGUF quantization not available. The pure-Python inline "
                "quantization failed, and the llama.cpp quantize binary was "
                "not found. Install llama.cpp: pip install llama-cpp-python"
            )

        try:
            subprocess.run(
                [quantize, str(src_gguf), str(dest), self.quantization],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise GGUFConversionError(
                f"quantize timed out after {self.timeout_seconds}s"
            ) from exc
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                msg = (
                    f"quantize failed (rc={exc.returncode}): "
                    f"{exc.stderr[-1500:] if exc.stderr else 'no stderr'}"
                )
            else:
                msg = f"quantize binary not found: {quantize}"
            raise GGUFConversionError(msg) from exc

        if not dest.exists():
            raise GGUFConversionError("quantize produced no output file")

        self._log_event(
            "quantize_completed",
            dest=dest.name,
            size_bytes=dest.stat().st_size,
            mode="llamacpp",
        )

        return dest

    # ------------------------------------------------------------------ #
    # Emit + verify
    # ------------------------------------------------------------------ #
    def _emit(self, src_gguf: Path, final: Path) -> None:
        """Copy the final GGUF into the job dir and update the manifest."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Copy to the final destination (atomic-ish: copy to temp first).
        tmp_final = final.with_suffix(final.suffix + ".tmp")
        shutil.copy2(src_gguf, tmp_final)
        tmp_final.replace(final)

        # Write manifest.
        manifest = {
            "format": "gguf",
            "base_model": self.base_model,
            "quantization": self.quantization,
            "file": final.name,
            "size_bytes": final.stat().st_size,
            "gguf_version": self.gguf_version,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "converter": "gguf (pure-Python, inline quantize) / llama.cpp (fallback)",
        }
        (self.out_dir / "gguf-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        self._log_event("emitted", file=final.name, size_bytes=final.stat().st_size)

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def _verify_loadable(self, gguf_path: Path) -> None:
        """Verify the GGUF can actually be loaded by llama.cpp (hard gate)."""
        with open(gguf_path, "rb") as f:
            header = f.read(8)
        if len(header) < 4 or header[:4] != b"GGUF":
            raise GGUFConversionError(f"{gguf_path.name} is not a valid GGUF (bad magic)")
        if len(header) < 8:
            raise GGUFConversionError(f"{gguf_path.name} is truncated (no version)")
        if Llama is not None:
            try:
                _llm = Llama(model_path=str(gguf_path), n_ctx=64, verbose=False)
                del _llm
            except Exception as exc:
                raise GGUFConversionError(
                    f"{gguf_path.name} failed llama-cpp-python load test: {exc}"
                ) from exc

    def _verify(self, gguf_path: Path) -> None:
        """Verify the GGUF is parseable and contains the required metadata.

        In addition to the header/magic check, this parses the GGUF metadata
        KV section and requires at least one ``tokenizer.ggml.*`` key. A GGUF
        with architecture + weights metadata but **zero tokenizer keys** is an
        incomplete export: llama.cpp / HomeBred-LLM cannot tokenize prompts,
        so the file is unloadable at inference time even though the header
        and tensor data look fine.

        It also re-opens the file that actually landed on disk with
        ``gguf.GGUFReader`` and asserts the llama.cpp-style tensor names are
        present (``token_embd.weight``, at least one ``blk.0.*`` tensor, and
        the correct number of ``blk.N.*`` block groups). This mirrors the hard
        gate enforced mid-conversion but catches bugs in the *write* path
        itself (e.g. a tensor dropped during serialization), not just the
        name-mapping step. This is the check that should have caught the
        original missing-``try_suffixes`` bug before it shipped a broken
        file to the user.
        """
        self._log_event("verify_started", file=gguf_path.name)

        # Basic header check.
        with open(gguf_path, "rb") as f:
            header = f.read(8)
        if len(header) < 4 or header[:4] != b"GGUF":
            raise GGUFConversionError(f"{gguf_path.name} is not a valid GGUF (bad magic)")
        if len(header) < 8:
            raise GGUFConversionError(f"{gguf_path.name} is truncated (no version)")

        # Parse the metadata KV section to confirm tokenizer keys exist.
        token_keys = self._metadata_keys(gguf_path)
        if not any(k.startswith("tokenizer.ggml.") or k.startswith("tokenizer.")
                   for k in token_keys):
            raise GGUFConversionError(
                f"{gguf_path.name} is an incomplete GGUF export: it has "
                f"{len(token_keys)} metadata keys for architecture/weights but "
                "zero tokenizer keys. llama.cpp / HomeBred-LLM cannot tokenize "
                "prompts without tokenizer metadata, so the file would be "
                "unloadable at inference time. Ensure the training job saved "
                "its tokenizer (tokenizer.json / tokenizer.model) alongside "
                "the model weights, then re-run the export."
            )

        # Re-open the file that actually landed on disk and verify its tensor
        # names are llama.cpp-style. This catches mapping bugs AND write-path
        # bugs (dropped tensors, wrong block count, etc.).
        self._verify_tensor_structure(gguf_path)

        # Full llama-cpp-python load test (if available).
        if Llama is not None:
            try:
                _llm = Llama(model_path=str(gguf_path), n_ctx=64, verbose=False)
                del _llm  # release the model
            except Exception as exc:  # noqa: BLE001
                raise GGUFConversionError(
                    f"{gguf_path.name} failed llama-cpp-python load test: {exc}"
                ) from exc

        self._log_event("verify_ok", file=gguf_path.name)

    @staticmethod
    def _verify_tensor_structure(gguf_path: Path) -> None:
        """Re-open the on-disk GGUF and verify its llama.cpp-style tensor names.

        Mirrors the mid-conversion hard gate but reads the file that actually
        landed on disk, catching bugs in the write path itself (not just the
        name-mapping step):

        * a ``token_embd.weight`` tensor must exist,
        * at least one ``blk.0.*`` tensor must exist,
        * the number of distinct ``blk.N.*`` block groups must match
          ``general.block_count`` from the metadata.

        The original incident shipped a GGUF whose header, tensor count and
        tokenizer metadata all looked fine but whose tensor names were the raw
        HuggingFace names -- llama.cpp needs exact ``blk.N.attn_q.weight`` etc.
        This check fails loudly on exactly that class of bug.
        """
        try:
            from gguf.gguf_reader import GGUFReader
        except ImportError:  # pragma: no cover - gguf is the primary converter.
            # gguf package isn't available (llama.cpp fallback path wrote the
            # file). Its converter already emits llama.cpp-style names, so we
            # can't do the tensor-name check here -- just skip it.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "gguf.GGUFReader unavailable; skipping tensor-name verification "
                "for %s", gguf_path.name,
            )
            return

        reader = GGUFReader(str(gguf_path))
        try:
            names = [tensor.name for tensor in reader.tensors]
        finally:
            # Release the memmap handle promptly on Windows so the temp-dir
            # cleanup below doesn't hit WinError 32 (file in use).
            del reader
            import gc
            gc.collect()

        if "token_embd.weight" not in names:
            raise GGUFConversionError(
                f"{gguf_path.name} is an unloadable GGUF: no 'token_embd.weight' "
                f"tensor found among {len(names)} tensors. llama.cpp requires "
                "the embedding tensor under its llama.cpp-style name. "
                "This is exactly the failure mode of the original "
                "missing-try_suffixes bug -- the file looked complete but had "
                "raw HuggingFace tensor names."
            )

        blk0 = [n for n in names if n.startswith("blk.0.")]
        if not blk0:
            raise GGUFConversionError(
                f"{gguf_path.name} is an unloadable GGUF: no 'blk.0.*' tensors "
                f"found among {len(names)} tensors. llama.cpp requires at least "
                "one per-block tensor under its llama.cpp-style name "
                "(blk.N.attn_q.weight, blk.N.ffn_up.weight, ...)."
            )

        # Count distinct block groups blk.N.* and compare to general.block_count.
        block_ids = set()
        for n in names:
            if n.startswith("blk."):
                rest = n[len("blk."):]
                dot = rest.index(".") if "." in rest else -1
                if dot > 0:
                    try:
                        block_ids.add(int(rest[:dot]))
                    except ValueError:
                        pass  # not a blk.N.* tensor — ignore.
        if not block_ids:
            raise GGUFConversionError(
                f"{gguf_path.name} is an unloadable GGUF: tensor names contain "
                f"no blk.N.* block tensors (found {len(names)} tensors total)."
            )

        expected_blocks = None
        # Re-fetch block count from a fresh reader (the one above was closed).
        try:
            from gguf.gguf_reader import GGUFReader as _Reader
            r2 = _Reader(str(gguf_path))
            try:
                bcount_field = r2.fields.get("general.block_count")
                if bcount_field is not None:
                    try:
                        expected_blocks = int(bcount_field.contents())
                    except Exception:  # noqa: BLE001
                        expected_blocks = None
            finally:
                del r2
                gc.collect()
        except Exception:  # noqa: BLE001
            expected_blocks = None

        if expected_blocks is not None:
            actual_max = max(block_ids)
            expected_max = expected_blocks - 1
            if actual_max != expected_max:
                raise GGUFConversionError(
                    f"{gguf_path.name} block/tensor count mismatch: "
                    f"general.block_count={expected_blocks} but the tensor data "
                    f"only contains block ids 0..{actual_max} "
                    f"(expected 0..{expected_max}). The write path dropped "
                    "tensors -- this GGUF would be unloadable by llama.cpp."
                )

    @staticmethod
    def _metadata_keys(gguf_path: Path) -> list[str]:
        """Walk the GGUF metadata KV section and return all key names.

        Primary path: the vendored ``gguf.GGUFReader`` loads the KV fields
        via a lazy mmap (``gguf.lazy``), so it never reads tensor data —
        cheap even for multi-GB models. We use it to list ``.fields`` keys,
        which the reader parses robustly for every GGUF version.

        Fallback: if the ``gguf`` package is unavailable, we walk the raw
        header + KV section with seek-only reads, never touching weights.
        """
        # Primary: vendored reader's fields (lazy mmap — KV only, no weights).
        try:
            from gguf.gguf_reader import GGUFReader
            reader = GGUFReader(str(gguf_path))
            return list(reader.fields.keys())
        except Exception as exc:  # noqa: BLE001
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "GGUFReader failed for %s (%s); falling back to raw KV walk",
                gguf_path.name, exc,
            )
        # If the reader raised because the file isn't GGUF at all, the
        # fallback below will also fail with a clear error.

        GGUF_MAGIC = b"GGUF"
        # GGUFValueType enum:
        #  0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32,
        #  6 float32, 7 bool, 8 string, 9 array, 10 uint64, 11 int64,
        #  12 float64
        GGUF_TYPE_SIZES = {
            0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
            10: 8, 11: 8, 12: 8,
        }

        def read_exact(f, n: int) -> bytes:
            data = f.read(n)
            if len(data) != n:
                raise GGUFConversionError(
                    f"{gguf_path.name} GGUF metadata is truncated"
                )
            return data

        def read_u64(f) -> int:
            return struct.unpack("<Q", read_exact(f, 8))[0]

        def read_u32(f) -> int:
            return struct.unpack("<I", read_exact(f, 4))[0]

        def read_str(f) -> str:
            n = read_u64(f)
            if n > 64 * 1024 * 1024:  # sanity: strings shouldn't be huge
                raise GGUFConversionError(
                    f"{gguf_path.name} GGUF string length {n} is implausible"
                )
            return read_exact(f, n).decode("utf-8", errors="replace")

        keys: list[str] = []

        with open(gguf_path, "rb") as f:
            magic = read_exact(f, 4)
            if magic != GGUF_MAGIC:
                raise GGUFConversionError(f"{gguf_path.name} is not a valid GGUF (bad magic)")
            _version = struct.unpack("<I", read_exact(f, 4))[0]

            n_tensors = read_u64(f)
            n_kv = read_u64(f)
            if n_kv > 1_000_000:  # sanity bound
                raise GGUFConversionError(
                    f"{gguf_path.name} GGUF metadata count {n_kv} is implausible"
                )

            for _ in range(n_kv):
                key = read_str(f)
                keys.append(key)

                # Value type is a u32 per the GGUF spec (4 bytes, little
                # endian) — matching what llama.cpp and the gguf writer emit.
                # Reading a single byte desyncs the parser by 3 bytes and
                # mis-reads the next string length (e.g. bogus 83886080).
                ftype = read_u32(f)

                if ftype == 8:  # string
                    read_str(f)
                elif ftype == 9:  # array
                    elem_type = read_u32(f)
                    n_elem = read_u64(f)
                    if elem_type == 8:  # array of strings
                        for _i in range(n_elem):
                            read_str(f)
                    else:
                        elem_size = GGUF_TYPE_SIZES.get(elem_type)
                        if elem_size is None:
                            raise GGUFConversionError(
                                f"{gguf_path.name} GGUF array element type {elem_type} unsupported"
                            )
                        if n_elem * elem_size > 1024 * 1024 * 1024:
                            raise GGUFConversionError(
                                f"{gguf_path.name} GGUF array value too large"
                            )
                        f.seek(n_elem * elem_size, os.SEEK_CUR)
                else:
                    size = GGUF_TYPE_SIZES.get(ftype)
                    if size is None:
                        raise GGUFConversionError(
                            f"{gguf_path.name} GGUF value type {ftype} unsupported"
                        )
                    f.seek(size, os.SEEK_CUR)

            # n_tensors is metadata only — we don't walk tensor info here.
            _ = n_tensors

        return keys


def _safe_rmtree(path: Path, max_retries: int = 5, retry_delay: float = 0.5) -> None:
    """Delete a directory tree, retrying on Windows when files are locked.

    On Windows, file handles (memmap, torch, etc.) may linger briefly even
    after garbage collection, causing PermissionError / WinError 32. This
    function retries with a short delay to let the OS release the handles.
    If all retries fail, the error is logged but not raised — the temp dir
    is expendable and we don't want to fail a successful conversion over
    cleanup.
    """
    import gc

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except (PermissionError, OSError) as exc:
            if attempt < max_retries - 1:
                gc.collect()
                time.sleep(retry_delay * (attempt + 1))
            else:
                # Last resort: best-effort with ignore_errors.
                shutil.rmtree(path, ignore_errors=True)
                # Log to stderr (not stdout — this is not a progress event).
                print(
                    f"WARNING: could not fully clean temp dir {path}: {exc}",
                    file=sys.stderr,
                )


# Module-level registry for lock fds (per-process; harmless across processes).
_gc_lock_fds: dict[Path, int] = {}


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for on-demand GGUF conversion.

    The Go exporter invokes this when a user clicks "Download GGUF model":

        python -m trainer.gguf \\
            --job-dir  <job working dir> \\
            --base-model <huggingface repo id> \\
            --adapter <path to LoRA adapter dir>   (or --model-dir for merged)
            --quantization q4_k_m \\
            --output-name my-task

    The converted file is written to ``job-dir/gguf/<name>.gguf`` and a
    ``gguf-manifest.json`` is placed beside it. Exits 0 on success, 1 on
    any conversion error (an error message is printed to stderr).

    Events are emitted as JSON lines on stdout (``conversion_started``,
    ``cache_hit``, ``merge_started``, ``convert_hf_to_gguf_started``, etc.)
    for the Go side to tail and report progress.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Convert a Distillery-trained model to GGUF (HomeBred-LLM / llama.cpp).")
    parser.add_argument("--job-dir", required=True, help="Job working directory (contains adapter/model + optional tokenizer).")
    parser.add_argument("--base-model", required=True, help="HuggingFace repo id of the base model.")
    parser.add_argument("--adapter", help="Path to a LoRA adapter directory (merged into base model).")
    parser.add_argument("--model-dir", help="Path to a full merged HF model directory (skips merging).")
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Convert the base model directly (no adapter/merged model). "
        "Fallback when no trained weights exist on disk.",
    )
    parser.add_argument("--quantization", default="q4_k_m", choices=sorted(VALID_QUANTIZATIONS), help="GGUF quantization scheme.")
    parser.add_argument("--output-name", default="task-model", help="Base name for the output .gguf file.")
    parser.add_argument("--cache-dir", help="Optional HuggingFace model cache directory.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the requested GGUF already exists.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=f"Conversion timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS}).")
    parser.add_argument(
        "--gguf-version",
        type=int,
        default=DEFAULT_GGUF_VERSION,
        choices=[2, 3],
        help=f"GGUF format version to emit (default {DEFAULT_GGUF_VERSION}). "
        "GGUF v3 (the default) requires a llama.cpp backend built from b2588 "
        "or newer (e.g. LLamaSharp >= 0.28.6). If the target HomeBred-LLM "
        "build only supports GGUF v2 (e.g. LLamaSharp 0.27.0), pass "
        "--gguf-version 2. Architectures like qwen3 require v3.",
    )
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir)
    if not job_dir.exists():
        print(f"FATAL: job dir {job_dir} does not exist", file=sys.stderr)
        return 1

    converter = GGUFConverter(
        base_model=args.base_model,
        job_dir=job_dir,
        output_name=args.output_name,
        quantization=args.quantization,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force=args.force,
        timeout_seconds=args.timeout,
        gguf_version=args.gguf_version,
    )

    try:
        if args.adapter:
            adapter_dir = Path(args.adapter)
            if not adapter_dir.exists():
                print(f"FATAL: adapter dir {adapter_dir} does not exist", file=sys.stderr)
                return 1
            gguf_path = converter.convert_adapter(adapter_dir)
        elif args.model_dir:
            model_dir = Path(args.model_dir)
            if not model_dir.exists():
                print(f"FATAL: model dir {model_dir} does not exist", file=sys.stderr)
                return 1
            gguf_path = converter.convert_merged(model_dir)
        elif args.base_model_only:
            gguf_path = converter.convert_base_model()
        else:
            print("FATAL: one of --adapter, --model-dir, or --base-model-only is required", file=sys.stderr)
            return 1
    except GGUFConversionError as exc:
        GGUFConverter._log_event("error", message=str(exc))
        print(f"GGUF conversion failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        GGUFConverter._log_event("error", message=f"unexpected error: {exc}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    GGUFConverter._log_event("done", file=str(gguf_path))
    print(f"GGUF written to {gguf_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())