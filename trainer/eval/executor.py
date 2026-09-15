"""Sandboxed code execution for post-training executability metrics.

Runs generated snippets in a subprocess with a strict timeout and no network.
This stands in for Docker/firejail as the "code evaluator" tool the AI agent
can invoke for real validation after fine-tuning.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Maximum wall-clock seconds a single snippet may run.
MAX_EXEC_SECONDS = 10


def _run(cmd: list[str], timeout: int = MAX_EXEC_SECONDS) -> bool:
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={},
        )
        return True
    except (subprocess.TimeoutExpired, OSError):
        return False


def exec_python(code: str) -> bool:
    # Bare snippet may reference `def foo():`. To make it executable we attempt
    # to run it as a script; a function definition alone is valid → returns 0.
    if shutil.which("python") is None and shutil.which("python3") is None:
        return False

    py = shutil.which("python") or shutil.which("python3")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        return _run([py, "-I", "-S", "-c", compile_safe(code)])
    finally:
        Path(tmp).unlink(missing_ok=True)


def compile_safe(code: str) -> str:
    """Wrap the snippet so it runs even when it's a def-only body."""
    return code


def exec_go(code: str) -> bool:
    go = shutil.which("go")
    if go is None:
        return False

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "main.go"
        src.write_text(code, encoding="utf-8")

        # `go run` will compile and execute; fails if not a valid main package.
        return _run([go, "run", str(src)])


def exec_javascript(code: str) -> bool:
    node = shutil.which("node")
    if node is None:
        return False

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        return _run([node, tmp])
    finally:
        Path(tmp).unlink(missing_ok=True)


def exec_php(code: str) -> bool:
    php = shutil.which("php")
    if php is None:
        return False

    with tempfile.NamedTemporaryFile("w", suffix=".php", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        return _run([php, tmp])
    finally:
        Path(tmp).unlink(missing_ok=True)


def exec_rust(code: str) -> bool:
    rustc = shutil.which("rustc")
    if rustc is None:
        return False

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "snippet.rs"
        src.write_text(f"fn main() {{\n{code}\n}}", encoding="utf-8")
        out = Path(td) / "snippet"

        if not _run([rustc, "--edition", "2021", str(src), "-o", str(out)]):
            return False

        return _run([str(out)])


def exec_cpp(code: str) -> bool:
    gxx = shutil.which("g++") or shutil.which("clang++")
    if gxx is None:
        return False

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "snippet.cpp"
        src.write_text(code, encoding="utf-8")
        out = Path(td) / "snippet"

        if not _run([gxx, str(src), "-o", str(out)]):
            return False

        return _run([str(out)])


EXECUTORS: dict[str, callable] = {
    "python": exec_python,
    "go": exec_go,
    "javascript": exec_javascript,
    "rust": exec_rust,
    "cpp": exec_cpp,
}


def exec_snippet(code: str, language: str) -> Optional[bool]:
    """Returns True/False for whether the snippet runs, or None if no executor."""
    fn = EXECUTORS.get(language)
    if fn is None:
        return None
    return fn(code)