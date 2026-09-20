"""Unit tests for the trainer's per-language syntax validators."""
from __future__ import annotations

import pytest

from trainer.eval import syntax


def test_validate_python_accepts_valid_snippet() -> None:
    assert syntax.validate_python("def foo(x):\n    return x + 1\n") is True


def test_validate_python_rejects_syntax_error() -> None:
    assert syntax.validate_python("def foo(:\n    pass\n") is False


def test_validate_java_balanced_braces() -> None:
    assert syntax.validate_java("public void f() { return; }") is True
    assert syntax.validate_java("public void f() { return; ") is False


def test_validate_csharp_balanced_parens() -> None:
    assert syntax.validate_csharp("void F(int x) { }") is True
    assert syntax.validate_csharp("void F(int x { }") is False


def test_validate_unknown_language_is_permissive() -> None:
    assert syntax.validate_snippet("anything", "klingon") is True


@pytest.mark.skipif(syntax.shutil.which("gofmt") is None, reason="gofmt not installed")
def test_validate_go_with_gofmt() -> None:
    assert syntax.validate_go("package main\nfunc main() {}\n") is True