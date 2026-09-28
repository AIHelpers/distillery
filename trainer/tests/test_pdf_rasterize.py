"""Unit tests for the PDF -> per-page PNG rasterizer (trainer/pdf_rasterize.py).

This module is new with the vision-language / document AI feature (plan 06)
and had no test coverage. Unlike the other trainer/tests modules, exercising
it for real needs PyMuPDF (a light, pure-C-extension dependency -- no
torch/transformers), so these build tiny in-memory PDFs with PyMuPDF itself
and drive the script's `main()` end to end via a patched argv, checking the
files and manifest it writes rather than re-implementing its logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

fitz = pytest.importorskip("pymupdf", reason="PyMuPDF not installed; see trainer/requirements.txt")

from trainer import pdf_rasterize


def _make_pdf(path: Path, *, pages: int = 1, width: float = 200, height: float = 300) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=width, height=height)
        page.insert_text((20, 20), f"page {i + 1}")
    doc.save(str(path))
    doc.close()


def _run_main(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["pdf_rasterize.py", *args])
    return pdf_rasterize.main()


def test_rasterizes_each_page_and_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=3)
    out_dir = tmp_path / "out"

    rc = _run_main(monkeypatch, ["--input", str(pdf_path), "--out-dir", str(out_dir), "--dpi", "100"])

    assert rc == 0
    assert (out_dir / "page_0001.png").exists()
    assert (out_dir / "page_0002.png").exists()
    assert (out_dir / "page_0003.png").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["dpi"] == 100
    assert manifest["source_pages"] == 3
    assert [p["page"] for p in manifest["pages"]] == [1, 2, 3]
    assert all(p["file"] == f"page_{p['page']:04d}.png" for p in manifest["pages"])
    assert all(p["width"] > 0 and p["height"] > 0 for p in manifest["pages"])


def test_dpi_is_clamped_to_valid_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=1)
    out_dir = tmp_path / "out"

    rc = _run_main(monkeypatch, ["--input", str(pdf_path), "--out-dir", str(out_dir), "--dpi", "99999"])

    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["dpi"] == pdf_rasterize.MAX_DPI


def test_max_pages_caps_how_many_pages_are_rasterized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=5)
    out_dir = tmp_path / "out"

    rc = _run_main(monkeypatch, ["--input", str(pdf_path), "--out-dir", str(out_dir), "--max-pages", "2"])

    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["pages"]) == 2
    assert manifest["source_pages"] == 5  # true page count is still reported.
    assert not (out_dir / "page_0003.png").exists()


def test_large_page_is_downscaled_below_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A physically huge page at a high DPI would exceed MAX_PAGE_SIDE_PX; the
    # rasterizer must re-render at a lower zoom rather than emit an oversized
    # bitmap.
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=1, width=3000, height=3000)
    out_dir = tmp_path / "out"

    rc = _run_main(monkeypatch, ["--input", str(pdf_path), "--out-dir", str(out_dir), "--dpi", "400"])

    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    page = manifest["pages"][0]
    assert page["width"] <= pdf_rasterize.MAX_PAGE_SIDE_PX
    assert page["height"] <= pdf_rasterize.MAX_PAGE_SIDE_PX


def test_corrupt_pdf_returns_error_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_path = tmp_path / "not_a_pdf.pdf"
    bad_path.write_bytes(b"this is not a pdf file")
    out_dir = tmp_path / "out"

    rc = _run_main(monkeypatch, ["--input", str(bad_path), "--out-dir", str(out_dir)])

    assert rc == 1


def test_missing_pymupdf_reports_fatal_and_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=1)
    out_dir = tmp_path / "out"

    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)

    rc = _run_main(monkeypatch, ["--input", str(pdf_path), "--out-dir", str(out_dir)])

    assert rc == 2
    assert "PyMuPDF is not installed" in capsys.readouterr().err
