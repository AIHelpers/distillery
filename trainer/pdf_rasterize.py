#!/usr/bin/env python3
"""PDF -> per-page PNG rasterizer, used by the vision-language dataset
importer (plan 06, section 2: "PDFs rasterized page by page (pdf -> PNG at
bounded DPI)").

Kept as a standalone script (rather than folded into trainer/run.py) because
it has a much lighter dependency (PyMuPDF only -- no torch/transformers) and
is invoked synchronously from the Go HTTP handler at import time, not as a
background training job.

    python -m trainer.pdf_rasterize \
        --input  /path/to/document.pdf \
        --out-dir /path/to/output_dir \
        --dpi 150

Writes page_0001.png, page_0002.png, ... to --out-dir plus a manifest.json:

    {"pages": [{"page": 1, "file": "page_0001.png", "width": 1275, "height": 1650}, ...]}

Exits 0 on success, non-zero with a message on stderr otherwise (missing
PyMuPDF, corrupt PDF, etc). The bounded DPI keeps a huge PDF from producing
unusably large page images (and matching the plan's "PNG at bounded DPI").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIN_DPI = 50
MAX_DPI = 400
DEFAULT_DPI = 150

# A page rendered above this pixel count (either dimension) is downscaled so
# a huge-format PDF page can't produce an unusably large image regardless of
# the requested DPI.
MAX_PAGE_SIDE_PX = 4000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source PDF.")
    parser.add_argument("--out-dir", required=True, help="Directory to write page_NNNN.png + manifest.json into.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Rasterization DPI ({MIN_DPI}-{MAX_DPI}, default {DEFAULT_DPI}).")
    parser.add_argument("--max-pages", type=int, default=500, help="Safety cap on pages rasterized from one PDF.")
    args = parser.parse_args()

    dpi = max(MIN_DPI, min(MAX_DPI, args.dpi))

    try:
        import pymupdf as fitz  # The `pymupdf` import name is preferred; `fitz` is deprecated upstream.
    except ImportError:
        try:
            import fitz  # type: ignore  # Older pymupdf releases only exposed this name.
        except ImportError:
            print(
                "FATAL: PyMuPDF is not installed. Install it with `pip install pymupdf` "
                "(see trainer/requirements.txt) to enable PDF import.",
                file=sys.stderr,
            )
            return 2

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(src))
    except Exception as exc:  # noqa: BLE001 - report any PyMuPDF open failure.
        print(f"FATAL: could not open PDF: {exc}", file=sys.stderr)
        return 1

    if doc.page_count == 0:
        print("FATAL: PDF has no pages", file=sys.stderr)
        return 1

    zoom = dpi / 72.0  # PyMuPDF's default render resolution is 72 DPI.
    matrix = fitz.Matrix(zoom, zoom)

    pages_meta = []
    n_pages = min(doc.page_count, args.max_pages)

    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Downscale (re-render at a lower zoom) if either side exceeds the cap,
        # rather than resizing the rasterized bitmap, to keep text crisp.
        if pix.width > MAX_PAGE_SIDE_PX or pix.height > MAX_PAGE_SIDE_PX:
            shrink = MAX_PAGE_SIDE_PX / max(pix.width, pix.height)
            capped_matrix = fitz.Matrix(zoom * shrink, zoom * shrink)
            pix = page.get_pixmap(matrix=capped_matrix, alpha=False)

        filename = f"page_{i + 1:04d}.png"
        pix.save(str(out_dir / filename))

        pages_meta.append({"page": i + 1, "file": filename, "width": pix.width, "height": pix.height})

    source_pages = doc.page_count
    doc.close()

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pages": pages_meta, "dpi": dpi, "source_pages": source_pages}, f)

    print(f"rasterized {len(pages_meta)} page(s) at {dpi} DPI -> {out_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
