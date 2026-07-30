#!/usr/bin/env python3
"""Compile a LaTeX manuscript and optionally render PDF pages for review."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> int:
    print(" ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("main_tex")
    parser.add_argument("--render-dir")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    tex_path = Path(args.main_tex).resolve()
    if not tex_path.exists():
        print(f"ERROR: missing TeX file {tex_path}")
        return 1
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("ERROR: latexmk is not available")
        return 1
    code = run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-halt-on-error",
            tex_path.name,
        ],
        tex_path.parent,
    )
    if code:
        return code
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        print(f"ERROR: expected PDF was not created: {pdf_path}")
        return 1
    if args.no_render:
        print(pdf_path)
        return 0
    # Prefer a native executable on Windows. A command-wrapper earlier on PATH
    # may point to an unavailable bundled runtime even when TeX Live provides
    # a working pdftoppm.exe.
    pdftoppm = shutil.which("pdftoppm.exe") or shutil.which("pdftoppm")
    if not pdftoppm:
        print("WARNING: pdftoppm is unavailable; PDF rendering skipped")
        print(pdf_path)
        return 0
    render_dir = (
        Path(args.render_dir).resolve()
        if args.render_dir
        else tex_path.parent / f"{tex_path.stem}-rendered"
    )
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / "page"
    code = run([pdftoppm, "-png", "-r", "150", str(pdf_path), str(prefix)], tex_path.parent)
    if not code:
        print(f"Rendered pages: {render_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
