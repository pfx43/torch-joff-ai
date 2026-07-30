#!/usr/bin/env python3
"""Audit LaTeX labels, references, citations, figures, and common draft residue."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:eqref|ref|autoref|cref|Cref)\{([^}]+)\}")
CITE_RE = re.compile(r"\\(?:cite|citep|citet)\s*(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
FIGURE_EXTENSIONS = (".pdf", ".svg", ".eps", ".png", ".jpg", ".jpeg")


def find_figure(tex_path: Path, target: str) -> bool:
    candidate = (tex_path.parent / target).resolve()
    if candidate.suffix:
        return candidate.exists()
    return any(candidate.with_suffix(ext).exists() for ext in FIGURE_EXTENSIONS)


def audit(root: Path) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    tex_files = sorted(root.rglob("*.tex"))
    if not tex_files:
        return {"errors": ["no .tex files found"], "warnings": []}
    labels: list[str] = []
    refs: list[tuple[Path, str]] = []
    cites: list[tuple[Path, str]] = []
    for path in tex_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        labels.extend(LABEL_RE.findall(text))
        refs.extend((path, value) for value in REF_RE.findall(text))
        for group in CITE_RE.findall(text):
            cites.extend((path, key.strip()) for key in group.split(",") if key.strip())
        for target in GRAPHICS_RE.findall(text):
            if not find_figure(path, target):
                errors.append(
                    f"{path.relative_to(root).as_posix()}: missing figure {target}"
                )
        for marker in ("@@TOLARIA", "\ufffd"):
            if marker in text:
                errors.append(
                    f"{path.relative_to(root).as_posix()}: contains invalid marker {marker}"
                )
        for marker in ("TODO", "TBD", "FIXME"):
            if re.search(rf"\b{marker}\b", text):
                warnings.append(
                    f"{path.relative_to(root).as_posix()}: contains {marker}"
                )
    for label, count in Counter(labels).items():
        if count > 1:
            errors.append(f"duplicate LaTeX label {label!r} ({count} occurrences)")
    label_set = set(labels)
    for path, ref in refs:
        if ref not in label_set:
            errors.append(
                f"{path.relative_to(root).as_posix()}: undefined reference {ref}"
            )
    bib_keys: set[str] = set()
    for path in root.rglob("*.bib"):
        bib_keys.update(BIB_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    if cites and not bib_keys:
        warnings.append("citations exist but no BibTeX entries were found")
    elif bib_keys:
        for path, key in cites:
            if key not in bib_keys:
                errors.append(
                    f"{path.relative_to(root).as_posix()}: missing BibTeX key {key}"
                )
    for log_path in root.rglob("*.log"):
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in (
            (r"LaTeX Warning: There were undefined references", "undefined references"),
            (r"Citation .* undefined", "undefined citations"),
            (r"Overfull \\[hv]box", "overfull box"),
        ):
            if re.search(pattern, log_text):
                warnings.append(
                    f"{log_path.relative_to(root).as_posix()}: {label} reported"
                )
    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(Path(args.project_root).resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(
            f"Audit complete: {len(report['errors'])} error(s), "
            f"{len(report['warnings'])} warning(s)."
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
