#!/usr/bin/env python3
"""Audit one manuscript revision folder and its fixed role filenames."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BASELINE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
FOLDER_RE = re.compile(r"(?P<id>[a-z0-9]+(?:-[a-z0-9]+)*)-r(?P<revision>\d+)")
SNAPSHOT_COMMENT_RE = re.compile(
    r"^%\s*Frozen snapshot:\s*<(?P<id>[^,>]+),\s*(?P<revision>\d+)>\s*$",
    re.MULTILINE,
)
VISUAL_EXTENSIONS = {
    ".ai", ".drawio", ".eps", ".pdf", ".png", ".pptx", ".svg", ".tex"
}


def metadata_value(text: str, label: str) -> str:
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(?P<value>.*?)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group("value").strip() if match else ""


def unquote_code(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1].strip()
    return value


def check_bound_metadata(
    path: Path,
    baseline_id: str,
    revision: str,
    folder_name: str,
    errors: list[str],
) -> str:
    if not path.is_file():
        errors.append(f"required artifact is missing: {path.name}")
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if metadata_value(text, "Baseline ID") != baseline_id:
        errors.append(f"{path.name}: Baseline ID does not match {baseline_id}")
    if metadata_value(text, "Context revision") != revision:
        errors.append(f"{path.name}: Context revision does not match {revision}")
    if metadata_value(text, "Revision folder") != folder_name:
        errors.append(f"{path.name}: Revision folder must be {folder_name}")
    return text


def audit(root: Path, require_pdf: bool, require_stage4: bool) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    folder_match = FOLDER_RE.fullmatch(root.name)
    if not folder_match:
        errors.append(
            "project directory must be named <short-baseline-id>-r<revision>"
        )
        return {"errors": errors, "warnings": warnings}
    folder_id = folder_match.group("id")
    folder_revision = folder_match.group("revision")
    if len(folder_id) > 24:
        errors.append("Baseline ID in the revision-folder name exceeds 24 characters")

    context = root / "MANUSCRIPT_CONTEXT.md"
    if not context.is_file():
        errors.append("required artifact is missing: MANUSCRIPT_CONTEXT.md")
        return {"errors": errors, "warnings": warnings}
    text = context.read_text(encoding="utf-8", errors="replace")
    baseline_id = metadata_value(text, "Baseline ID")
    revision = metadata_value(text, "Context revision")
    status = metadata_value(text, "Context status").upper()
    if not BASELINE_RE.fullmatch(baseline_id) or len(baseline_id) > 24:
        errors.append(
            "MANUSCRIPT_CONTEXT.md: Baseline ID must be a readable lowercase "
            "slug of at most 24 characters"
        )
    if not revision.isdigit() or int(revision) < 1:
        errors.append("MANUSCRIPT_CONTEXT.md: Context revision must be positive")
    if status != "FROZEN":
        errors.append("MANUSCRIPT_CONTEXT.md: Context status must be FROZEN")
    if baseline_id != folder_id or revision != folder_revision:
        errors.append(
            "MANUSCRIPT_CONTEXT.md: Baseline ID/revision do not match the "
            "revision-folder name"
        )
    if metadata_value(text, "Revision folder") != root.name:
        errors.append(
            f"MANUSCRIPT_CONTEXT.md: Revision folder must be {root.name}"
        )
    main_source = unquote_code(metadata_value(text, "Main manuscript source"))
    if Path(main_source).name != "manuscript.tex":
        errors.append(
            "MANUSCRIPT_CONTEXT.md: Main manuscript source must be manuscript.tex"
        )

    tex_path = root / "manuscript.tex"
    if not tex_path.is_file():
        errors.append("required artifact is missing: manuscript.tex")
    else:
        tex = tex_path.read_text(encoding="utf-8", errors="replace")
        snapshot = SNAPSHOT_COMMENT_RE.search(tex)
        if not snapshot:
            errors.append("manuscript.tex: missing frozen-snapshot comment")
        elif (
            snapshot.group("id").strip() != baseline_id
            or snapshot.group("revision") != revision
        ):
            errors.append("manuscript.tex: frozen-snapshot comment does not match")

    writing_log = root / "WRITING_LOOP_LOG.md"
    log_text = check_bound_metadata(
        writing_log, baseline_id, revision, root.name, errors
    )
    if log_text:
        log_main = unquote_code(metadata_value(log_text, "Main source"))
        log_context = unquote_code(metadata_value(log_text, "Context file"))
        if Path(log_main).name != "manuscript.tex":
            errors.append("WRITING_LOOP_LOG.md: Main source must be manuscript.tex")
        if Path(log_context).name != "MANUSCRIPT_CONTEXT.md":
            errors.append(
                "WRITING_LOOP_LOG.md: Context file must be MANUSCRIPT_CONTEXT.md"
            )

    bib_files = sorted(root.glob("*.bib"))
    if bib_files and (root / "references.bib") not in bib_files:
        errors.append("the canonical bibliography filename is references.bib")

    if require_pdf and not (root / "manuscript.pdf").is_file():
        errors.append("required compiled PDF is missing: manuscript.pdf")

    if require_stage4:
        stage4_log = root / "STAGE4_FIGURE_EXPERIMENT_LOG.md"
        stage4_text = check_bound_metadata(
            stage4_log, baseline_id, revision, root.name, errors
        )
        if stage4_text:
            stage4_context = unquote_code(
                metadata_value(stage4_text, "Context file")
            )
            stage4_main_stem = unquote_code(
                metadata_value(stage4_text, "Main TeX/PDF stem")
            )
            if Path(stage4_context).name != "MANUSCRIPT_CONTEXT.md":
                errors.append(
                    "STAGE4_FIGURE_EXPERIMENT_LOG.md: Context file must be "
                    "MANUSCRIPT_CONTEXT.md"
                )
            if stage4_main_stem != "manuscript":
                errors.append(
                    "STAGE4_FIGURE_EXPERIMENT_LOG.md: Main TeX/PDF stem must "
                    "be manuscript"
                )

    patterns = {
        "figures": re.compile(
            r"fig-\d{2}-(?:principle|model|workflow)-[a-z0-9]+"
            r"(?:-[a-z0-9]+)*"
        ),
        "plots": re.compile(r"plot-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*"),
    }
    for folder_name, pattern in patterns.items():
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for artifact in folder.rglob("*"):
            if not artifact.is_file() or artifact.suffix.lower() not in VISUAL_EXTENSIONS:
                continue
            if not pattern.fullmatch(artifact.stem):
                errors.append(
                    f"{artifact.relative_to(root)}: invalid {folder_name} filename"
                )

    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("revision_folder", nargs="?", default=".")
    parser.add_argument("--require-pdf", action="store_true")
    parser.add_argument("--require-stage4", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(
        Path(args.revision_folder).resolve(),
        args.require_pdf,
        args.require_stage4,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(
            f"Artifact-name audit complete: {len(report['errors'])} error(s), "
            f"{len(report['warnings'])} warning(s)."
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
