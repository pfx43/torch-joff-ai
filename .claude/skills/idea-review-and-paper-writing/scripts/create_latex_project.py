#!/usr/bin/env python3
"""Initialize a manuscript project from a verified official template archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

from fetch_latex_templates import CACHE_ROOT, fetch, load_manifest


LOCK_NAME = "TEMPLATE_LOCK.json"
BASELINE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PROTECTED_DIRECTIVE_RE = re.compile(
    r"^\s*\\(?:documentclass|usepackage|RequirePackage|input|includeonly|"
    r"bibliographystyle|pagestyle|thispagestyle|setlength|addtolength|"
    r"renewcommand|newcommand|def|titleformat|titlespacing|linespread|geometry)\b"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def protected_directives(path: Path, encoding: str) -> list[str]:
    text = path.read_bytes().decode(encoding)
    return [
        line.rstrip()
        for line in text.splitlines()
        if PROTECTED_DIRECTIVE_RE.match(line)
    ]


def safe_member_name(name: str) -> PurePosixPath:
    candidate = PurePosixPath(name.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and ":" in candidate.parts[0])
    ):
        raise ValueError(f"unsafe ZIP member path: {name}")
    return candidate


def initialize(
    template_id: str,
    record: dict,
    destination: Path,
    archive: Path,
    baseline_id: str,
    revision: int,
) -> None:
    destination = destination.resolve()
    revision_folder = f"{baseline_id}-r{revision}"
    if destination.name != revision_folder:
        raise ValueError(
            f"destination directory must be named {revision_folder}"
        )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    root = PurePosixPath(record["archive_root"])
    wanted = set(record["project_files"])
    extracted: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in bundle.infolist():
            safe = safe_member_name(info.filename)
            try:
                relative = safe.relative_to(root)
            except ValueError:
                continue
            relative_name = relative.as_posix()
            if relative_name in wanted:
                members[relative_name] = info

        missing = sorted(wanted.difference(members))
        if missing:
            raise ValueError("archive lacks registered file(s): " + ", ".join(missing))

        for relative_name in record["project_files"]:
            target = destination.joinpath(*PurePosixPath(relative_name).parts)
            resolved = target.resolve()
            if destination != resolved and destination not in resolved.parents:
                raise ValueError(f"unsafe extraction target: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(members[relative_name]))
            extracted.add(relative_name)

    original_main_file = record["main_file"]
    main_file = "manuscript.tex"
    source_main = destination / original_main_file
    target_main = destination / main_file
    if source_main.resolve() != target_main.resolve():
        source_main.rename(target_main)
        extracted.discard(original_main_file)
        extracted.add(main_file)
    snapshot_comment = (
        f"% Frozen snapshot: <{baseline_id}, {revision}>\n"
    ).encode(record["encoding"])
    source_bytes = target_main.read_bytes()
    if not source_bytes.startswith(snapshot_comment):
        target_main.write_bytes(snapshot_comment + source_bytes)
    protected_hashes = {
        name: sha256_file(destination / name)
        for name in sorted(extracted.difference({main_file}))
    }
    directives = protected_directives(destination / main_file, record["encoding"])
    lock = {
        "schema_version": 3,
        "template_id": template_id,
        "display_name": record["display_name"],
        "source_url": record["download_url"],
        "archive_sha256": record["sha256"],
        "baseline_id": baseline_id,
        "context_revision": revision,
        "revision_folder": revision_folder,
        "original_main_file": original_main_file,
        "main_file": main_file,
        "main_encoding": record["encoding"],
        "protected_file_sha256": protected_hashes,
        "protected_directives": directives,
        "protected_directives_sha256": sha256_bytes(
            "\n".join(directives).encode("utf-8")
        ),
        "editing_boundary": (
            "Edit article content only. Do not change template formatting, "
            "class files, header files, package configuration, or protected directives."
        ),
    }
    (destination / LOCK_NAME).write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Initialized {record['display_name']} at {destination}")
    print(f"Edit manuscript content in: {destination / main_file}")
    print(f"Keep template files and formatting directives unchanged; audit with {LOCK_NAME}.")
    print(f"TOOLCHAIN: {record['toolchain_note']}")
    for dependency in record.get("known_dependencies", []):
        if not dependency.get("bundled", False):
            print(
                f"NOTICE: {dependency['name']} is not bundled. "
                f"{dependency.get('note', 'Obtain it lawfully for the required toolchain.')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--revision", required=True, type=int)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require an already cached archive instead of downloading it.",
    )
    args = parser.parse_args()

    if not BASELINE_RE.fullmatch(args.baseline_id):
        parser.error(
            "--baseline-id must be a lowercase slug matching "
            "[a-z0-9]+(?:-[a-z0-9]+)*"
        )
    if len(args.baseline_id) > 24:
        parser.error("--baseline-id must contain at most 24 characters")
    if args.revision < 1:
        parser.error("--revision must be a positive integer")

    manifest = load_manifest()
    templates = manifest["templates"]
    if args.template not in templates:
        parser.error(
            f"unknown template id {args.template!r}; choose from "
            + ", ".join(templates)
        )
    record = templates[args.template]
    archive = CACHE_ROOT / args.template / record["archive_filename"]
    try:
        if not archive.exists():
            if args.offline:
                raise ValueError(f"verified archive is not cached: {archive}")
            archive = fetch(args.template, record)
        else:
            from fetch_latex_templates import verify

            verify(archive, record)
        initialize(
            args.template,
            record,
            args.destination,
            archive,
            args.baseline_id,
            args.revision,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
