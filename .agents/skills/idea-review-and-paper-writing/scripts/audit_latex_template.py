#!/usr/bin/env python3
"""Audit an initialized project for changes to official template formatting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from create_latex_project import LOCK_NAME, protected_directives, sha256_file


def digest_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def audit(project: Path) -> list[str]:
    errors: list[str] = []
    project = project.resolve()
    lock_path = project / LOCK_NAME
    if not lock_path.is_file():
        return [f"missing {LOCK_NAME}: {lock_path}"]
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {LOCK_NAME}: {exc}"]

    for relative, expected in lock.get("protected_file_sha256", {}).items():
        path = project / relative
        if not path.is_file():
            errors.append(f"protected template file is missing: {relative}")
        elif sha256_file(path).lower() != expected.lower():
            errors.append(f"protected template file was modified: {relative}")

    main_path = project / lock.get("main_file", "")
    if not main_path.is_file():
        errors.append(f"main manuscript file is missing: {lock.get('main_file')}")
    else:
        try:
            current = protected_directives(main_path, lock["main_encoding"])
        except (LookupError, UnicodeDecodeError) as exc:
            errors.append(
                f"main file no longer matches its registered encoding "
                f"{lock.get('main_encoding')}: {exc}"
            )
        else:
            if digest_lines(current) != lock.get("protected_directives_sha256"):
                errors.append(
                    "main file template directives changed; restore the official "
                    "document class, inputs, packages, bibliography style, and layout commands"
                )
            if current != lock.get("protected_directives"):
                errors.append(
                    "main file protected directive order or text no longer matches the template"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    errors = audit(args.project)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Template audit failed with {len(errors)} error(s).")
        return 1
    print("Official template integrity audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
