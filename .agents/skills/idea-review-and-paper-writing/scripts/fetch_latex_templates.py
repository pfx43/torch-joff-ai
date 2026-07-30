#!/usr/bin/env python3
"""Download registered official LaTeX template archives with hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "latex-templates" / "sources.json"
CACHE_ROOT = ROOT / "assets" / "latex-templates" / "cache"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(path: Path, record: dict) -> None:
    actual_size = path.stat().st_size
    if actual_size != record["size_bytes"]:
        raise ValueError(
            f"{path}: expected {record['size_bytes']} bytes, got {actual_size}"
        )
    actual_hash = digest(path)
    if actual_hash.lower() != record["sha256"].lower():
        raise ValueError(
            f"{path}: SHA-256 mismatch; expected {record['sha256']}, got {actual_hash}"
        )


def fetch(template_id: str, record: dict, force: bool = False) -> Path:
    target_dir = CACHE_ROOT / template_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / record["archive_filename"]
    if target.exists() and not force:
        verify(target, record)
        print(f"Verified cached {template_id}: {target}")
        return target

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        record["download_url"],
        headers={"User-Agent": "Codex official-template fetcher/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        verify(partial, record)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Downloaded and verified {template_id}: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        action="append",
        help="Template id to fetch; repeat for more than one. Defaults to all.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    templates = manifest["templates"]
    selected = args.template or list(templates)
    unknown = sorted(set(selected).difference(templates))
    if unknown:
        parser.error("unknown template id(s): " + ", ".join(unknown))

    try:
        for template_id in selected:
            fetch(template_id, templates[template_id], force=args.force)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
