#!/usr/bin/env python3
"""Audit editable scientific-figure sources and raster exports."""

from __future__ import annotations

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


EDITABLE_SOURCE_EXTENSIONS = {".drawio", ".svg", ".pptx", ".tex"}
VECTOR_EXPORT_EXTENSIONS = {".pdf", ".eps"}
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return struct.unpack(">II", header[16:24])
    return None


def jpeg_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index : index + 2], "big")
        if marker in range(0xC0, 0xC4) and index + 7 < len(data):
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += max(length, 2)
    return None


def audit(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]
    editable_stems = {
        path.relative_to(root).with_suffix("").as_posix()
        for path in files
        if path.suffix.lower() in EDITABLE_SOURCE_EXTENSIONS
    }
    for path in files:
        suffix = path.suffix.lower()
        rel = path.relative_to(root).as_posix()
        try:
            if suffix == ".svg":
                tree = ET.parse(path)
                root_element = tree.getroot()
                if "viewBox" not in root_element.attrib:
                    warnings.append(f"{rel}: SVG lacks a viewBox")
                tags = [element.tag.rsplit("}", 1)[-1] for element in tree.iter()]
                if not any(tag in {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"} for tag in tags):
                    warnings.append(f"{rel}: SVG contains no common vector primitives")
            elif suffix == ".drawio":
                tree = ET.parse(path)
                cells = [
                    element
                    for element in tree.iter()
                    if element.tag.rsplit("}", 1)[-1] == "mxCell"
                ]
                if len(cells) < 2:
                    warnings.append(f"{rel}: draw.io source contains few editable cells")
            elif suffix == ".pptx":
                with zipfile.ZipFile(path) as archive:
                    names = set(archive.namelist())
                    if "[Content_Types].xml" not in names:
                        errors.append(f"{rel}: invalid PPTX package")
                    if not any(name.startswith("ppt/slides/slide") for name in names):
                        warnings.append(f"{rel}: PPTX contains no slides")
            elif suffix in RASTER_EXTENSIONS:
                size = png_size(path) if suffix == ".png" else jpeg_size(path)
                if not size:
                    errors.append(f"{rel}: unreadable raster dimensions")
                elif min(size) < 600:
                    warnings.append(f"{rel}: small raster export {size[0]}x{size[1]}")
                stem = path.relative_to(root).with_suffix("").as_posix()
                if stem not in editable_stems:
                    warnings.append(f"{rel}: no same-stem editable source found")
            elif suffix in VECTOR_EXPORT_EXTENSIONS:
                stem = path.relative_to(root).with_suffix("").as_posix()
                if stem not in editable_stems:
                    warnings.append(f"{rel}: vector export has no same-stem editable source")
        except (ET.ParseError, OSError, zipfile.BadZipFile) as exc:
            errors.append(f"{rel}: {exc}")
    if not files:
        warnings.append("no figure files found")
    return sorted(set(errors)), sorted(set(warnings))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_root", nargs="?", default=".")
    args = parser.parse_args()
    errors, warnings = audit(Path(args.figure_root).resolve())
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Figure audit complete: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
