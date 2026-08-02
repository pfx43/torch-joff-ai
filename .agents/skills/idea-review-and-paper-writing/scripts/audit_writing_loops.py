#!/usr/bin/env python3
"""Audit the frozen Stage-2 baseline and Stage-3 manuscript-writing loops."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SECTION_RE = re.compile(
    r"\\(?P<level>section|subsection|subsubsection)\*?\{(?P<title>[^{}]+)\}"
)
ABSTRACT_RE = re.compile(
    r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.DOTALL
)
DISPLAY_RE = re.compile(
    r"\\begin\{(?:equation|align|gather|multline|displaymath)\*?\}|\\\[|\$\$"
)
CONTRIBUTION_LEAD_RE = re.compile(
    r"(?:main|principal|primary)\s+contributions?\b", re.IGNORECASE
)
BEGIN_ENUM_RE = re.compile(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", re.DOTALL)
ITEM_RE = re.compile(r"\\item\b")
COMMAND_RE = re.compile(r"\\[A-Za-z@]+(?:\[[^\]]*\])?")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
DIMENSION_N_RE = re.compile(r"\\(?:mathbb\{R\}|R)\s*\^\s*\{\s*n(?:_|[A-Za-z])")


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        output: list[str] = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            output.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(output))
    return "\n".join(lines)


def select_main_tex(root: Path, requested: str | None) -> Path | None:
    if requested:
        path = (root / requested).resolve()
        return path if path.is_file() else None
    candidates = sorted(root.rglob("*.tex"))
    main = [
        path
        for path in candidates
        if "\\documentclass" in path.read_text(encoding="utf-8", errors="replace")
    ]
    if len(main) == 1:
        return main[0]
    return main[0] if main else (candidates[0] if len(candidates) == 1 else None)


def extract_heading_block(text: str, title_pattern: str) -> str | None:
    match = re.search(
        rf"\\subsection\*?\{{(?:{title_pattern})\}}", text, re.IGNORECASE
    )
    if not match:
        return None
    following = SECTION_RE.search(text, match.end())
    end = following.start() if following else len(text)
    return text[match.end() : end]


def contribution_items(text: str) -> list[str]:
    lead = CONTRIBUTION_LEAD_RE.search(text)
    if not lead:
        return []
    enum = BEGIN_ENUM_RE.search(text, lead.end())
    if not enum:
        return []
    chunks = re.split(r"\\item\b", enum.group(1))
    return [chunk.strip() for chunk in chunks[1:] if chunk.strip()]


def prose_word_count(text: str) -> int:
    cleaned = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    cleaned = COMMAND_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[{}~]", " ", cleaned)
    return len(WORD_RE.findall(cleaned))


def extract_markdown_section(text: str, heading: str) -> str | None:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$"
        rf"(?P<body>.*?)(?=^##\s+|\Z)",
        text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group("body") if match else None


def parse_markdown_table_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        if index + 1 >= len(lines):
            continue
        separator = lines[index + 1].strip()
        if not re.fullmatch(r"\|?[\s:|-]+\|?", separator):
            continue
        headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for data_line in lines[index + 2 :]:
            if not data_line.strip().startswith("|"):
                break
            cells = [
                cell.strip() for cell in data_line.strip().strip("|").split("|")
            ]
            if len(cells) != len(headers):
                continue
            if not any(cells):
                continue
            rows.append(dict(zip(headers, cells)))
        return headers, rows
    return [], []


def parse_markdown_table(
    path: Path, section: str | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if section:
        block = extract_markdown_section(text, section)
        if block is None:
            return [], []
        text = block
    return parse_markdown_table_text(text)


def normalized_symbol(symbol: str) -> str:
    return re.sub(r"\s+", "", symbol.strip().strip("$"))


def base_family(symbol: str) -> str:
    value = normalized_symbol(symbol)
    value = re.sub(
        r"\\(?:bm|mathbf|boldsymbol|mathcal|mathscr|mathfrak|mathbb|mathrm)"
        r"\s*\{([^{}]+)\}",
        r"\1",
        value,
    )
    value = re.sub(
        r"\\(?:boldsymbol|mathbf|mathcal|mathscr|mathfrak|mathbb|mathrm|bm)",
        "",
        value,
    )
    value = re.sub(r"[_^].*$", "", value)
    greek = re.search(r"\\([A-Za-z]+)", value)
    if greek:
        return greek.group(1).lower()
    latin = re.search(r"[A-Za-z]", value)
    return latin.group(0).lower() if latin else value.lower()


def audit_tex(path: Path, errors: list[str], warnings: list[str]) -> tuple[int, int]:
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    headings = list(SECTION_RE.finditer(text))
    if not headings:
        errors.append(f"{path.name}: no section headings found")
        return 0, 0

    for heading in headings:
        if heading.group("title").strip().lower() == "monitoring objectives":
            errors.append(
                f"{path.name}: replace 'Monitoring Objectives' with "
                "'Problem Formulation' or 'Problem Description'"
            )

    first_section_end = headings[1].start() if len(headings) > 1 else len(text)
    if DISPLAY_RE.search(text[:first_section_end]):
        errors.append(
            f"{path.name}: displayed mathematics appears before Section II"
        )

    problem = extract_heading_block(text, r"Problem Formulation|Problem Description")
    if problem is None:
        errors.append(
            f"{path.name}: Section II lacks 'Problem Formulation' or "
            "'Problem Description'"
        )
        legacy_problem = extract_heading_block(text, r"Monitoring Objectives")
        problem_count = len(ITEM_RE.findall(legacy_problem or ""))
        if problem_count:
            errors.append(
                f"{path.name}: the legacy monitoring checklist has "
                f"{problem_count} items and must be reformulated as central "
                "subproblems"
            )
    else:
        problem_count = len(ITEM_RE.findall(problem))
        if problem_count < 1:
            errors.append(f"{path.name}: problem subsection has no numbered subproblem")
        elif problem_count > 3:
            errors.append(
                f"{path.name}: problem subsection has {problem_count} items; "
                "retain at most three central subproblems"
            )

    items = contribution_items(text)
    contribution_count = len(items)
    if contribution_count == 0:
        errors.append(f"{path.name}: principal contribution list was not found")
    elif not 2 <= contribution_count <= 3:
        errors.append(
            f"{path.name}: found {contribution_count} principal contributions; "
            "retain two or three"
        )
    for index, item in enumerate(items, start=1):
        count = prose_word_count(item)
        if count > 80:
            warnings.append(
                f"{path.name}: contribution {index} has {count} words; "
                "compress it to the problem-facing construction and result"
            )

    if problem_count and contribution_count and problem_count != contribution_count:
        errors.append(
            f"{path.name}: problem count ({problem_count}) and contribution "
            f"count ({contribution_count}) are not aligned"
        )

    abstract = ABSTRACT_RE.search(text)
    if not abstract:
        errors.append(f"{path.name}: abstract environment was not found")
    else:
        count = prose_word_count(abstract.group("body"))
        if count > 220:
            errors.append(
                f"{path.name}: abstract has {count} English words; "
                "the default working maximum is 220"
            )
        elif count < 150:
            warnings.append(
                f"{path.name}: abstract has {count} English words; "
                "check whether the central result and scope are complete"
            )

    if DIMENSION_N_RE.search(text):
        errors.append(
            f"{path.name}: dimension symbols use the reserved 'n' family; "
            "use the registered dimension family"
        )
    return problem_count, contribution_count


def audit_notation_registry(
    path: Path, errors: list[str], warnings: list[str]
) -> None:
    required = {
        "Symbol",
        "Semantic family",
        "Meaning",
        "Naming basis / convention",
        "Object type",
        "Dimension",
        "Typography",
        "First definition",
        "Scope",
    }
    headers, rows = parse_markdown_table(path, "Notation registry")
    missing = sorted(required.difference(headers))
    if missing:
        errors.append(
            f"{path.name}: missing registry columns: " + ", ".join(missing)
        )
    if not rows:
        errors.append(f"{path.name}: notation registry is empty")
        return

    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    type_aliases = {
        "scalar": "scalar",
        "vector": "vector",
        "matrix": "matrix",
        "mapping": "mapping",
        "map": "mapping",
        "operator": "operator",
        "set": "set/space",
        "space": "set/space",
        "set/space": "set/space",
        "index": "index",
        "constant": "constant",
    }
    naming_basis_markers = (
        "field standard",
        "domain convention",
        "journal convention",
        "mathematical convention",
        "english initial",
        "semantic mnemonic",
        "project-specific",
        "领域惯例",
        "期刊惯例",
        "数学惯例",
        "英文首字母",
        "语义助记",
        "项目专用",
    )
    for row_number, row in enumerate(rows, start=1):
        symbol = row["Symbol"].strip()
        if not symbol:
            continue
        empty = [name for name in required if name in row and not row[name].strip()]
        if empty and not missing:
            errors.append(
                f"{path.name}: registry row {row_number} lacks "
                + ", ".join(sorted(empty))
            )
        if "," in symbol or ";" in symbol:
            errors.append(
                f"{path.name}: registry row {row_number} combines multiple "
                "symbols; use one object per row"
            )
        exact[normalized_symbol(symbol)].append(row)
        families[base_family(symbol)].append(row)

        naming_basis = row.get("Naming basis / convention", "").strip()
        naming_basis_lower = naming_basis.lower()
        if naming_basis and not any(
            marker in naming_basis_lower for marker in naming_basis_markers
        ):
            errors.append(
                f"{path.name}: registry row {row_number} has an unrecognized "
                "naming basis; use a field/journal standard, mathematical "
                "convention, English initial, semantic mnemonic, or an "
                "explicitly justified project-specific choice"
            )
        project_specific = (
            "project-specific" in naming_basis_lower
            or "项目专用" in naming_basis_lower
        )
        if naming_basis and project_specific:
            rationale = re.split(r"[:：]", naming_basis, maxsplit=1)
            if len(rationale) != 2 or not rationale[1].strip():
                errors.append(
                    f"{path.name}: registry row {row_number} uses a "
                    "project-specific symbol without recording its rationale"
                )

        declared_text = row.get("Object type", "").strip().lower()
        declared = type_aliases.get(declared_text)
        if not declared:
            if "Object type" in row:
                warnings.append(
                    f"{path.name}: registry row {row_number} has unrecognized "
                    f"object type {row['Object type']!r}"
                )
            continue
        bold = "\\bm" in symbol or "\\boldsymbol" in symbol or "\\mathbf" in symbol
        mapping = "\\mathcal" in symbol or "\\mathscr" in symbol
        blackboard = "\\mathbb" in symbol
        if declared in {"vector", "matrix"} and not bold:
            errors.append(
                f"{path.name}: {symbol} is declared as {declared} but is not bold"
            )
        if declared == "scalar" and bold:
            errors.append(
                f"{path.name}: {symbol} is declared as scalar but is bold"
            )
        if declared in {"mapping", "operator"} and not mapping:
            warnings.append(
                f"{path.name}: {symbol} is declared as {declared}; "
                "verify the registered operator/mapping font"
            )
        if declared == "set/space" and not (blackboard or mapping):
            warnings.append(
                f"{path.name}: {symbol} is declared as set/space; "
                "verify its registered set typography"
            )

    for symbol, entries in exact.items():
        meanings = {entry["Meaning"].strip().lower() for entry in entries}
        object_types = {
            type_aliases.get(
                entry.get("Object type", "").strip().lower(),
                entry.get("Object type", "").strip().lower(),
            )
            for entry in entries
        }
        if len(meanings) > 1:
            errors.append(
                f"{path.name}: exact symbol {symbol!r} has multiple meanings"
            )
        if len(object_types) > 1:
            errors.append(
                f"{path.name}: exact symbol {symbol!r} has multiple object types"
            )
        if len(entries) > 1 and len(meanings) == 1 and len(object_types) == 1:
            warnings.append(
                f"{path.name}: exact symbol {symbol!r} has duplicate registry rows"
            )

    for family, entries in families.items():
        semantics = set()
        for entry in entries:
            semantic = entry.get("Semantic family", "").strip().lower()
            if not semantic:
                meaning = entry.get("Meaning", "").strip().lower()
                keyword_groups = (
                    ("fault", "fault"),
                    ("hilbert", "space"),
                    ("history", "history"),
                    ("disturbance", "noise"),
                    ("noise", "noise"),
                    ("whitening", "weight"),
                    ("weight", "weight"),
                    ("mapping", "mapping"),
                    ("transition map", "mapping"),
                    ("controller command", "input"),
                    ("plant action", "input"),
                )
                semantic = next(
                    (
                        label
                        for keyword, label in keyword_groups
                        if keyword in meaning
                    ),
                    "",
                )
            if semantic:
                semantics.add(semantic)
        if len(semantics) > 1:
            details = ", ".join(
                sorted(
                    {
                        f"{entry['Symbol']} -> "
                        f"{entry.get('Semantic family') or entry.get('Meaning')}"
                        for entry in entries
                    }
                )
            )
            errors.append(
                f"{path.name}: base family {family!r} crosses semantic families: "
                + details
            )


def metadata_value(text: str, label: str) -> str:
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(?P<value>.*?)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group("value").strip() if match else ""


def count_markdown_tables(text: str) -> int:
    lines = text.splitlines()
    return sum(
        1
        for index, line in enumerate(lines[:-1])
        if line.strip().startswith("|")
        and re.fullmatch(r"\|?[\s:|-]+\|?", lines[index + 1].strip())
    )


def audit_context(
    path: Path,
    problem_count: int,
    contribution_count: int,
    errors: list[str],
    warnings: list[str],
) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    required_sections = (
        "Context identity and control",
        "Background and task baseline",
        "Stage 1 evidence boundary",
        "Problem–contribution–result alignment",
        "Technical main line and causal story",
        "Theoretical route and dependency",
        "Model-definition order",
        "Loss and optimization baseline",
        "Notation registry",
        "Terminology and abbreviation registry",
        "Chapter and subsection blueprint",
        "Planned narrative and paragraph progression",
        "Stage 3 LaTeX schematic requirements",
        "Stage 4 evidence and figure handoff",
        "Style calibration plan",
        "Stage 2 conception gate record",
    )
    for section in required_sections:
        if extract_markdown_section(text, section) is None:
            errors.append(f"{path.name}: missing Stage-2 section {section!r}")

    status = metadata_value(text, "Context status").upper()
    baseline_id = metadata_value(text, "Baseline ID")
    revision = metadata_value(text, "Context revision")
    revision_folder = metadata_value(text, "Revision folder")
    if status != "FROZEN":
        errors.append(f"{path.name}: Context status must be FROZEN before Stage 3")
    if not baseline_id:
        errors.append(f"{path.name}: Baseline ID is blank")
    if not revision:
        errors.append(f"{path.name}: Context revision is blank")
    elif not revision.isdigit() or int(revision) < 1:
        errors.append(f"{path.name}: Context revision must be a positive integer")
    if baseline_id and revision and revision_folder != f"{baseline_id}-r{revision}":
        errors.append(
            f"{path.name}: Revision folder must be {baseline_id}-r{revision}"
        )

    if re.search(r"\|[^\n]*\|\s*(?:BLOCKED|FAIL)\s*\|\s*$", text, re.MULTILINE):
        errors.append(f"{path.name}: frozen context contains unresolved BLOCKED/FAIL table status")
    if re.search(r"^-\s*[^:\n]+:\s*$", text, re.MULTILINE):
        warnings.append(f"{path.name}: frozen context contains blank list-valued fields")
    if re.search(r"Actual top-level sequence:\s*$", text, re.MULTILINE):
        errors.append(f"{path.name}: actual top-level chapter sequence is blank")
    if count_markdown_tables(text) < 12:
        errors.append(f"{path.name}: Stage-2 context is missing required decision tables")

    alignment = extract_markdown_section(text, "Problem–contribution–result alignment") or ""
    problem_ids = set(re.findall(r"\|\s*P(\d+)\s*\|", alignment))
    contribution_ids = set(re.findall(r"\|[^\n]*\|\s*C(\d+)\s*\|", alignment))
    if len(problem_ids) < max(problem_count, 1):
        errors.append(
            f"{path.name}: context records {len(problem_ids)} central problem IDs; "
            f"the manuscript requires {problem_count or 'at least one'}"
        )
    if len(contribution_ids) < max(contribution_count, 1):
        errors.append(
            f"{path.name}: context records {len(contribution_ids)} contribution IDs; "
            f"the manuscript requires {contribution_count or 'at least one'}"
        )

    audit_notation_registry(path, errors, warnings)
    return baseline_id, revision


def audit_evidence_rows(
    path: Path,
    section: str,
    required_fields: tuple[str, ...],
    errors: list[str],
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    block = extract_markdown_section(text, section)
    if block is None:
        errors.append(f"{path.name}: missing Stage-3 section {section!r}")
        return
    headers, rows = parse_markdown_table_text(block)
    missing = [field for field in required_fields if field not in headers]
    if missing:
        errors.append(f"{path.name}: {section} table lacks " + ", ".join(missing))
        return
    if not rows:
        errors.append(f"{path.name}: {section} table is empty")
        return
    for row_number, row in enumerate(rows, start=1):
        if row.get("Status", "").strip().upper() != "PASS":
            continue
        empty = [field for field in required_fields if not row.get(field, "").strip()]
        if empty:
            errors.append(
                f"{path.name}: {section} row {row_number} is PASS but lacks "
                + ", ".join(empty)
            )
        bare = [
            field
            for field in required_fields
            if field not in {"Status", "Last checked", "Subsection", "Paragraph ID"}
            and row.get(field, "").strip().upper() == "PASS"
        ]
        if bare:
            errors.append(
                f"{path.name}: {section} row {row_number} records bare PASS "
                "without evidence for " + ", ".join(bare)
            )


def audit_writing_log(
    path: Path,
    context_baseline: str,
    context_revision: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "## Baseline binding",
        "## Paragraph purpose and progression map",
        "## Subsection loop record",
        "## LaTeX schematic inventory",
        "## Manuscript-wide alignment record",
        "## Stage 3 gate record",
        "## Context-deviation log",
        "基线一致性",
        "章节先后性",
        "叙事逻辑性",
        "表达忌生硬",
        "语句忌空洞",
        "符号一致性",
        "名词专有化",
        "公式严谨性",
        "示图完整性",
        "全文对齐性",
    )
    for marker in required_markers:
        if marker not in text:
            errors.append(f"{path.name}: missing Stage-3 writing-loop marker {marker!r}")

    if re.search(r"\|[^\n]*\|\s*(?:BLOCKED|FAIL)\s*\|\s*$", text, re.MULTILINE):
        errors.append(f"{path.name}: contains unresolved BLOCKED/FAIL table status")
    log_status = metadata_value(text, "Log status").upper()
    if log_status != "PASS":
        errors.append(f"{path.name}: Log status must be PASS for delivery")

    bound_id = metadata_value(text, "Baseline ID")
    bound_revision = metadata_value(text, "Context revision")
    bound_folder = metadata_value(text, "Revision folder")
    if not bound_id or bound_id != context_baseline:
        errors.append(f"{path.name}: baseline binding does not match the versioned context")
    if not bound_revision or bound_revision != context_revision:
        errors.append(f"{path.name}: revision binding does not match the versioned context")
    if (
        context_baseline
        and context_revision
        and bound_folder != f"{context_baseline}-r{context_revision}"
    ):
        errors.append(
            f"{path.name}: revision-folder binding does not match the frozen context"
        )
    if re.search(r"Last synchronized:\s*$", text, re.MULTILINE):
        warnings.append(f"{path.name}: synchronization date is blank")
    if count_markdown_tables(text) < 6:
        errors.append(f"{path.name}: expected six Stage-3 control tables")

    audit_evidence_rows(
        path,
        "Paragraph purpose and progression map",
        (
            "Paragraph ID",
            "Subsection",
            "Single theme or purpose",
            "Why this paragraph is needed",
            "Logical or causal link from previous paragraph",
            "Abstraction level and progression",
            "Reader-facing output or supported claim",
            "Evidence inspected and revision action",
            "Last checked",
            "Status",
        ),
        errors,
    )
    loop_block = extract_markdown_section(text, "Subsection loop record")
    if loop_block is None:
        errors.append(f"{path.name}: missing Stage-3 subsection loop table")
    else:
        headers, rows = parse_markdown_table_text(loop_block)
        required_loop_fields = (
            "Subsection",
            "Loop ID",
            "概括词语",
            "Evidence inspected",
            "Conflict found",
            "Revision/no-change reason",
            "Affected location",
            "Last checked",
            "Status",
        )
        missing = [field for field in required_loop_fields if field not in headers]
        if missing:
            errors.append(
                f"{path.name}: Subsection loop record lacks " + ", ".join(missing)
            )
        elif not rows:
            errors.append(f"{path.name}: Subsection loop record is empty")
        else:
            expected_loops = {f"W{index}" for index in range(1, 11)}
            observed: dict[str, set[str]] = defaultdict(set)
            for row_number, row in enumerate(rows, start=1):
                subsection = row.get("Subsection", "").strip()
                loop_id = row.get("Loop ID", "").strip().upper()
                if subsection and loop_id:
                    observed[subsection].add(loop_id)
                if row.get("Status", "").strip().upper() != "PASS":
                    continue
                empty = [
                    field
                    for field in required_loop_fields
                    if not row.get(field, "").strip()
                ]
                if empty:
                    errors.append(
                        f"{path.name}: subsection-loop row {row_number} is PASS "
                        "but lacks " + ", ".join(empty)
                    )
                for field in (
                    "概括词语",
                    "Evidence inspected",
                    "Conflict found",
                    "Revision/no-change reason",
                    "Affected location",
                ):
                    if row.get(field, "").strip().upper() == "PASS":
                        errors.append(
                            f"{path.name}: subsection-loop row {row_number} "
                            f"records bare PASS for {field}"
                        )
            if not observed:
                errors.append(f"{path.name}: no subsection has recorded loops")
            for subsection, loop_ids in observed.items():
                missing_loops = expected_loops.difference(loop_ids)
                extra_loops = loop_ids.difference(expected_loops)
                if missing_loops:
                    errors.append(
                        f"{path.name}: subsection {subsection!r} lacks loops "
                        + ", ".join(sorted(missing_loops))
                    )
                if extra_loops:
                    errors.append(
                        f"{path.name}: subsection {subsection!r} has unknown loops "
                        + ", ".join(sorted(extra_loops))
                    )

    gates = extract_markdown_section(text, "Stage 3 gate record") or ""
    for gate in ("S3-0", "S3-1", "S3-2", "S3-3", "S3-4", "S3-5"):
        if not re.search(rf"\|\s*{re.escape(gate)}\s*:", gates):
            errors.append(f"{path.name}: Stage-3 gate record lacks {gate}")


def audit(root: Path, requested_tex: str | None) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    tex_path = select_main_tex(root, requested_tex)
    if tex_path is None:
        errors.append(
            "unable to select one main LaTeX source; pass --tex with a relative path"
        )
        return {"errors": errors, "warnings": warnings}

    problem_count, contribution_count = audit_tex(tex_path, errors, warnings)

    context = root / "MANUSCRIPT_CONTEXT.md"
    baseline_id = ""
    context_revision = ""
    if not context.is_file():
        errors.append(
            "revision context is missing: MANUSCRIPT_CONTEXT.md"
        )
    else:
        baseline_id, context_revision = audit_context(
            context,
            problem_count,
            contribution_count,
            errors,
            warnings,
        )

    writing_log = root / "WRITING_LOOP_LOG.md"
    if not writing_log.is_file():
        errors.append(
            "revision writing log is missing: WRITING_LOOP_LOG.md"
        )
    else:
        audit_writing_log(
            writing_log,
            baseline_id,
            context_revision,
            errors,
            warnings,
        )

    return {
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument("--tex", help="main .tex path relative to project root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(Path(args.project_root).resolve(), args.tex)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(
            f"Writing-loop audit complete: {len(report['errors'])} error(s), "
            f"{len(report['warnings'])} warning(s)."
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
