#!/usr/bin/env python3
"""Validate this skill's structure, links, Markdown rules, and rule uniqueness."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
LINK_RE = re.compile(r"\]\(([^)]+)\)")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, text[match.end() :]


def normalize_rule(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "|", "```", "<!--")):
        return None
    if stripped.startswith("- "):
        stripped = stripped[2:]
    if len(stripped) < 50 or stripped.count("`") > 8:
        return None
    return re.sub(r"\s+", " ", stripped).casefold()


def authoritative_files(root: Path) -> list[Path]:
    paths = [
        root / "SKILL.md",
        root / "references" / "living-user-rules.md",
        root / "references" / "technical-validity-and-implementation.md",
        root / "references" / "tfs-reference-patterns.md",
    ]
    paths.extend(sorted((root / "references" / "stages").glob("*.md")))
    paths.extend(
        path
        for path in sorted((root / "references" / "domains").glob("*.md"))
        if path.name.lower() != "readme.md"
    )
    return [path for path in paths if path.exists()]


def audit_markdown(root: Path, errors: list[str]) -> None:
    stale_names = {
        "references/idea-exploration-checks.md",
        "references/MANUSCRIPT_CONTEXT.template.md",
        "cases/CASE_TEMPLATE.md",
        "references/stages/README.md",
    }
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        lines = text.splitlines()
        in_display = False
        for index, line in enumerate(lines):
            number = index + 1
            previous = lines[index - 1] if index else ""
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if re.match(r"^\d+\.\s", line) and index and previous != "":
                errors.append(f"{rel}:{number}: ordered-list item lacks a blank line")
            if line == "$$":
                if not in_display:
                    if index and previous != "":
                        errors.append(
                            f"{rel}:{number}: opening display delimiter lacks a blank line"
                        )
                    in_display = True
                else:
                    if index + 1 < len(lines) and following != "":
                        errors.append(
                            f"{rel}:{number}: closing display delimiter lacks a blank line"
                        )
                    in_display = False
            if "\ufffd" in line or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", line):
                errors.append(f"{rel}:{number}: invalid control or replacement character")
            for target in LINK_RE.findall(line):
                if re.match(r"^(?:https?://|mailto:|#)", target) or target.startswith("<"):
                    continue
                clean = target.split("#", 1)[0]
                if clean and not (path.parent / clean).exists():
                    errors.append(f"{rel}:{number}: missing link target {target}")
        if in_display:
            errors.append(f"{rel}: unmatched display-math delimiters")
        if any(name in text for name in stale_names):
            errors.append(f"{rel}: contains a stale resource path")
        if (
            rel.startswith(("references/", "cases/", "assets/templates/"))
            and path.name.lower() != "readme.md"
            and len(lines) > 100
            and "## Contents" not in "\n".join(lines[:35])
        ):
            errors.append(f"{rel}: long document lacks a top Contents section")


def audit_unique_rules(root: Path, errors: list[str]) -> None:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for path in authoritative_files(root):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            normalized = normalize_rule(line)
            if normalized:
                occurrences[normalized].append(
                    f"{path.relative_to(root).as_posix()}:{number}"
                )
    for locations in occurrences.values():
        if len(locations) > 1:
            errors.append(
                "duplicate authoritative rule text: " + ", ".join(locations)
            )


def audit_agent_metadata(
    root: Path, skill_name: str, errors: list[str], warnings: list[str]
) -> None:
    path = root / "agents" / "openai.yaml"
    if not path.exists():
        warnings.append("agents/openai.yaml is missing")
        return
    text = path.read_text(encoding="utf-8")
    short_match = re.search(r'^\s*short_description:\s*"([^"]+)"', text, re.MULTILINE)
    prompt_match = re.search(r'^\s*default_prompt:\s*"([^"]+)"', text, re.MULTILINE)
    if not short_match or not 25 <= len(short_match.group(1)) <= 64:
        errors.append("agents/openai.yaml short_description must be 25–64 characters")
    if not prompt_match or f"${skill_name}" not in prompt_match.group(1):
        errors.append(
            f"agents/openai.yaml default_prompt must explicitly mention ${skill_name}"
        )


def audit_repository_contract(root: Path, errors: list[str]) -> None:
    required_files = {
        "README.md",
        "VERSION",
        "agents/openai.yaml",
        "assets/templates/figure-plan.md",
        "assets/templates/idea-assessment.md",
        "assets/templates/manuscript-context.md",
        "assets/templates/notation-ledger.md",
        "assets/templates/paper-case.md",
        "assets/templates/section-role-matrix.md",
        "assets/palettes/scientific-figure-palettes.svg",
        "assets/latex-templates/sources.json",
        "cases/README.md",
        "cases/fault-diagnosis/koopman-ts-attention-unknown-fault-diagnosis.md",
        "cases/soft-sensing/long-memory-contraction-observer-quality-prediction.md",
        "references/domains/README.md",
        "references/domains/fault-diagnosis.md",
        "references/domains/soft-sensing-and-observers.md",
        "references/latex-template-workflow.md",
        "references/manuscript-quality-gates.md",
        "references/rule-scope-map.md",
        "references/scientific-figure-palettes.md",
        "references/source-rule-coverage.md",
        "references/stages/idea-exploration.md",
        "references/stages/journal-paper-writing-and-figures.md",
        "references/technical-validity-and-implementation.md",
        "references/tfs-reference-patterns.md",
        "references/typical-errors.md",
        "references/user-writing-requirements-and-preferences.md",
        "scripts/audit_figures.py",
        "scripts/audit_manuscript.py",
        "scripts/audit_writing_loops.py",
        "scripts/compile_manuscript.py",
        "scripts/fetch_latex_templates.py",
        "scripts/create_latex_project.py",
        "scripts/audit_latex_template.py",
        "scripts/render_palette_strips.py",
        "tests/forward-tests.json",
        "tests/fixtures/writing-loop/main.tex",
        "tests/fixtures/writing-loop/NOTATION_LEDGER.md",
        "tests/fixtures/writing-loop/NOTATION_LEDGER_TYPE_CONFLICT.md",
        "tests/fixtures/writing-loop/SECTION_ROLE_MATRIX.md",
    }
    for relative in sorted(required_files):
        if not (root / relative).is_file():
            errors.append(f"required repository resource is missing: {relative}")

    stage_dir = root / "references" / "stages"
    stage_files = (
        sorted(path.name for path in stage_dir.glob("*.md"))
        if stage_dir.is_dir()
        else []
    )
    expected_stages = [
        "idea-exploration.md",
        "journal-paper-writing-and-figures.md",
    ]
    if stage_files != expected_stages:
        errors.append(
            "the skill must have exactly the two stage documents: "
            + ", ".join(expected_stages)
        )

    content_contracts = {
        "SKILL.md": (
            "Stage 1",
            "Stage 2",
            "references/manuscript-quality-gates.md",
            "sentence-to-sentence logic",
            "narrative causality",
            "references/rule-scope-map.md",
            "references/latex-template-workflow.md",
        ),
        "references/stages/idea-exploration.md": (
            "## Gate A:",
            "## Gate B:",
            "## Gate C:",
            "## Repetition points",
        ),
        "references/stages/journal-paper-writing-and-figures.md": (
            ".drawio",
            "structured SVG",
            ".pptx",
            "CCF-Figure",
            "references/manuscript-quality-gates.md",
            "references/scientific-figure-palettes.md",
            "Scientific integrity and copyright",
            "references/latex-template-workflow.md",
        ),
        "references/manuscript-quality-gates.md": (
            "## Gate 0:",
            "## Gate 1:",
            "## Gate 2:",
            "## Gate 3:",
            "## Gate 4:",
            "## Gate 5:",
            "SECTION_ROLE_MATRIX.md",
            "NOTATION_LEDGER.md",
            "naming basis",
            "first-definition route",
            "Monitoring Objectives",
            "## Seven mandatory subsection audits",
            "Chapter and subsection arrangement",
            "Sentence-to-sentence logic",
            "Narrative causality",
            "Formula rigor",
            "Model-description completeness",
            "Training, validation, testing, and deployment clarity",
            "DRAFT -> CHECK -> FAIL -> REVISE -> CHECK",
        ),
        "assets/templates/section-role-matrix.md": (
            "## Chapter arrangement conformance",
            "Actual top-level sequence:",
            "## Subsection writing-loop record",
            "Sentence-to-sentence logic",
            "Narrative causality",
            "Symbol consistency",
            "Formula rigor",
            "Model completeness",
            "Training / validation / testing / deployment clarity",
            "Evidence inspected and revision action",
        ),
        "assets/templates/manuscript-context.md": (
            "## Contents",
            "## Notation control artifact",
            "Canonical ledger: `NOTATION_LEDGER.md`",
            "Do not maintain a second symbol table",
            "semantic family, naming basis or convention, and scope",
        ),
        "scripts/audit_writing_loops.py": (
            "required_markers",
            "actual top-level chapter sequence is blank",
            "bare PASS without evidence",
            "subsection-loop row",
            "multiple object types",
            "naming basis; use a field/journal",
        ),
        "assets/templates/notation-ledger.md": (
            "Naming basis / convention",
            "field standard:",
            "mathematical convention:",
            "English initial:",
            "semantic mnemonic:",
            "project-specific:",
            "Introduction Notation",
        ),
        "references/living-user-rules.md": (
            "semantically meaningful English initial or mnemonic",
            "Record that naming basis in the Markdown notation ledger",
        ),
        "references/source-rule-coverage.md": (
            "8bc7e1d",
            "## Legacy-version coverage",
            "## Web-conversation coverage",
            "## Reconciled conflicts",
            "three to five concrete",
            r"\mathbb H^{\infty}",
        ),
        "references/user-writing-requirements-and-preferences.md": (
            "## 通用写作原则",
            "## 论文结构",
            "## 中文与英文论文语言",
            "## 公式、排版和 PDF",
            "## 质量指标预测非线性观测器项目",
            "## Koopman、T–S 模糊与 Attention 故障诊断项目",
            "## 缺失数据补全综述",
            "## 审稿、查重和评分",
            "## 期刊、会议与投稿调研报告",
            "## 志愿填报和研究报告",
            "## 图像、视频和游戏素材提示词",
            "## 一般文字风格和邮件",
            "## Obsidian 写作工作流",
            "## 通用规则与项目专用规则的边界",
            "## 尚未形成固定偏好的事项",
        ),
        "references/scientific-figure-palettes.md": (
            "Coolors Trending Palettes",
            "summer-ocean-breeze",
            "dark-sunset",
            "earthy-green",
            "fresh-greens",
            "ocean-breeze",
            "warm-autumn-glow",
            "whimsical-dreams",
            "vibrant-spring",
            "cool-waters",
            "watermelon-sorbet",
            "sunny-beach-day-deep-orange",
            "bold-berry",
            "pastel-dreamland-adventure",
            "bold-hues",
            "earthy-tones-muted",
            "Select exactly one named palette strip",
            "#E63946",
            "#1D3557",
        ),
        "references/latex-template-workflow.md": (
            "assets/latex-templates/sources.json",
            "TEMPLATE_LOCK.json",
            "picins.sty",
            "Do not patch the template around the problem.",
            "default for every Chinese-language paper",
            "\\begin{multicols}{2}",
        ),
        "references/tfs-reference-patterns.md": (
            "TFS-oriented narrative",
            "cases/<task-group>/",
        ),
        "cases/README.md": (
            "Cases in the same task group may be consulted",
            "model-predictive-control/",
            "data-completion/",
            "One file represents one paper.",
        ),
        "references/domains/README.md": (
            "model predictive control",
            "data completion",
        ),
        "README.md": (
            "## 目录",
            "## 两阶段闭环规则",
            "### 阶段 1：构思探索闭环",
            "Gate A：现有工作重合",
            "### 阶段 2：论文写作闭环",
            "Gate 0：初始化",
            "Gate 3 的七项小节检查缺一不可",
            "DRAFT -> CHECK -> FAIL -> REVISE -> CHECK",
            "## 仓库结构",
            "## 可编辑科学图件",
            "## 官方 LaTeX 模板",
            "## 验证与前向测试",
            "## 版本与发布",
        ),
        "README.en.md": (
            "## Contents",
            "## Two-stage loop rules",
            "### Stage 1: idea-exploration loop",
            "Gate A: prior-art overlap",
            "### Stage 2: manuscript-writing loop",
            "Gate 0: initialize",
            "All seven Gate 3 subsection audits are mandatory",
            "DRAFT -> CHECK -> FAIL -> REVISE -> CHECK",
            "## Repository map",
            "## Editable scientific figures",
            "## Official LaTeX templates",
            "## Validation and forward tests",
            "## Versioning and release",
        ),
    }
    for relative, required_fragments in content_contracts.items():
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in required_fragments:
            if fragment not in text:
                errors.append(f"{relative}: missing contract fragment {fragment!r}")

    tests_path = root / "tests" / "forward-tests.json"
    if tests_path.is_file():
        try:
            cases = json.loads(tests_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"tests/forward-tests.json: invalid JSON: {exc}")
        else:
            required_ids = {
                "stage1-prior-art",
                "stage1-assumptions",
                "stage1-model-specific-analysis",
                "stage2-editable-vector-figure",
                "stage2-ai-draft-palette-reconstruction",
                "stage2-official-latex-template",
                "stage2-chinese-default-template",
                "stage2-notation-registry-loop",
                "stage2-meaningful-symbol-selection-loop",
                "stage2-problem-contribution-loop",
                "stage2-seven-priority-writing-loop",
                "legacy-rule-restoration",
                "web-writing-scope-boundary",
                "case-transfer-boundary",
            }
            if not isinstance(cases, list):
                errors.append("tests/forward-tests.json must contain a list")
            else:
                ids = [
                    case.get("id")
                    for case in cases
                    if isinstance(case, dict)
                ]
                if len(ids) != len(set(ids)):
                    errors.append("tests/forward-tests.json contains duplicate ids")
                missing_ids = required_ids.difference(ids)
                if missing_ids:
                    errors.append(
                        "tests/forward-tests.json is missing required cases: "
                        + ", ".join(sorted(missing_ids))
                    )
                for index, case in enumerate(cases):
                    if not isinstance(case, dict):
                        errors.append(
                            f"tests/forward-tests.json case {index} is not an object"
                        )
                        continue
                    for key in ("id", "stage", "prompt", "rubric", "forbidden"):
                        if not case.get(key):
                            errors.append(
                                "tests/forward-tests.json "
                                f"case {index} lacks nonempty {key!r}"
                            )


def audit_template_registry(root: Path, errors: list[str]) -> None:
    manifest_path = root / "assets" / "latex-templates" / "sources.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"assets/latex-templates/sources.json: invalid JSON: {exc}")
        return
    templates = manifest.get("templates")
    expected_ids = {"ieee-journal", "control-theory-and-applications"}
    if not isinstance(templates, dict) or set(templates) != expected_ids:
        errors.append(
            "template registry must contain exactly: " + ", ".join(sorted(expected_ids))
        )
        return
    required = {
        "display_name",
        "publisher",
        "official_guidance_url",
        "catalog_url",
        "download_url",
        "archive_filename",
        "size_bytes",
        "sha256",
        "archive_root",
        "main_file",
        "encoding",
        "toolchain_note",
        "license",
        "redistribution_note",
        "project_files",
        "known_dependencies",
    }
    for template_id, record in templates.items():
        missing = sorted(required.difference(record))
        if missing:
            errors.append(
                f"template registry {template_id} lacks: " + ", ".join(missing)
            )
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            errors.append(f"template registry {template_id} has invalid SHA-256")
        if not isinstance(record["size_bytes"], int) or record["size_bytes"] <= 0:
            errors.append(f"template registry {template_id} has invalid size_bytes")
        if record["main_file"] not in record["project_files"]:
            errors.append(
                f"template registry {template_id} main_file is not in project_files"
            )
        if len(record["project_files"]) != len(set(record["project_files"])):
            errors.append(
                f"template registry {template_id} contains duplicate project_files"
            )
        for url_field in ("official_guidance_url", "catalog_url", "download_url"):
            if not str(record[url_field]).startswith("https://"):
                errors.append(
                    f"template registry {template_id} {url_field} must use HTTPS"
                )
    ignore_path = root / ".gitignore"
    if not ignore_path.is_file() or "/assets/latex-templates/cache/" not in ignore_path.read_text(
        encoding="utf-8"
    ):
        errors.append("official template cache must be excluded by .gitignore")


def audit_writing_loop_fixture(root: Path, errors: list[str]) -> None:
    script = root / "scripts" / "audit_writing_loops.py"
    fixture = root / "tests" / "fixtures" / "writing-loop"
    if not script.is_file() or not fixture.is_dir():
        return
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), str(fixture), "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        errors.append("writing-loop fixture failed: " + detail)
        return
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"writing-loop fixture returned invalid JSON: {exc}")
        return
    if report.get("errors"):
        errors.append(
            "writing-loop fixture reported errors: "
            + "; ".join(str(item) for item in report["errors"])
        )

    negative_ledger = fixture / "NOTATION_LEDGER_TYPE_CONFLICT.md"
    if not negative_ledger.is_file():
        return
    with tempfile.TemporaryDirectory(prefix="writing-loop-conflict-") as temp_name:
        temp_root = Path(temp_name)
        shutil.copy2(fixture / "main.tex", temp_root / "main.tex")
        shutil.copy2(
            fixture / "SECTION_ROLE_MATRIX.md",
            temp_root / "SECTION_ROLE_MATRIX.md",
        )
        shutil.copy2(negative_ledger, temp_root / "NOTATION_LEDGER.md")
        negative = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), str(temp_root), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        try:
            negative_report = json.loads(negative.stdout)
        except json.JSONDecodeError as exc:
            errors.append(
                "writing-loop type-conflict fixture returned invalid JSON: "
                + str(exc)
            )
            return
        expected = (
            "has multiple object types",
            "project-specific symbol without recording its rationale",
        )
        reported = [str(item) for item in negative_report.get("errors", [])]
        if negative.returncode == 0 or not all(
            any(marker in item for item in reported) for marker in expected
        ):
            errors.append(
                "writing-loop conflict fixture did not produce the required "
                "object-type and project-specific-rationale errors"
            )


def validate(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_path = root / "SKILL.md"
    if not skill_path.exists():
        return ["SKILL.md is missing"], warnings
    skill_text = skill_path.read_text(encoding="utf-8")
    try:
        frontmatter, body = parse_frontmatter(skill_text)
    except ValueError as exc:
        return [str(exc)], warnings
    if set(frontmatter) != {"name", "description"}:
        errors.append("SKILL.md frontmatter must contain only name and description")
    skill_name = frontmatter.get("name", "")
    if skill_name != root.name:
        errors.append(
            f"skill name {skill_name!r} does not match folder name {root.name!r}"
        )
    if len(skill_text.splitlines()) > 500:
        errors.append("SKILL.md exceeds 500 lines")
    if not body.strip():
        errors.append("SKILL.md body is empty")
    version_path = root / "VERSION"
    if not version_path.exists():
        errors.append("VERSION is missing")
    elif not SEMVER_RE.fullmatch(version_path.read_text(encoding="utf-8").strip()):
        errors.append("VERSION must contain MAJOR.MINOR.PATCH")
    audit_markdown(root, errors)
    audit_unique_rules(root, errors)
    audit_repository_contract(root, errors)
    audit_template_registry(root, errors)
    audit_writing_loop_fixture(root, errors)
    audit_agent_metadata(root, skill_name, errors, warnings)
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("skill_dir", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.skill_dir).resolve()
    errors, warnings = validate(root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print("Skill validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
