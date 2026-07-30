---
name: idea-review-and-paper-writing
description: Explore and validate research ideas, then write, revise, illustrate, and quality-check academic manuscripts in IEEE Transactions style, especially IEEE TFS papers involving nonlinear systems, T-S fuzzy models, Koopman operators, observers, fault diagnosis, neural networks, stability proofs, scientific figures, and LaTeX. Use for novelty and feasibility exploration, model-specific theoretical analysis, manuscript rewriting, notation unification, theorem/proof strengthening, IEEE organization, paper figures, LaTeX generation, PDF compilation, or incorporation of durable rules.
---

# Idea Review and Paper Writing

Use exactly two work stages. Supporting references and cases are resources, not additional stages.

## Route the task

- Use **Stage 1** for novelty review, idea comparison, assumptions, feasibility, model design, theory-route exploration, or experiment planning. Read `references/stages/idea-exploration.md`.
- Use **Stage 2** for manuscript drafting, revision, notation, formal results, citations, experiments, LaTeX, scientific figures, compilation, or delivery. Read `references/stages/journal-paper-writing-and-figures.md` and execute `references/manuscript-quality-gates.md`; for a scientific-figure task, also read `references/scientific-figure-palettes.md`; for every Chinese-language LaTeX manuscript or any IEEE LaTeX work, also read `references/latex-template-workflow.md`.
- Return a newly exposed technical or novelty question from Stage 2 to Stage 1 before writing it as a confirmed claim.

## Load only relevant context

- Read `references/living-user-rules.md` for durable cross-manuscript preferences and the formal-claim taxonomy.
- Read the matching section of `references/user-writing-requirements-and-preferences.md` when the task concerns a recorded project-specific writing rule, review/report format, prompt-writing preference, personal prose style, workflow preference, or an explicitly unknown preference. Locate the narrowest relevant heading first with `rg -n "^## |^### " references/user-writing-requirements-and-preferences.md`, then read that complete section. Do not globalize a project-only entry.
- Read `references/technical-validity-and-implementation.md` whenever a model, theorem, loss, constraint, training loop, experiment, or online algorithm is evaluated or changed.
- Read only the matching domain file: `references/domains/fault-diagnosis.md` or `references/domains/soft-sensing-and-observers.md`. Follow `references/domains/README.md` before adding another domain.
- Read `references/tfs-reference-patterns.md` only for relevant IEEE TFS work.
- Read `references/typical-errors.md` as the pre-delivery negative checklist.
- Read the active project's `MANUSCRIPT_CONTEXT.md`. Create it from `assets/templates/manuscript-context.md` when the project is sufficiently established.
- Read an exact matching case when one exists. Cases from the same task may suggest candidate routes and known risks, but they do not establish novelty, assumptions, proofs, or applicability for the current paper. Follow `cases/README.md`.

## Maintain one source of truth

Use `references/rule-scope-map.md` to place each reusable rule in one authoritative file. Keep summaries and links short; do not maintain independent full copies.

- Put cross-manuscript preferences in `references/living-user-rules.md`.
- Preserve the complete user-issued inventory and its explicit unknown boundaries in `references/user-writing-requirements-and-preferences.md`; use `references/source-rule-coverage.md` to audit legacy and web-source retention during refactors.
- Put domain-reusable rules under `references/domains/`.
- Put active one-paper decisions in `MANUSCRIPT_CONTEXT.md`.
- Put longer one-paper history in one case under `cases/<task-group>/`, created from `assets/templates/paper-case.md`.
- Generalize recurring errors in `references/typical-errors.md`.

## Enforce the Stage 2 writing loop

Do not treat section organization, notation consistency, or
abstract–problem–contribution alignment as a final proofreading pass. Maintain
`SECTION_ROLE_MATRIX.md` and `NOTATION_LEDGER.md`, register symbols before use,
and repeat the subsection and manuscript gates in
`references/manuscript-quality-gates.md`. A failed or blocked gate prevents
advancement and delivery.

After every substantive subsection edit, recheck its chapter placement against
the skill sequence and audit sentence-to-sentence logic, narrative causality,
symbol consistency, formula rigor, model-description completeness, and the
clarity and separation of training, validation, testing, and deployment.
For every new or changed symbol, verify its first-definition route and naming
basis: prefer a field or mathematical convention, then a semantically meaningful
English initial or mnemonic, and use another unclaimed symbol only with an
explicit rationale. Meaning, object-type, or semantic-family conflicts require
a global rename before drafting continues.

## Validate

- Run `python -X utf8 scripts/validate_skill.py .` after changing the skill.
- In Stage 2, run `python -X utf8 scripts/audit_writing_loops.py <project-root>` plus the relevant manuscript, figure, LaTeX, PDF, and visual checks defined in the stage guide.
- Preserve source assets and report unresolved evidence or validation limits explicitly.

Maintain one canonical checkout of this repository. If the skill is exposed from another Codex skill directory, use a junction or symbolic link to the canonical checkout instead of maintaining an independently edited duplicate.
