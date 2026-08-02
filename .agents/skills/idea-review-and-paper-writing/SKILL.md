---
name: idea-review-and-paper-writing
description: Review research ideas, freeze a paper-specific conception baseline, write and revise the non-experimental manuscript, then produce scientific figures and evidence-grounded experimental prose in IEEE Transactions style. Use for novelty and feasibility review, paper positioning, narrative and section design, notation and terminology control, theorem/proof strengthening, manuscript rewriting, LaTeX schematics, model/principle/workflow figures, experiment-result narration, LaTeX generation, PDF compilation, or durable academic-writing rule updates.
---

# Idea Review and Paper Writing

Use exactly four work stages. Read only the current stage and its directly
linked resources.

## Route the task

- **Stage 1 — idea review:** review novelty, closest work, assumptions,
  information availability, realizability, model routes, and analytical scope.
  Read `references/stages/idea-exploration.md`.
- **Stage 2 — paper conception:** settle the background/task, contributions,
  technical story, model-definition order, notation, terminology, chapter
  architecture, and narrative plan. Read
  `references/stages/paper-conception.md` and work only in
  `MANUSCRIPT_CONTEXT.md` inside the active revision folder, created from
  `assets/templates/manuscript-context.md`.
- **Stage 3 — manuscript writing:** write and revise the non-experimental
  technical manuscript from the frozen conception, including compact LaTeX
  schematics for required principle, model, and workflow figures. Read
  `references/stages/manuscript-writing.md` and execute
  `references/manuscript-writing-loop.md` through
  `WRITING_LOOP_LOG.md` in that folder.
- **Stage 4 — figures and experiments:** turn Stage 3 schematics into polished
  scientific figures, integrate externally produced quantitative plots, write
  evidence-grounded experiment text, and finalize result-dependent manuscript
  parts. Read `references/stages/figures-and-experiments.md`,
  `references/figure-composition-rules.md`, and only the matching figure cases.

## Identify an exact frozen baseline

- `Baseline ID` is a short, readable identifier of one manuscript's conception
  lineage, normally one to three terms, preferably no more than 16 characters
  and never more than 24, such as `koop-fd`. Keep it unchanged while revising
  the same paper.
- `Context revision` is a monotonically increasing integer. Increment it every
  time Stage 2 changes a frozen conception and freezes it again.
- The pair `Baseline ID + Context revision` identifies the exact Stage 2
  snapshot used by Stage 3 and Stage 4. A genuinely different central paper
  task receives a new baseline ID and normally a new paper case.
- Name the revision folder `<baseline-id>-r<context-revision>`. Keep short role
  filenames such as `manuscript.tex`, `manuscript.pdf`, and
  `MANUSCRIPT_CONTEXT.md` inside it. Read `references/artifact-naming.md` in
  Stages 2–4.

## Stage transitions

- Stage 1 ends with a recorded idea assessment and a sufficiently confirmed
  route; it does not draft claims as settled prose.
- Stage 2 ends only when the versioned context file is `FROZEN`, has a nonblank
  stable baseline ID, positive revision, matching revision folder, and passes all
  conception checks.
- Stage 3 follows that exact snapshot and writes no experimental result prose.
  A conception change returns to Stage 2; a new novelty or validity issue
  returns first to Stage 1.
- Stage 4 begins only when the technical draft and LaTeX figure schematics are
  stable and actual experimental results or reviewed plot exports are
  available. It must not invent missing evidence.

## Load only relevant context

- Read `references/living-user-rules.md` for durable cross-manuscript rules and
  the formal-claim taxonomy.
- Read only the exact matching domain file and paper case. Follow
  `references/domains/README.md` and `cases/README.md`.
- For prose calibration, read `references/style-exemplars/README.md`, then one
  task-matched published-paper case and only the needed synthesis file.
- For figure calibration, read `cases/figure-exemplars/README.md`, then only
  the matching one of the three figure-category cases. Reference figures are
  inventory entries inside that category, not separate case directories, and
  are layout evidence rather than technical or copyright authority.
- Read `references/technical-validity-and-implementation.md` only when a model,
  theorem, loss, constraint, algorithm, or computation is evaluated or changed.
- Read `references/tfs-reference-patterns.md` only for relevant IEEE TFS work,
  `references/scientific-figure-palettes.md` only in Stage 4, and
  `references/typical-errors.md` during Stage 3 or Stage 4 QA.

## Keep one source of truth

- `MANUSCRIPT_CONTEXT.md` inside the active revision folder owns all Stage 2
  conception decisions for that frozen revision.
- `WRITING_LOOP_LOG.md` records Stage 3 paragraph, subsection,
  schematic, and manuscript-alignment evidence; it cannot change the frozen
  conception.
- Stage 4 figure plans and delivery records own figure construction and
  experimental-result integration, without redefining the technical method.
- Cases preserve one-paper histories or one of three figure-category
  inventories; references hold reusable rules; assets hold templates and
  authorized reusable visual resources. Use
  `references/rule-scope-map.md` when maintaining boundaries.

## Preserve every user-supplied detail

- When maintaining this skill with `skill-creator`, use its advice for routing,
  boundary definition, deduplication, validation, and progressive disclosure;
  never treat concision as permission to discard a user rule, exception,
  example, project-specific detail, or explicit unknown boundary.
- Before moving, merging, or rewriting rules, follow
  `references/detail-preservation-and-refactoring.md` and update the coverage
  ledger. Keep one authoritative copy and route to it instead of deleting
  semantic content.

## Validate

- Run `python -X utf8 scripts/validate_skill.py .` after changing the skill.
- Run `python -X utf8 scripts/audit_artifact_names.py <revision-folder>` in Stage 3
  and add `--require-pdf --require-stage4` before Stage 4 delivery.
- In Stage 3, run `scripts/audit_writing_loops.py` and the manuscript/LaTeX
  checks.
- In Stage 4, run `scripts/audit_figures.py`, inspect every export at final
  size, then rerun manuscript, compilation, PDF, and visual checks.
- Preserve source assets and disclose unresolved evidence, reference, or
  validation limits.

Maintain one canonical checkout. Expose another skill location through a
junction or symbolic link rather than an independently edited duplicate.
