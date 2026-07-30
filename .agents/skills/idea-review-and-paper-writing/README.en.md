# Idea Review and Paper Writing

[简体中文](README.md) | [English](README.en.md)

Current version: **3.5.0**

This repository contains a two-stage Codex skill for developing research ideas and turning confirmed work into an IEEE-style journal paper with publication-ready scientific figures. It is designed for more than language polishing: it also audits novelty, assumptions, information availability, theoretical scope, computational realizability, evidence, notation, LaTeX, and figure integrity.

The skill is especially developed for fuzzy systems, nonlinear systems, Koopman models, observers, fault diagnosis, neural networks, stability analysis, and related data-driven control topics. Domain-specific guidance is modular so that further research directions, such as model predictive control and data completion, can be added without turning one project's assumptions into global rules.

The plain-text [`VERSION`](VERSION) file is the repository's semantic version source. Publish GitHub releases with a matching `vMAJOR.MINOR.PATCH` tag; keep version metadata out of `SKILL.md` frontmatter because Codex skill discovery expects only `name` and `description`.

## Contents

- [Start here](#start-here)
- [Two-stage loop rules](#two-stage-loop-rules)
- [Rule hierarchy](#rule-hierarchy)
- [Repository map](#repository-map)
- [Typical task routes](#typical-task-routes)
- [Editable scientific figures](#editable-scientific-figures)
- [Official LaTeX templates](#official-latex-templates)
- [Validation and forward tests](#validation-and-forward-tests)
- [Versioning and release](#versioning-and-release)

## Start here

The skill has exactly two work stages. Start by selecting the current stage.

1. Read [`SKILL.md`](SKILL.md) for the two-stage boundary and routing rules.

2. For Stage 1, read [`references/stages/idea-exploration.md`](references/stages/idea-exploration.md).

3. For Stage 2, read [`references/stages/journal-paper-writing-and-figures.md`](references/stages/journal-paper-writing-and-figures.md) and execute [`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md).

4. In either stage, read [`references/living-user-rules.md`](references/living-user-rules.md) and the supporting documents routed by `SKILL.md`.

5. Read only the matching domain and exact paper case. Do not use unrelated domain or case content as authority.

6. Read the active project's `MANUSCRIPT_CONTEXT.md`. If it does not exist and the project is sufficiently established, create it from [`assets/templates/manuscript-context.md`](assets/templates/manuscript-context.md).

## Two-stage loop rules

This section is the executable homepage summary. The complete authority for
Stage 1 is
[`references/stages/idea-exploration.md`](references/stages/idea-exploration.md);
the complete authority for Stage 2 is
[`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md).
The README does not establish an independently editable rule set. If this
summary drifts from an authoritative file, follow the authoritative file and
resynchronize the summary.

### Stage 1: idea-exploration loop

Stage 1 must not turn an unresolved idea into a manuscript contribution. Every
pass answers three questions in order:

| Gate | Central question | Passing requirement | If it fails |
|---|---|---|---|
| Gate A: prior-art overlap | Has the idea or result already been substantively completed? | Use current primary literature and record the search date, sources, queries, closest work, substantive overlap, and exact novelty delta | Mark it `unresolved` or `exploratory`; do not use absolute priority claims |
| Gate B: assumptions and realizability | Do information, assumptions, training, numerical computation, online use, and validation form a closed route? | Separate measured, training-only, online-available, and unavailable quantities; establish trainability, offline/online computation, complexity, and validation | Revise assumptions, downgrade the route, or separate added structure as an alternative; do not silently change the information setting |
| Gate C: model-specific theory | Is analysis for the particular model a nontrivial contribution? | The conclusion must depend essentially on the structure, state complete assumptions, contain a checkable derivation, and produce a computable condition, design rule, bound, stability/detectability result, or another testable consequence | Narrow the scope, complete the proof, or present it as a property or analytical result; model-combination uniqueness alone is insufficient |

Repeat Gates A, B, and C:

- when the idea is first proposed;
- whenever the model structure or information setting changes;
- before selecting core contributions, promoting a route to `confirmed`, or
  drafting a theorem, proposition, or central analytical claim;
- after updating the closest literature or when experiment design shows that a
  claim is not testable;
- before finalizing the title, abstract, Introduction contributions, and
  Conclusion.

Record every pass in `idea-assessment.md` or the matching case and synchronize
the decision to `MANUSCRIPT_CONTEXT.md`. The passing loop is:

`candidate idea -> Gates A/B/C -> confirmed -> Stage 2`

If any gate fails:

`candidate idea -> failed check -> downgrade/revise assumptions/separate an alternative/redefine the contribution -> recheck`

Move to Stage 2 only when the technical problem, information boundary, closest
literature, two or three candidate contribution themes, assumptions and
realizability, selected route, analytical results, and validation plan are
sufficiently explicit and the central route is `confirmed`. Secondary
unresolved items may remain, but the title, abstract, contributions, and formal
claims must not present them as completed results.

### Stage 2: manuscript-writing loop

Stage 2 uses `SECTION_ROLE_MATRIX.md` and `NOTATION_LEDGER.md` as control
artifacts and executes Gates 0–5:

| Gate | Action | Blocking condition |
|---|---|---|
| Gate 0: initialize | Read the complete manuscript and `MANUSCRIPT_CONTEXT.md`; create or update the section-role matrix and notation ledger | A control artifact is absent or stale, or section, symbol, dimension, and problem–contribution conflicts are not marked |
| Gate 1: section responsibilities | Give every section one primary scientific question and one reader-facing output; check the compact sequence, prerequisites, duplicated responsibilities, and problem–contribution–body mapping | Duplicate responsibility, incorrect dependency order, unjustified deviation from the skill sequence, or mismatched problem/contribution count or order |
| Gate 2: notation registry | Prefer field/journal convention, then mathematical convention, meaningful English initials or mnemonics; register semantics, naming basis, type, dimension, typography, first-definition route, and scope | A symbol is unregistered or conflicts in meaning, type, semantic family, typography, or first definition |
| Gate 3: subsection audit | After every substantive subsection edit, reread it, reconcile its responsibility and notation, and perform all seven checks below | Any check lacks concrete evidence and revision action or remains `FAIL`/`BLOCKED` |
| Gate 4: main-line alignment | Align the abstract, Problem Formulation/Description, two or three principal contribution themes, body results, and Conclusion in count, order, and scope | These locations state different task counts, ordering, or conclusions, or treat a module/implementation detail as a contribution |
| Gate 5: delivery audit | Run the writing-loop and manuscript audits, compile and inspect PDF/figures, and recheck the final ledger, matrix, subsection records, and Gate 4 | An unresolved warning, error, stale check, or overstated delivery status remains |

All seven Gate 3 subsection audits are mandatory:

1. **Chapter and subsection arrangement:** check the compact skill sequence,
   prerequisites, one responsibility per section, and the journal or scientific
   justification for every deviation.

2. **Sentence-to-sentence logic:** give every sentence a clear role, referent,
   and logical dependency; remove unexplained jumps, ambiguous references, and
   repeated conclusions.

3. **Narrative causality:** establish the
   problem/limitation–method need–structural action–derived result–evidence
   chain; module order and chronology are not mechanism explanations.

4. **Symbol consistency:** register every symbol before use, explain it in the
   Introduction-end Notation paragraph or at first use, and preserve one
   meaning, naming basis, object type, typography, dimension, and scope.

5. **Formula rigor:** check definition and assumption order, indices,
   dimensions, operators, initial conditions, data sources, algebraic steps,
   validity conditions, boundary cases, and conclusion scope.

6. **Model-description completeness:** specify inputs, outputs, physical and
   latent states, known and unavailable quantities, disturbances or faults,
   fixed and learned mappings, parameters, assumptions, update order, time
   indices, and component interfaces.

7. **Training/validation/testing/deployment clarity:** separate data provenance
   and splits, inputs and labels, preprocessing fit scope, objectives and
   constraints, optimization, model selection, test-only operations, metrics,
   uncertainty, offline/online computation, and leakage prevention.

Each subsection follows:

`DRAFT -> CHECK -> PASS -> NEXT`

On failure:

`DRAFT -> CHECK -> FAIL -> REVISE -> CHECK`

A bare `PASS` without evidence is not complete. Every pass records the check
time, inspected evidence, conflict, revision action, affected locations, and
resulting status. A later change to chapter order, causal chain, notation,
formula, model interface, training/testing flow, central problem, or
contribution invalidates every dependent `PASS` and requires those gates to be
rerun.

Stage 2 is complete only when the section-role matrix, notation ledger,
seven-audit record for every subsection, abstract–problem–contribution–body
alignment, automated audits, compilation, and visual checks have passed, with
only explicitly disclosed nonblocking limitations remaining.

## Rule hierarchy

The documents have different authority and scope.

| Layer | Purpose | When it applies |
|---|---|---|
| Explicit current user instruction | The user's decision for the active task | Always highest priority |
| `MANUSCRIPT_CONTEXT.md` | Confirmed and exploratory decisions for one manuscript | Only for that manuscript |
| Matching case file | Longer discussion history, alternatives, and unresolved issues for one identified paper | Only for that exact paper |
| Domain guidance | Valid patterns, cautions, and alternative routes for a research direction | Only when the manuscript matches the domain and the route is selected |
| Living user rules | Durable preferences and reasoning rules that transfer across manuscripts | Every manuscript task |
| `SKILL.md` defaults | Workflow and IEEE-oriented defaults | Unless a higher-priority instruction or project decision overrides a default |
| Typical errors and reference patterns | Negative checks and literature-informed structural guidance | When routed by `SKILL.md` |

An idea recorded in a domain document is not automatically a claim, assumption, or chosen method. The active project must classify important decisions as:

- `confirmed`: adopted and allowed to govern the manuscript;
- `exploratory`: worth analyzing but not yet a manuscript claim;
- `alternative`: a competing route that must not be mixed with the selected route;
- `rejected`: considered and deliberately excluded;
- `unresolved`: requires evidence, derivation, or an explicit user decision.

Do not silently promote an exploratory or alternative idea to a confirmed paper assumption.

## Repository map

### Core workflow

- [`SKILL.md`](SKILL.md): mandatory two-stage routing, shared rules, reference use, and delivery.
- [`agents/openai.yaml`](agents/openai.yaml): interface metadata for invoking the skill.

### Stage 1: idea exploration

- [`references/stages/idea-exploration.md`](references/stages/idea-exploration.md): repeated checks for prior-art overlap, assumptions and practical realizability, and model-specific theoretical contributions.

### Stage 2: journal-paper writing and figures

- [`references/stages/journal-paper-writing-and-figures.md`](references/stages/journal-paper-writing-and-figures.md): manuscript construction, notation, formal claims, experiments, scientific diagrams, quantitative plots, captions, visual integrity, reproducible figure sources, LaTeX, and PDF verification.
- [`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md): mandatory loop for chapter conformance, sentence logic, causal narrative, symbol naming rationale and consistency, formula rigor, model completeness, training/validation/testing/deployment clarity, and abstract–problem–contribution alignment.
- [`references/latex-template-workflow.md`](references/latex-template-workflow.md): official IEEE and 《控制理论与应用》 template selection, verified download, content-only editing boundary, dependency handling, and integrity audit.

### Shared supporting rules

- [`references/living-user-rules.md`](references/living-user-rules.md): durable preferences, notation conventions, contribution discipline, theorem taxonomy, manuscript voice, response language, and update protocol.
- [`references/user-writing-requirements-and-preferences.md`](references/user-writing-requirements-and-preferences.md): complete web-conversation inventory covering global rules, project-specific rules, non-manuscript writing preferences, and explicit unknown boundaries.
- [`references/source-rule-coverage.md`](references/source-rule-coverage.md): migration coverage, conflict reconciliation, and provenance for the first `ieee-english-paper-polish` version and the web inventory.
- [`references/technical-validity-and-implementation.md`](references/technical-validity-and-implementation.md): first-principles checks for observability and information availability, nonlinear claim scope, trainability, computability, hard constraints, offline/online separation, complexity, experiments, and review.
- [`references/rule-scope-map.md`](references/rule-scope-map.md): maintenance index showing the authoritative file and scope of each rule family; use it to avoid duplication and misplaced updates.
- [`references/typical-errors.md`](references/typical-errors.md): prohibited recurring patterns and their replacements.
- [`assets/templates/manuscript-context.md`](assets/templates/manuscript-context.md): active project-context template.
- [`assets/templates/idea-assessment.md`](assets/templates/idea-assessment.md): Stage 1 assessment artifact for applying the universal idea checks.
- [`assets/templates/section-role-matrix.md`](assets/templates/section-role-matrix.md): Stage 2 registry for section responsibilities and problem–contribution mappings.
- [`assets/templates/notation-ledger.md`](assets/templates/notation-ledger.md): Stage 2 registry for symbol semantics, naming rationale, object types, dimensions, first-definition routes, and scope.
- [`assets/templates/figure-plan.md`](assets/templates/figure-plan.md): Stage 2 planning and validation artifact for one scientific figure.

### IEEE TFS reference guidance

- [`references/tfs-reference-patterns.md`](references/tfs-reference-patterns.md): reusable model-to-proof chains, TFS narrative patterns, dynamic-threshold guidance, and citation discipline. Read it for TFS fault diagnosis, T–S/IT2 fuzzy observers, residual thresholds, event-triggered diagnosis, knowledge-distilled fuzzy diagnosis, and related reference-aware work.

### Paper-specific case library

- [`cases/README.md`](cases/README.md): case-library boundary, authority, naming, synchronization, and promotion rules.
- [`assets/templates/paper-case.md`](assets/templates/paper-case.md): template for one paper-specific discussion record.
- [`cases/fault-diagnosis/`](cases/fault-diagnosis/): individual fault-diagnosis paper cases.
- [`cases/soft-sensing/`](cases/soft-sensing/): individual soft-sensing and nonlinear-observer paper cases.

Each paper has one case file. Cases preserve local discussions and alternatives but do not create rules for other manuscripts.

### Research domains

- [`references/domains/README.md`](references/domains/README.md): domain-routing rules and instructions for adding a new research direction.
- [`references/domains/fault-diagnosis.md`](references/domains/fault-diagnosis.md): unknown-fault information boundaries, Koopman–T–S–attention modeling, measurement-decoupled normal references, joint residuals, dynamic thresholds, post-filtering, structured lifted disturbance bases, and online implementation.
- [`references/domains/soft-sensing-and-observers.md`](references/domains/soft-sensing-and-observers.md): latent-state interpretation, nonlinear observer stability, measurable/unmeasurable state partitions, contraction layers, long-memory networks, and quality-variable prediction.

Model predictive control and data completion are reserved as future domain modules. Create their documents only when concrete, reusable rules are available; do not fill them with invented defaults.

## Typical task routes

### Stage 1 tasks

New ideas, literature novelty checks, model selection, assumption analysis, feasibility, theoretical-route exploration, and experiment planning begin in `references/stages/idea-exploration.md`. Use the technical-validity and domain documents as supporting checks. A case may record one paper's discussion, but it does not contain the universal Stage 1 rules.

### Stage 2 tasks

Language and structure revision, theorem presentation, complete manuscript drafting, LaTeX delivery, architecture diagrams, workflows, plots, captions, and PDF visual QA use `references/stages/journal-paper-writing-and-figures.md`. If Stage 2 exposes an unresolved technical question, return that issue to Stage 1.

### Adding a new research direction

Follow `references/domains/README.md`. Put reusable domain knowledge in a new domain file, put one-paper choices in `MANUSCRIPT_CONTEXT.md`, and promote only genuinely cross-manuscript rules to `living-user-rules.md`.

### Adding a paper case

Follow `cases/README.md`. Create one file under the matching task group, link it to the active `MANUSCRIPT_CONTEXT.md`, and do not use it as authority for another paper.

## Editable scientific figures

For architecture, mechanism, and workflow diagrams, the preferred deliverable is an editable source plus the requested publication export. Supported source routes include `.drawio`, structured SVG, editable PowerPoint `.pptx`, TikZ, and reproducible plotting scripts. A bitmap preview is not a substitute for an editable source.

When the optional [Draw.io Scientific Illustrator](https://github.com/icebird1998/drawio-scientific-illustrator) integration is installed and callable, the Stage 2 guide routes suitable figures through its live draw.io workflow and preserves the `.drawio` source. The integration's MIT license covers its code; it does not make third-party reference figures free of copyright restrictions.

Use [`assets/templates/figure-plan.md`](assets/templates/figure-plan.md) to define figure meaning, visual encoding, source format, manuscript placement, and validation before drawing.

## Official LaTeX templates

The official archive registry is [`assets/latex-templates/sources.json`](assets/latex-templates/sources.json). Downloaded archives stay in a gitignored local cache because the two publishers expose different redistribution terms.

Unless the user explicitly selects another journal template, every Chinese-language paper uses the registered 《控制理论与应用》 template. English journal papers use the registered IEEE journal template by default when no more specific target template has been selected. Both registered manuscript bodies are two-column; the Chinese template uses a one-column front-matter context before entering its two-column body.

```powershell
python -X utf8 scripts/fetch_latex_templates.py
python -X utf8 scripts/create_latex_project.py --template ieee-journal --destination <project-directory>
python -X utf8 scripts/create_latex_project.py --template control-theory-and-applications --destination <project-directory>
python -X utf8 scripts/audit_latex_template.py <project-directory>
```

Initialize only the active journal. Write manuscript content in the copied main file and leave classes, headers, packages, bibliography styles, geometry, fonts, and other template controls unchanged. The initializer preserves source bytes and records a `TEMPLATE_LOCK.json` integrity baseline.

## Validation and forward tests

Run the repository validator after every structural or rule update:

```powershell
python -X utf8 scripts/validate_skill.py .
```

Stage 2 includes separate helpers for LaTeX source audits, editable-figure audits, compilation, and rendered-page review:

```powershell
python -X utf8 scripts/audit_manuscript.py <project-root>
python -X utf8 scripts/audit_figures.py <figure-root>
python -X utf8 scripts/compile_manuscript.py <main-tex>
python -X utf8 scripts/audit_latex_template.py <project-root>
```

[`tests/forward-tests.json`](tests/forward-tests.json) contains regression scenarios for the three repeated Stage 1 gates, editable-vector delivery, official-template preservation, and cross-case transfer boundaries. The fixture records prompts, required behavior, and forbidden behavior; an evaluator should run the skill on each prompt and score the response against those fields.

## Versioning and release

Use semantic versioning:

- increment `MAJOR` for incompatible workflow, routing, or rule-ownership changes;
- increment `MINOR` for backward-compatible domain guidance, templates, tools, or capabilities;
- increment `PATCH` for corrections that do not change intended behavior.

Before a release, run the repository validator and relevant forward tests, update [`VERSION`](VERSION), commit the reviewed source state, create the matching Git tag such as `v3.0.0`, and publish that exact commit. Creating a tag or GitHub release is a separate external action and should be performed only when explicitly requested.
