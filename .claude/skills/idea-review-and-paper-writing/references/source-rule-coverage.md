# Source-rule coverage and reconciliation

This file records how the recoverable first version of `ieee-english-paper-polish` and the supplied web-conversation inventory are represented in the current skill. It is a provenance and coverage index, not a second editable copy of the operational rules.

## Contents

- Recoverable sources
- Legacy-version coverage
- Web-conversation coverage
- Reconciled conflicts
- Structural migration ledger
- Completeness protocol

## Recoverable sources

- The first Git commit is `8bc7e1d` (`Add IEEE English paper polish skill`).
- Its complete `SKILL.md`, `references/living-user-rules.md`, `references/tfs-reference-patterns.md`, and `references/typical-errors.md` remain recoverable from Git.
- The web-conversation source is retained in full, with its scope boundaries, in `user-writing-requirements-and-preferences.md`.
- The writing-loop work first introduced during the uncommitted 3.2.0 phase is preserved in `manuscript-writing-loop.md`, `writing-loop-log.md`, `audit_writing_loops.py`, the Stage 3 route, and the forward tests.
- The later direct user correction dated 2026-07-29 made chapter conformance plus sentence logic, causal narrative, symbol consistency, formula rigor, model description, and training/testing-flow clarity mandatory checks. The newest 2026-08-01 correction retains their underlying scientific requirements but removes `model completeness` and `training/validation/testing/deployment clarity` as Stage 3 loop titles: model order is handled in Stage 2 and Stage 3 W2/W8, while experimental workflow and result prose move to Stage 4.
- The direct user correction dated 2026-07-30 requires every formula symbol to be explained either in the Introduction-end Notation section or at first use, requires a Markdown symbol-registry table with a recorded naming basis, prioritizes field and mathematical conventions before meaningful English initials or mnemonics, and makes meaning or object-type conflicts trigger a global rename within the repeated writing loop. The registry now resides inside the sole Stage 2 context rather than a separate ledger file.
- The first direct user correction dated 2026-08-01 makes sentence-form naturalness and variation, cause/purpose-before-result progression, macro-to-micro exposition, and one identifiable purpose for every paragraph mandatory repeated Stage 3 checks. It also requests curated, concrete architecture, narrative, phrase, sentence-form, and terminology examples from the user's published papers rather than generic advice.
- A later direct user correction dated 2026-08-01 first replaced the former two-stage workflow with Stage 1 idea review, Stage 2 paper conception, and Stage 3 manuscript writing; made `MANUSCRIPT_CONTEXT.md` the sole Stage 2 conception baseline; and moved concrete published-paper style examples into one-paper files under the task-grouped case library.
- The newest direct correction dated 2026-08-01 establishes four stages by adding Stage 4 figures and experiments; defines `Baseline ID` as a stable paper-lineage slug and `Context revision` as the incrementing frozen version; requires every stage loop to begin with a name-plus-summary table; replaces the Stage 3 loop with baseline consistency, chapter precedence, narrative logic, non-rigid/nonempty prose, symbol consistency, specialized terminology, formula rigor, schematic completeness, and manuscript alignment; requires Stage 3 LaTeX schematics; moves polished figures and experimental prose to Stage 4; sets detailed model/workflow/principle figure composition rules; and creates a rights-aware, figure-type-grouped reference-case library while retaining palettes under `assets/palettes/`.
- A subsequent direct correction temporarily required `<baseline-id>-r<revision>` to appear in TeX, PDF, context/log, figure/plot, and delivery filenames; accepted Alibaba Iconfont and EmojiAll as candidate icon-search sources subject to per-asset license verification and Chrome-assisted interactive download when needed; corrected the figure-case boundary from one case per reference figure to exactly three category-level cases with multi-reference inventories; and required an explicit whole-skill file-boundary audit.
- The latest naming correction keeps Baseline ID short and moves `<baseline-id>-r<revision>` to the revision-folder name. Files inside use stable role names such as `MANUSCRIPT_CONTEXT.md`, `manuscript.tex`, `manuscript.pdf`, and the two loop logs; figure and plot names remain short inside their dedicated subfolders.
- The latest maintenance correction requires `skill-creator` to be used only for review, boundary assignment, exact-semantic deduplication, validation, and progressive disclosure. No user-supplied detail may be removed for concision; every rule, exception, example, project-specific detail, and explicit unknown boundary must be classified, preserved, routed, and protected by coverage evidence or tests.

## Legacy-version coverage

| Legacy rule family | Current operational location | Preservation note |
|---|---|---|
| Read the manuscript, bibliography, instructions, references, and active context before editing | `SKILL.md`; `stages/manuscript-writing.md` | Preserved under selective Stage 3 loading |
| Create and maintain `MANUSCRIPT_CONTEXT.md` | `SKILL.md`; `assets/templates/manuscript-context.md`; `stages/paper-conception.md` | Strengthened as the sole frozen Stage 2 conception baseline |
| Build the notation registry and search the entire source before adding symbols | Context template and Stage 2 C5; `manuscript-writing-loop.md` | Preserved inside the single context source of truth and repeated in Stage 3 |
| Separate preliminaries from paper-specific contributions | `living-user-rules.md`; Stage 2 conception and Stage 3 writing guides | Preserved |
| Compact IEEE section hierarchy and mathematical `Problem Formulation` | `living-user-rules.md`; `paper-conception.md`; `manuscript-writing-loop.md` | Preserved; accepts `Problem Description` as an equivalent target-journal label |
| Theoretical modeling before learned forward modeling and workflow | `living-user-rules.md`; Stage 2 blueprint; Stage 3 writing guide | Preserved |
| Long proofs in appendices and gradients only when contribution-relevant | Stage 3 guide | Preserved |
| Close each mathematical argument with motivation, equation, definitions, dimensions, and consequence | Stage 3 guide; `technical-validity-and-implementation.md` | Preserved and expanded |
| Compile, inspect references and boxes, render, and visually verify | Stage 3 technical draft; Stage 4 final delivery; LaTeX workflow | Preserved and expanded |
| Add durable cross-manuscript rules and keep paper-specific decisions local | `living-user-rules.md`; `rule-scope-map.md` | Preserved |
| Keep one canonical checkout and expose other locations through links | `SKILL.md` | Preserved under the renamed current repository |
| Use recurring negative examples before delivery | `typical-errors.md`; `SKILL.md` | Preserved |
| Markdown delimiters, blank lines, en dash, corruption, and placeholder audit | Stage 3 guide; validation scripts | Preserved |
| No displayed equation before Section II | `living-user-rules.md`; writing-loop audit | Preserved |
| First-use definitions and Notation paragraph exception | `living-user-rules.md`; notation gate | Preserved |
| Field/mathematical convention, meaningful symbol choice, and recorded naming basis | `living-user-rules.md`; Stage 2 context notation gate and template | Added as a blocking repeated check without weakening collision rules |
| One semantic family per base character and one typography per object type | `living-user-rules.md`; notation gate | Preserved and enforced |
| Fault, normal, time, input, dimension, disturbance, weight, mapping, set, and operator families | `living-user-rules.md`; context notation table and audit | Preserved |
| Minimize similar Greek variants | `living-user-rules.md` | Preserved |
| Blackboard bold only for sets and spaces | `living-user-rules.md` | Preserved |
| Explicit stacks rather than `\operatorname{col}` | `living-user-rules.md`; `typical-errors.md` | Preserved |
| Distinct fault family and separate physical channels | `living-user-rules.md`; `domains/fault-diagnosis.md` | Preserved |
| History, Hilbert-space, output-map, Koopman, and attention-key collisions | `living-user-rules.md`; notation gate | Preserved with the conflict resolution below |
| One number for related multi-line equations | `living-user-rules.md` | Preserved |
| Controlled Koopman operator and distinct finite predictor | `domains/fault-diagnosis.md`; TFS patterns | Preserved |
| T–S membership conditions and membership provenance | `domains/fault-diagnosis.md` | Preserved and expanded |
| Local-rule stability does not imply arbitrary interpolated stability | `technical-validity-and-implementation.md` | Preserved |
| Differentiable strict norm certificate and actual gradient | `technical-validity-and-implementation.md` | Preserved |
| Frobenius upper bound is not exact spectral normalization | `technical-validity-and-implementation.md` | Preserved |
| Discrete sliding-mode lower and upper gains, reachability, and invariance | `domains/fault-diagnosis.md` | Preserved |
| One-step and multi-step actuator/sensor signature conditions | `domains/fault-diagnosis.md` | Preserved and broadened to process faults |
| Ding-style dynamic threshold is not a rolling empirical threshold | `domains/fault-diagnosis.md`; TFS patterns | Preserved and expanded |
| Never invent experiments, data, hardware, datasets, figures, or citations | `technical-validity-and-implementation.md`; Stage 4 guide | Preserved and expanded |
| IEEE/TFS Introduction tension, proof chain, experiments, and terminology | `living-user-rules.md`; TFS patterns; Stage 2 conception, Stage 3 writing, and Stage 4 experiment guides | Preserved |
| Reference papers teach structure but do not supply unverified citations or copyable text | TFS patterns; Stage 2 style-calibration plan; Stage 3 writing guide | Preserved |
| Published papers provide curated, source-tagged style calibration without becoming technical authority | One-paper published-style cases under `cases/<task-group>/`; cross-case routing under `style-exemplars/` | Added with task grouping, rules-versus-examples boundary, and quality filtering |
| Deliver source, bibliography, PDF, context consistency, and a substantive-change report | Stage 3 technical handoff and Stage 4 final delivery | Preserved |

## Web-conversation coverage

| Web section | Complete source location | Operational locations |
|---|---|---|
| I. General writing requirements | `user-writing-requirements-and-preferences.md` | `living-user-rules.md`; `technical-validity-and-implementation.md`; Stage 1 guide |
| II. Paper structure | Same | `living-user-rules.md`; Stage 2 conception; Stage 3 writing loop |
| III. Chinese and English style | Same | `living-user-rules.md`; Stage 3 guide and loop |
| IV. Formula, layout, and PDF preferences | Same | Stage 3/4 guides; `latex-template-workflow.md`; target-project context |
| V. Quality-prediction nonlinear observer | Same | `domains/soft-sensing-and-observers.md`; matching soft-sensing case |
| VI. Koopman–T–S–attention diagnosis | Same | `domains/fault-diagnosis.md`; matching fault-diagnosis case; TFS patterns |
| VII. Missing-data-completion survey | Same | Complete source only until a matching domain or case is created |
| VIII. Review, overlap checking, and scoring | Same | Complete source plus Stage 1 novelty audit and technical-validity audit |
| IX. Journal and conference reports | Same | Complete source; live source verification required |
| X. Admissions and research reports | Same | Complete source; live source verification required |
| XI. Image, video, and game prompt writing | Same | Complete source; use only for a matching prompt task |
| XII. General prose and email style | Same | `living-user-rules.md` for confirmed cross-task style; complete source for email evidence limits |
| XIII. Obsidian workflow | Same | Complete source only; not an academic-manuscript default |
| XIV. Global versus project-specific scope | Same | `rule-scope-map.md`; domain guides; case files |
| XV. Explicitly unknown preferences | Same | Complete source; these items must not be inferred as fixed defaults |

## Reconciled conflicts

### Core themes versus concrete contribution claims

The web inventory and later operational rule use two or three principal contribution themes. The first skill version also required three to five concrete contribution claims in the Introduction. Both details are retained by separating levels:

- retain two or three principal problem-facing themes and align them with the central subproblems;
- when necessary, express three to five concrete, verifiable construction, analysis, or validation claims under those themes;
- do not misrepresent subordinate claims as unrelated main lines.

### `H` for history and for a Hilbert space

The first version reserved `H` for neural or history objects and also gave `\mathbb H^{\infty}` as an example for an infinite-dimensional Hilbert space. The semantic-family rule does not permit those unrelated meanings to coexist merely through font changes. The current rule preserves both intentions:

- reserve one registered base character for the history or neural object;
- denote the infinite-dimensional Hilbert space explicitly;
- if `H` is already occupied, use another registered space character instead of reusing `H`.

### Historical canonical path

The first version named `D:\AI\cc-switch\skills\ieee-english-paper-polish` as the canonical path. The repository was subsequently renamed and now lives at `C:\Users\Fuzz4\.cc-switch\skills\idea-review-and-paper-writing`. The non-duplication rule is preserved, while the obsolete literal path is retained here only as migration provenance and is not an active editing target.

The first TFS reference note also mentioned the environment-specific library `D:\_[PPPaper]\TFS_ref`. Its reusable role is preserved in `tfs-reference-patterns.md`: when that library or another extracted roadmap is actually supplied, use it for structural and proof-pattern guidance, verify primary sources and BibTeX records, and never copy its prose or equations. The historical literal path is not assumed to exist in a different environment.

### Exact section labels and target templates

The first version required `Problem Formulation`; the current rule also accepts `Problem Description` when it is the appropriate target-journal label. The mathematical requirements, central-subproblem count, and prohibition on a prose `Monitoring Objectives` checklist remain unchanged.

The web inventory records a stable `ctexart` A4 two-column profile for a specific Chinese-paper project. It remains available as a project-scoped preference and does not override an official publisher template or become a universal Chinese-journal default.

## Structural migration ledger

The following containers were retired or split during the four-stage
reorganization. Their independent details remain owned as follows:

| Retired/split container | Preserved detail families | Current authoritative owner |
|---|---|---|
| `assets/templates/notation-ledger.md` | one row per object; semantic family; meaning; naming basis; object type; dimension; typography; first-definition location; scope; reserved/rejected families; conflict log | notation, reserved-family, and conflict sections of `assets/templates/manuscript-context.md`; Stage 2 C5; Stage 3 W6; `audit_writing_loops.py` |
| `assets/templates/section-role-matrix.md` | default chapter sequence; actual-sequence deviation; problem/contribution alignment; one question/output per section; causal transition; evidence-bearing subsection and gate records | chapter, contribution, and narrative sections of `assets/templates/manuscript-context.md`; paragraph/subsection records in `assets/templates/writing-loop-log.md`; C2/C7/C8 and W1–W10 |
| `references/manuscript-quality-gates.md` | initialization, section responsibility, pre-draft notation, repeated subsection checks, abstract/problem/contribution alignment, whole-manuscript audit, recovery, and definition of done | `stages/paper-conception.md`; `manuscript-writing-loop.md`; `stages/figures-and-experiments.md`; both loop-log templates; deterministic audit scripts |
| `references/stages/journal-paper-writing-and-figures.md` | entry conditions; manuscript construction; Markdown rules; figure planning/routing; diagram, quantitative-plot, visual, caption, copyright, reproducibility, template, PDF, and delivery requirements | four stage guides; `figure-composition-rules.md`; `latex-template-workflow.md`; `scientific-figure-palettes.md`; `technical-validity-and-implementation.md` |
| old writing-loop fixture files | positive notation/chapter/logic/formula/schematic evidence and negative object-type/project-specific-rationale cases | `tests/fixtures/writing-loop-fixture-r2/` using the current revision-folder contract |

The former `model completeness` and
`training/validation/testing/deployment clarity` labels were removed only as
Stage 3 loop titles by a later direct user correction. Their substantive
requirements remain distributed across Stage 2 model-definition order, Stage 3
W2/W8, technical-validity rules, and Stage 4 evidence/experiment checks.

## Completeness protocol

Before removing or compressing a legacy or user-supplied rule:

- identify its source and scope;
- locate every independent constraint, exception, example, and unknown boundary;
- map each item to one current authoritative location;
- merge only exact semantic overlap;
- preserve project-specific details in a domain guide or case instead of globalizing them;
- document any conflict and the compatibility rule here;
- extend `validate_skill.py` or the forward tests when a future refactor could silently remove the rule family.

The operational nondeletion and refactoring procedure is defined in
`detail-preservation-and-refactoring.md`. This coverage index records outcomes
and provenance; it does not replace that maintenance contract.
