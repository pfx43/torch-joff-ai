# Source-rule coverage and reconciliation

This file records how the recoverable first version of `ieee-english-paper-polish` and the supplied web-conversation inventory are represented in the current skill. It is a provenance and coverage index, not a second editable copy of the operational rules.

The standalone legacy skill was removed after this migration was verified. Its
name remains below only where needed to identify the historical source; it is
not an active skill entry or fallback workflow.

## Contents

- Recoverable sources
- Legacy-version coverage
- Web-conversation coverage
- Reconciled conflicts
- Completeness protocol

## Recoverable sources

- The first Git commit is `8bc7e1d` (`Add IEEE English paper polish skill`).
- Its complete `SKILL.md`, `references/living-user-rules.md`, `references/tfs-reference-patterns.md`, and `references/typical-errors.md` remain recoverable from Git.
- The web-conversation source is retained in full, with its scope boundaries, in `user-writing-requirements-and-preferences.md`.
- The writing-loop work first introduced during the uncommitted 3.2.0 phase is preserved in `manuscript-quality-gates.md`, its two templates, `audit_writing_loops.py`, the Stage 2 route, and the forward tests.
- The later direct user correction dated 2026-07-29 makes chapter conformance plus sentence logic, causal narrative, symbol consistency, formula rigor, model completeness, and training/testing-flow clarity mandatory in every substantive subsection loop.
- The direct user correction dated 2026-07-30 requires every formula symbol to be explained either in the Introduction-end Notation section or at first use, requires a Markdown symbol ledger with a recorded naming basis, prioritizes field and mathematical conventions before meaningful English initials or mnemonics, and makes meaning or object-type conflicts trigger a global rename within the repeated writing loop.

## Legacy-version coverage

| Legacy rule family | Current operational location | Preservation note |
|---|---|---|
| Read the manuscript, bibliography, instructions, references, and active context before editing | `SKILL.md`; `stages/journal-paper-writing-and-figures.md` | Preserved and generalized to the two-stage route |
| Create and maintain `MANUSCRIPT_CONTEXT.md` | `SKILL.md`; `assets/templates/manuscript-context.md`; Stage 2 guide | Preserved with one-paper scope |
| Build the notation ledger and search the entire source before adding symbols | `manuscript-quality-gates.md`; `assets/templates/notation-ledger.md` | Strengthened into a blocking, repeated gate |
| Separate preliminaries from paper-specific contributions | `living-user-rules.md`; Stage 2 guide | Preserved |
| Compact IEEE section hierarchy and mathematical `Problem Formulation` | `living-user-rules.md`; `manuscript-quality-gates.md` | Preserved; accepts `Problem Description` as an equivalent target-journal label |
| Theoretical modeling before learned forward modeling and workflow | `living-user-rules.md`; Stage 2 guide | Preserved |
| Long proofs in appendices and gradients only when contribution-relevant | Stage 2 guide | Preserved |
| Close each mathematical argument with motivation, equation, definitions, dimensions, and consequence | Stage 2 guide; `technical-validity-and-implementation.md` | Preserved and expanded |
| Compile, inspect references and boxes, render, and visually verify | Stage 2 guide; LaTeX workflow | Preserved and expanded |
| Add durable cross-manuscript rules and keep paper-specific decisions local | `living-user-rules.md`; `rule-scope-map.md` | Preserved |
| Keep one canonical checkout and expose other locations through links | `SKILL.md` | Preserved under the renamed current repository |
| Use recurring negative examples before delivery | `typical-errors.md`; `SKILL.md` | Preserved |
| Markdown delimiters, blank lines, en dash, corruption, and placeholder audit | Stage 2 guide; validation scripts | Preserved |
| No displayed equation before Section II | `living-user-rules.md`; writing-loop audit | Preserved |
| First-use definitions and Notation paragraph exception | `living-user-rules.md`; notation gate | Preserved |
| Field/mathematical convention, meaningful symbol choice, and recorded naming basis | `living-user-rules.md`; notation gate; notation template | Added as a blocking repeated check without weakening collision rules |
| One semantic family per base character and one typography per object type | `living-user-rules.md`; notation gate | Preserved and enforced |
| Fault, normal, time, input, dimension, disturbance, weight, mapping, set, and operator families | `living-user-rules.md`; notation template and audit | Preserved |
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
| Never invent experiments, data, hardware, datasets, or citations | `technical-validity-and-implementation.md` | Preserved |
| IEEE/TFS Introduction tension, proof chain, experiments, and terminology | `living-user-rules.md`; TFS patterns; Stage 2 guide | Preserved |
| Reference papers teach structure but do not supply unverified citations or copyable text | TFS patterns; Stage 2 guide | Preserved |
| Deliver source, bibliography, PDF, context consistency, and a substantive-change report | Stage 2 guide | Preserved |

## Web-conversation coverage

| Web section | Complete source location | Operational locations |
|---|---|---|
| I. General writing requirements | `user-writing-requirements-and-preferences.md` | `living-user-rules.md`; `technical-validity-and-implementation.md`; Stage 1 guide |
| II. Paper structure | Same | `living-user-rules.md`; Stage 2 guide; `manuscript-quality-gates.md` |
| III. Chinese and English style | Same | `living-user-rules.md`; Stage 2 guide |
| IV. Formula, layout, and PDF preferences | Same | Stage 2 guide; `latex-template-workflow.md`; target-project context |
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

## Completeness protocol

Before removing or compressing a legacy or user-supplied rule:

- identify its source and scope;
- locate every independent constraint, exception, example, and unknown boundary;
- map each item to one current authoritative location;
- merge only exact semantic overlap;
- preserve project-specific details in a domain guide or case instead of globalizing them;
- document any conflict and the compatibility rule here;
- extend `validate_skill.py` or the forward tests when a future refactor could silently remove the rule family.
