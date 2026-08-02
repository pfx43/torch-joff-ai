# Manuscript Context

This is the sole Stage 2 conception artifact and the paper's source of truth.
Record decisions, dependencies, and acceptance criteria—not manuscript-ready
paragraphs, experimental-result prose, or polished figures.

## Contents

- Context identity and control
- Background, task, and Stage 1 evidence
- Problem, contribution, and technical story
- Theory, model-definition, and optimization order
- Notation and terminology registries
- Chapter and narrative blueprints
- Stage 3 schematic and Stage 4 handoff requirements
- Conception gates and revision history

## Context identity and control

- Context status: DRAFT_CONTEXT
- Baseline ID: <!-- short paper-lineage slug; normally 1–3 terms, prefer <=16 and require <=24 characters -->
- Context revision: 0
- Revision folder: <!-- `<Baseline ID>-r<Context revision>`; this file remains `MANUSCRIPT_CONTEXT.md` inside it -->
- Frozen snapshot: <!-- `<Baseline ID, revision>` after freeze -->
- Frozen date:
- Supersedes revision:
- Working title:
- Target journal:
- Manuscript language:
- Paper type:
- Research domain:
- Stage 1 idea assessment:
- Active manuscript case:
- Selected prose exemplar cases:
- Main manuscript source: <!-- `manuscript.tex` after the first freeze -->

Use revision `1` for the first `FROZEN` snapshot and increase it after every
refreeze. A genuinely different central paper task receives a new baseline ID.

## Background and task baseline

- Application or scientific setting:
- Concrete failure, limitation, or unmet requirement:
- Why the limitation matters:
- Existing method families and their exact insufficiency here:
- Exact task solved:
- Adjacent tasks explicitly excluded:
- One-sentence paper-level question:

## Stage 1 evidence boundary

- Closest primary literature and search date:
- Exact overlap:
- Defensible novelty delta:
- Confirmed assumptions:
- Measured and known quantities:
- Learned, estimated, or bounded quantities:
- Unavailable or counterfactual quantities:
- Confirmed realizability route:
- Claims remaining exploratory, alternative, rejected, or unresolved:

## Problem–contribution–result alignment

Use two or three principal rows and retain the same order throughout.

| Problem ID | Central subproblem | Contribution ID | Problem-facing construction | Nontrivial result/capability | Body location | Stage 4 evidence need | Status |
|---|---|---|---|---|---|---|---|
| P1 |  | C1 |  |  |  |  | BLOCKED |
| P2 |  | C2 |  |  |  |  | BLOCKED |

## Technical main line and causal story

- One-sentence retellable main line:
- Overall theoretical/model mapping before decomposition:

| Step | Problem, limitation, condition, or purpose | Responsive construction | Mechanism or changed quantity | Derived consequence | Task role | Dependency on next step | Status |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  | BLOCKED |

## Theoretical route and dependency

| Result ID | Question resolved | Classification | Prerequisites/assumptions | Exact conclusion | Proof or citation route | Design/task meaning | Scope/limitation | Status |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  | BLOCKED |

## Model-definition order

| Order | Definition level | Object/module | Input | Operation or mapping | Output/interface | Required symbols/dimensions | Why this level precedes the next | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Theoretical object and task |  |  |  |  |  |  | BLOCKED |
| 2 | Overall model mapping |  |  |  |  |  |  | BLOCKED |
| 3 | Forward structure |  |  |  |  |  |  | BLOCKED |
| 4 | Local design detail |  |  |  |  |  |  | BLOCKED |
| 5 | Loss and optimization |  |  |  |  |  |  | BLOCKED |
| 6 | Task application |  |  |  |  |  |  | BLOCKED |

- Fixed mappings and parameters:
- Learned/designable mappings and parameters:
- Initialization and update order:
- Disturbances, uncertainties, faults, and constraints:
- Offline method computations:
- Online method computations:
- Numerical approximations and complexity boundary:

## Loss and optimization baseline

| Objective/constraint | Consumed forward quantity | Target property | Design/learned variables | Fixed/data-determined quantities | Update or solution order | Differentiability/computation note | Status |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  | BLOCKED |

This section defines how the proposed model is optimized. It is not the
experimental training/validation/testing protocol, which belongs to Stage 4.

## Notation registry

Use one row per mathematical object. Every symbol is explained either in the
Introduction-end Notation paragraph or at exact first use.

| Symbol | Semantic family | Meaning | Naming basis / convention | Object type | Dimension | Typography | First definition | Scope |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

Allowed object types include `scalar`, `vector`, `matrix`, `mapping`,
`operator`, `set/space`, `index`, and `constant`. Naming-basis examples:
`field standard: ...`, `journal convention: ...`, `mathematical convention:
...`, `English initial: ...`, `semantic mnemonic: ...`, or
`project-specific: <why no stronger convention is usable>`.

### Reserved and rejected symbol families

| Base family | Reserved meaning | Rejected competing meaning | Resolution |
|---|---|---|---|
|  |  |  |  |

### Notation conflict log

| Date | Collision/mismatch | Affected locations | Resolution | Status |
|---|---|---|---|---|
|  |  |  |  |  |

## Terminology and abbreviation registry

| Concept | Required full term | Abbreviation needed? | Approved abbreviation | First-use route | Field/journal/case basis | Prohibited or obsolete alternatives | Status |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  | BLOCKED |

Do not create an acronym merely because a phrase is long. Verify unfamiliar
terms and record one term per concept.

## Chapter and subsection blueprint

- Skill default non-experimental order: Introduction; Preliminaries and Problem
  Formulation; prerequisite theory; Proposed Method from overall mapping to
  forward structure and loss/optimization; task application; Conclusion draft;
  appendices.
- Actual top-level sequence:
- Target-journal/scientific justification for each deviation:

| Section/subsection | Single primary question | Required input | Reader-facing output | Problem ID | Contribution ID | Depends on | Causal/precedence transition | Conformance/deviation | Status |
|---|---|---|---|---|---|---|---|---|---|
| Introduction | Why is the paper needed and what does it establish? | Stable technical body | Need, gap, response, contributions | P1–P2 | C1–C2 | Stage 3 body | Need precedes response | Conforms | BLOCKED |
| Preliminaries and Problem Formulation | What objects and problems are defined? | Stage 1 information boundary | Standard definitions and numbered problems | P1–P2 | C1–C2 | Introduction need | Need becomes formal task | Conforms | BLOCKED |

## Planned narrative and paragraph progression

| Section/subsection | Planned paragraph sequence | Cause/purpose-before-response progression | Macro-to-micro path | Transition in | Transition out | Supported problem/contribution | Status |
|---|---|---|---|---|---|---|---|
| Introduction | Setting -> exact limitation/task -> literature mechanisms -> shared insufficiency -> questions -> overall response -> contributions -> roadmap |  |  |  |  | P1–P2 / C1–C2 | BLOCKED |

## Stage 3 LaTeX schematic requirements

| Figure ID | Type: principle/model/workflow | Scientific purpose | Required major frames | Required inputs/outputs/relations | Manuscript placement | Required or unnecessary with reason | Status |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  | BLOCKED |

Stage 3 fixes scientific topology with compact LaTeX/TikZ schematics. Stage 4
owns polished figure production.

## Stage 4 evidence and figure handoff

Record what Stage 4 must receive; do not write expected findings as results.

| Claim/property needing evidence | Required experiment output or plot | Required condition/metric metadata | Available source/status | Prohibited inference if evidence is missing | Status |
|---|---|---|---|---|---|
|  |  |  |  |  | BLOCKED |

| Figure ID | Stage 3 schematic source | Stage 4 reference-figure type | Editable final source expected | Palette/reference requirement | Handoff status |
|---|---|---|---|---|---|
|  |  |  |  |  | BLOCKED |

## Style calibration plan

| Task-matched prose case | Permitted calibration target | Selected pattern | Prohibited transfer |
|---|---|---|---|
|  |  |  | Technical claims, novelty, assumptions, citations, or whole paragraphs |

## Stage 2 conception gate record

| Gate | Last run | Evidence inspected | Conflict | Revision action | Affected context sections | Result |
|---|---|---|---|---|---|---|
| C1: background and task |  |  |  |  |  | BLOCKED |
| C2: contribution correspondence |  |  |  |  |  | BLOCKED |
| C3: technical main line |  |  |  |  |  | BLOCKED |
| C4: model-definition order |  |  |  |  |  | BLOCKED |
| C5: notation planning |  |  |  |  |  | BLOCKED |
| C6: terminology planning |  |  |  |  |  | BLOCKED |
| C7: chapter precedence |  |  |  |  |  | BLOCKED |
| C8: narrative blueprint |  |  |  |  |  | BLOCKED |
| C9: cross-check and freeze |  |  |  |  |  | BLOCKED |

## Context revision and invalidation log

| Date | Baseline ID | Revision | Status | Change | Reason/evidence | Invalidated downstream locations | Resolution |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

## Nonblocking unresolved items

- <!-- These cannot appear as completed manuscript or experimental claims. -->
