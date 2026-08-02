# Idea Review and Paper Writing

[简体中文](README.md) | [English](README.en.md)

Current version: **5.0.0**

This skill uses four stages: idea review, paper conception, non-experimental
technical writing, and final figure/experiment integration. Every stage presents
its loop as a name-plus-summary table before the detailed rules.

## Contents

- [Four-stage workflow](#four-stage-workflow)
- [Baseline identity](#baseline-identity)
- [Stage 1 loops](#stage-1-loops)
- [Stage 2 loops](#stage-2-loops)
- [Stage 3 loops](#stage-3-loops)
- [Stage 4 loops](#stage-4-loops)
- [Scientific-figure composition](#scientific-figure-composition)
- [Figure-exemplar library](#figure-exemplar-library)
- [File boundaries](#file-boundaries)
- [Validation and versioning](#validation-and-versioning)

## Four-stage workflow

| Stage | Core task | Main output | Explicit exclusion |
|---|---|---|---|
| 1: idea review | Prior art, novelty delta, task/information boundary, assumptions, theory, and realizability | `idea-assessment.md`, paper case | No settled manuscript claims |
| 2: paper conception | Background/task, contributions, technical story, model-definition order, notation, terminology, chapters, and narrative | Sole `MANUSCRIPT_CONTEXT.md` in the revision folder | No final prose, result text, or polished figures |
| 3: manuscript writing | Non-experimental theory/method prose, equations, proofs, algorithms, and LaTeX schematics | `manuscript.tex/.pdf`, `WRITING_LOOP_LOG.md`, LaTeX sketches | No experimental findings or final figure production |
| 4: figures and experiments | Scientific figures, external Python-plot integration, evidence-grounded result prose, and final delivery | Editable figure sources, experiment text, final PDF | No method redesign or fabricated plots |

```text
Stage 1 confirmed idea
  -> Stage 2 FROZEN <Baseline ID, revision>
  -> Stage 3 technical draft + LaTeX schematics
  -> Stage 4 figures + actual-result narration + final delivery
```

## Baseline identity

- `Baseline ID` is a stable readable slug for one paper lineage, such as
  `koopman-fault-diagnosis`. It stays unchanged for the same paper.
- `Context revision` is a positive integer. The first frozen conception is
  revision 1; every refreeze increments it.
- `FROZEN` means that revision passed all Stage 2 checks.
- `<Baseline ID, revision>` identifies the exact conception snapshot used by
  Stage 3 and Stage 4.
- Keep the Baseline ID short and recognizable, normally one to three terms,
  preferably at most 16 characters and never more than 24. Name the revision folder `<baseline-id>-r<revision>`,
  for example `koop-fd-r3/`.
- A genuinely different central paper task receives a new baseline ID and paper
  case rather than another revision of the old paper.

Use that stem in the main `.tex`/`.pdf`, context, Stage 3/4 logs, figures,
plots, and submission archive. See
[artifact-naming.md](references/artifact-naming.md).

## Stage 1 loops

Detailed rules: [idea-exploration.md](references/stages/idea-exploration.md)

| ID | Short name | Summary |
|---|---|---|
| I1 | Prior-art overlap | Determine whether closest work already establishes the same construction, mechanism, or result |
| I2 | Task boundary | Fix what is and is not solved and which information is truly available |
| I3 | Realizable assumptions | Close computation, optimization, online use, and evidence feasibility |
| I4 | Nontrivial theory | Require a structure-dependent computable, provable, or testable consequence |
| I5 | Focused contributions | Retain two or three connected problem-facing themes |
| I6 | Honest maturity | Separate confirmed, exploratory, alternative, rejected, and unresolved routes |

## Stage 2 loops

All Stage 2 rules act only on `MANUSCRIPT_CONTEXT.md` in the current revision
folder. Detailed rules:
[paper-conception.md](references/stages/paper-conception.md)

| ID | Short name | Summary |
|---|---|---|
| C1 | Background and task | Establish why the work is needed before stating its exact task |
| C2 | Contribution mapping | Align every problem, principal contribution, result, and evidence need |
| C3 | Technical main line | Connect limitation, need, construction, mechanism, consequence, and task role |
| C4 | Model-definition order | Theory to design, global to local, forward structure to loss, design to application |
| C5 | Notation planning | Register meaning, type, font, dimension, and first-definition route |
| C6 | Terminology planning | Use established terms and only necessary first-defined abbreviations |
| C7 | Chapter precedence | Order sections by scientific dependency rather than module count |
| C8 | Narrative blueprint | Plan paragraph purpose, causality, and macro-to-micro progression |
| C9 | Cross-check and freeze | Freeze only after positioning, technical, notation, term, structure, and narrative passes |

```text
DRAFT_CONTEXT -> CHECK -> REVISE -> CHECK -> FROZEN
```

## Stage 3 loops

Detailed rules: [manuscript-writing-loop.md](references/manuscript-writing-loop.md)

| ID | Short name | Summary |
|---|---|---|
| W1 | Baseline consistency | Do not silently change the frozen conception |
| W2 | Chapter precedence | Theory to model, global to local, forward to loss, design to task use |
| W3 | Narrative logic | Use real sentence/paragraph relations and purpose/cause before response/result |
| W4 | Avoid rigid prose | Replace declarative stacking with meaning-driven sentence variation |
| W5 | Avoid empty prose | Give every sentence and paragraph an indispensable argumentative function |
| W6 | Symbol consistency | First explanation, one symbol per meaning, one font per object class |
| W7 | Specialized terminology | Established terms, necessary abbreviations, first-use full forms, no coined jargon |
| W8 | Formula rigor | Complete definitions, dimensions, indices, conditions, derivation, and scope |
| W9 | Schematic completeness | Provide required principle/model/workflow LaTeX schematics |
| W10 | Manuscript alignment | Keep problems, contributions, body, schematics, abstract, and conclusion synchronized |

Model-description completeness and train/validation/test/deployment clarity are
not Stage 3 loop labels. Model order is fixed in Stage 2 and enforced by W2/W8;
experimental workflow and results belong to Stage 4.

## Stage 4 loops

Detailed rules:
[figures-and-experiments.md](references/stages/figures-and-experiments.md)

| ID | Short name | Summary |
|---|---|---|
| F1 | Evidence ready | Require stable schematics and actual results before final production/narration |
| F2 | Figure type matched | Distinguish principle, model-structure, task-workflow, and quantitative plots |
| F3 | Frames within frames | Use titled major frames and complete but concise contained details |
| F4 | Directed acyclic arrows | Connect shape to shape with clear one-way, non-dangling, visually acyclic arrows |
| F5 | Academic visual language | Use restrained editable shapes/icons, not poster or business graphics |
| F6 | Controlled density | Keep module meaning complete while limiting text and formulas |
| F7 | Symbol synchronization | Match context and manuscript glyphs, terms, and typography exactly |
| F8 | Reference provenance | Search top-tier primary papers and record source, transfer boundary, and rights |
| F9 | Readable palette | Use one cool/neutral strip and one semantically stable accent |
| E1 | Plot–result consistency | Consume reviewed plots from the external Python library; never hand-edit data |
| E2 | Evidence-grounded narration | State tested claim, observation, comparison, interpretation, and limitation |
| E3 | Restrained conclusion | Do not convert trends into proof, causality, significance, or universality |
| D1 | Figure–caption–text closure | Synchronize figures, captions, body, abstract, and conclusion |

## Scientific-figure composition

Detailed rules:
[figure-composition-rules.md](references/figure-composition-rules.md)

Universal requirements:

- no whole-figure title inside the graphic;
- every major frame has a concise title;
- arrows run shape-to-shape, one way, without dangling endpoints or visible
  cycles, although they may cross major-frame boundaries;
- text, formulas, symbols, and icons remain restrained and academic;
- local zoom-ins and repeated-block stacks are permitted when scientifically
  necessary;
- genuine recurrence is time/iteration-unrolled rather than falsified or drawn
  as a closed visual loop.

A model diagram uses major frames for overall/loss modules and contained shapes
for internal design and interfaces. A workflow uses major frames for steps such
as acquisition/preprocessing, offline modeling, and online monitoring, with a
one-way chain inside each step. Principle schematics have no fixed template and
require current related-paper search.

Quantitative plots are produced by a dedicated external Python library. This
skill reviews their exports and evidence metadata and writes the manuscript
text; it does not recreate or manually modify the plotting system.

## Figure-exemplar library

Start at
[cases/figure-exemplars/README.md](cases/figure-exemplars/README.md):

```text
cases/figure-exemplars/
  principle-diagrams/
    references.md
    images/
  model-structure-diagrams/
    references.md
    images/
  task-workflow-diagrams/
    references.md
    images/
```

The three figure types are the case boundaries. Add each reference figure as
one inventory row inside its category; do not create one directory per figure.
Store an image only when user-provided, openly licensed,
publisher-authorized, or otherwise lawful.

Palette visuals stay in [`assets/palettes/`](assets/palettes/), while
[`references/scientific-figure-palettes.md`](references/scientific-figure-palettes.md)
holds authoritative codes and semantic roles.

Licensed academic icons live in [`assets/icons/`](assets/icons/README.md).
Alibaba Iconfont and EmojiAll are candidate search sources, but every exact
asset, author, license, and publication right must be checked and registered.
Use the Codex Chrome browser plugin when dynamic or signed-in interactive
selection/download is required.

## File boundaries

| Resource | Sole responsibility | Read when |
|---|---|---|
| `SKILL.md` | Four-stage routing | Every trigger |
| `references/stages/idea-exploration.md` | Stage 1 loops | Idea/theory/feasibility review |
| `references/stages/paper-conception.md` | Stage 2 loops | Creating or changing context |
| `references/stages/manuscript-writing.md` | Stage 3 execution | Non-experimental technical writing |
| `references/manuscript-writing-loop.md` | W1–W10 | Every substantive Stage 3 edit |
| `references/stages/figures-and-experiments.md` | Stage 4 loops | Figures, actual results, final integration |
| `references/figure-composition-rules.md` | Detailed conceptual-figure construction | Stage 4 conceptual figures |
| `references/artifact-naming.md` | Short-ID revision-folder and internal-role naming contract | Stages 2–4 |
| `references/detail-preservation-and-refactoring.md` | No-detail-loss refactoring and migration evidence | Every skill-maintenance pass |
| `<short-baseline-id>-r<revision>/MANUSCRIPT_CONTEXT.md` | Sole conception baseline | Stage 2 creates; Stage 3/4 read |
| `<revision-folder>/WRITING_LOOP_LOG.md` | Stage 3 evidence | Stage 3 |
| `<revision-folder>/STAGE4_FIGURE_EXPERIMENT_LOG.md` | Stage 4 evidence | Stage 4 |
| `cases/<task-group>/` | One-paper history or prose exemplar | Exact manuscript task match |
| `cases/figure-exemplars/<one-of-three-types>/` | Category case and multi-reference inventory | Exact figure-type match |
| `assets/icons/` | Licensed reusable icons and registry | Stage 4 figure work |

## Validation and versioning

```powershell
python -X utf8 scripts/validate_skill.py .
python -X utf8 scripts/audit_artifact_names.py <revision-folder>
python -X utf8 scripts/audit_writing_loops.py <project-root>
python -X utf8 scripts/audit_manuscript.py <project-root>
python -X utf8 scripts/audit_figures.py <figure-root>
```

`tests/forward-tests.json` covers four-stage entry conditions, the ten Stage 3
loops, Stage 4 figure types and evidence boundaries, symbol/term conflicts, and
case transfer.

`VERSION` is the semantic-version source. Version 7.0.0 puts the short Baseline
ID/revision in the revision-folder name, keeps concise internal role filenames,
and adds a no-detail-loss contract for skill-creator refactors. Commit, tag, or
push only when the user explicitly requests publication.
