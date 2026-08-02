# Stage 3: Manuscript writing

Turn one frozen Stage 2 conception into a coherent non-experimental technical
draft. Own prose, equations, citations, formal environments, LaTeX, proofs,
method algorithms, and compact LaTeX figure schematics. Defer polished figures,
quantitative plots, experimental-result prose, and result-dependent finalization
to Stage 4.

## Contents

- Entry and selective inputs
- Recommended writing order
- Technical construction
- LaTeX schematic handoff
- Prose and formal writing
- LaTeX/PDF checks
- Baseline deviations
- Stage output

## Entry and selective inputs

Require `MANUSCRIPT_CONTEXT.md` in the active revision folder to be `FROZEN`
with a short stable baseline ID, positive revision, matching folder name, complete notation and
terminology registries, chapter plan, model-definition order, and narrative
blueprint. Read `references/artifact-naming.md`. Name the main source
`manuscript.tex`, record `% Frozen snapshot: <Baseline ID, revision>` near its
first line, and create `WRITING_LOOP_LOG.md` in the same revision folder.

Read the complete current source and bibliography, frozen context, exact paper
case, and only the prose exemplar selected in the context. Read technical or
domain guidance only for the subsection being written. Do not reload Stage 1,
Stage 2, figure cases, or experimental guidance during ordinary prose work.

## Recommended writing order

The final paper order and drafting order differ. Draft in this order unless a
scientific dependency or target template requires another route:

1. initialize the official LaTeX project and bind the frozen baseline;

2. write Preliminaries and Problem Formulation;

3. write prerequisite theoretical analysis before model design;

4. define the overall model mapping;

5. write the forward structure from main modules to local details;

6. write loss functions, constraints, optimization variables, update rules,
   algorithms, complexity, and offline/online method execution;

7. write the declared task application or genuinely independent second task;

8. create compact LaTeX schematics for the required principle, model structure,
   and task workflow figures;

9. write the Introduction from the stable technical body and context;

10. write a scoped Conclusion draft and result-neutral title/abstract/keywords,
    leaving experimental findings and comparisons for Stage 4;

11. synchronize contributions, roadmap, notation paragraph, terminology,
    equations, algorithms, captions/placeholders, and appendices.

Do not draft the Experiments section or invent result language in Stage 3.

## Technical construction

Preserve a submission-shaped non-experimental body: problem formulation,
necessary background, theory, overall method definition, forward computation,
local mechanisms, losses and optimization, task use, complexity, implementation
boundary, formal results and proofs, and references. A model name or layer list
does not replace governing relations, dimensions, interfaces, or update order.

Close each mathematical argument with motivation, a numbered relation,
immediate symbol/dimension definitions, and a stated consequence. Introduce a
formal result by naming the question it resolves; follow it with design,
implementation, scope, or limitation meaning. Do not enlarge a local,
conditional, one-step, or empirical conclusion.

Keep theoretical prerequisites before model design, overall mapping before
local modules, forward structure before loss/optimization, and model design
before task application. Put long proofs in appendices and include gradient
formulas only when the optimization derivation is contribution-relevant.

## LaTeX schematic handoff

For each figure need recorded in the context, determine whether the paper needs:

- a **principle schematic** explaining a mechanism or relation;
- a **model structure diagram** showing the overall map, modules, losses,
  inputs, and outputs;
- a **task workflow diagram** showing steps such as data acquisition and
  preprocessing, offline modeling, and online monitoring/use.

Create a compact compileable TikZ/LaTeX schematic or an equivalent LaTeX-coded
placeholder that fixes scientific topology, labels, symbols, and major groups.
It need not have final icons, spacing, or visual polish. When a category is not
needed, record the scientific reason rather than drawing a decorative figure.

Stage 3 schematics must already use manuscript symbols, have no whole-image
title inside the graphic, title every major group, and avoid invented arrows.
Stage 4 uses them as topology authority and visual-production input.

## Prose and formal writing

Give every paragraph one necessary purpose. Put cause, condition, limitation,
or objective before the response/result in ordinary exposition. Move from
paper-level question and overall mapping toward local equations, optimization,
and task consequences.

Vary active/passive voice, clauses, nonfinite constructions, coordination,
prepositional openers, and adverbials only when they express real relations. Do
not replace declarative stacking with decorative long sentences, forced passive
voice, generic modifiers, or transitions without antecedents.

Use established field terminology. Introduce only useful abbreviations, give
the full term at first use, keep one term per concept, and verify unfamiliar
terms through authoritative literature or the task-matched case. Published
examples calibrate rhetorical structure; they do not supply technical truth.

Execute every Stage 3 check in
`references/manuscript-writing-loop.md` after each substantive subsection
revision and record evidence rather than bare `PASS` values.

## LaTeX/PDF checks

Apply `references/latex-template-workflow.md`. Preserve official classes,
packages, geometry, fonts, bibliography style, and template controls. Compile,
resolve undefined references and overfull boxes, and render representative or
all pages to check equations, fonts, clipping, whitespace, searchable text,
schematic labels, captions/placeholders, and cross-references.

Run:

```powershell
python -X utf8 scripts/audit_artifact_names.py <revision-folder>
python -X utf8 scripts/audit_writing_loops.py <project-root>
python -X utf8 scripts/audit_manuscript.py <project-root>
python -X utf8 scripts/compile_manuscript.py manuscript.tex
```

The compiler must produce `manuscript.pdf`. A journal-mandated fixed entry
filename may exist only as a documented compatibility wrapper or temporary
upload copy, not as the canonical source/PDF delivered by the skill.

## Baseline deviations

If drafting requires a different task, contribution, theoretical route, model
order, core symbol/term, chapter responsibility, or causal dependency, stop the
affected prose and return to Stage 2. A new novelty, assumption, proof, or
feasibility issue returns first to Stage 1. After refreeze, invalidate and rerun
all dependent Stage 3 checks.

## Stage output

Stage 3 produces a context-aligned non-experimental technical manuscript,
consistent notation and terminology, checked formal results, a completed
versioned writing log, a canonically named `.tex`/`.pdf` pair, and compileable
versioned LaTeX schematics for required figures. Stage 4
receives these artifacts plus actual experiment results and reviewed plot
exports; Stage 3 does not claim that the paper is submission-complete.
