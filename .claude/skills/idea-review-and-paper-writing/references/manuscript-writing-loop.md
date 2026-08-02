# Stage 3 manuscript-writing loop

Check non-experimental manuscript execution against one frozen Stage 2
baseline. Record one evidence-bearing row per loop item and substantive
subsection in `WRITING_LOOP_LOG.md` inside the active revision folder.

## Contents

- Loop overview
- W1: baseline consistency
- W2: chapter precedence
- W3: narrative logic
- W4: non-rigid expression
- W5: nonempty purpose
- W6: symbol consistency
- W7: specialized terminology
- W8: formula rigor
- W9: schematic completeness
- W10: manuscript alignment
- Failure, repetition, and done

## Loop overview

| ID | 概括词语 | 简述 |
|---|---|---|
| W1 | 基线一致性 | 本小节不得静默改变被冻结的任务、贡献、结构和技术路线 |
| W2 | 章节先后性 | 理论到模型、整体到局部、前向到损失、设计到应用按依赖展开 |
| W3 | 叙事逻辑性 | 语句和段落具有真实衔接，并形成原因/目的—响应—结果链 |
| W4 | 表达忌生硬 | 避免短陈述句堆砌，按关系自然使用多种句式 |
| W5 | 语句忌空洞 | 每句、每段都有必要功能，不写无意义背景、重复总结或凑字数内容 |
| W6 | 符号一致性 | 首处解释、同义同符号、同类同字体、同一语义家族不冲突 |
| W7 | 名词专有化 | 使用领域标准名词，合理使用缩写，首处给全称，不杜撰术语 |
| W8 | 公式严谨性 | 定义、维度、索引、条件、推导和结论范围完整一致 |
| W9 | 示图完整性 | 原理、模型结构和任务流程需要时均有与正文一致的 LaTeX 简图 |
| W10 | 全文对齐性 | 问题、贡献、正文、图示、摘要和结论保持相同顺序和成熟度 |

## W1: baseline consistency — 基线一致性

- Verify `Context status: FROZEN`, exact `<Baseline ID, revision>` binding, and
  filename/metadata agreement with `references/artifact-naming.md`.
- Compare the subsection's question, output, symbols, terminology, and role
  with the frozen context.
- A new task, contribution, technical route, core symbol/term, or chapter role
  is a context deviation, not an ordinary prose edit.
- Stop, return to Stage 2, refreeze with an incremented revision, and invalidate
  dependent checks. Return first to Stage 1 for new novelty or validity issues.

## W2: chapter precedence — 章节先后性

Check scientific dependency rather than drafting history:

1. theoretical definitions and analysis before dependent model design;

2. overall system/model mapping before local modules and layers;

3. forward structure and interfaces before loss functions and optimization;

4. model construction before its diagnosis, estimation, prediction, control,
   or other task application;

5. prerequisites before formal results, and central results before corollaries
   or implementation implications.

Confirm each heading has one primary question and one reader-facing output.
Search other sections for duplicate responsibility; merge, move, or delete it.
Do not create a heading for one layer, one equation, or a minor training detail.

## W3: narrative logic — 叙事逻辑性

For every paragraph and adjacent sentence pair:

- identify premise, limitation, purpose, definition, mechanism, derivation,
  consequence, boundary, evidence need, or transition;
- verify an explicit relation rather than mere adjacency;
- state why something is needed before saying what is constructed;
- move from the controlling macro question/mapping to the required local detail;
- use `established output -> unresolved requirement -> next action` at section
  changes;
- remove vague pronouns, false connectors, repeated conclusions, unexplained
  topic shifts, and chronology presented as causality.

The preferred chain is:

`problem/limitation/purpose -> method need -> structural action -> changed quantity -> consequence -> task role`

Conclusion-first explanation is allowed only when it immediately states the
conditions, derivation, and scope needed to resolve a complex ambiguity.

## W4: non-rigid expression — 表达忌生硬

- Detect runs of short subject–verb–object sentences and repeated openings such
  as `The proposed method ...`.
- Select active or passive voice according to agency and emphasis.
- Use relative/subordinate clauses, nonfinite constructions, coordination,
  apposition, prepositional openers, and adverbials only when they encode a
  genuine condition, contrast, purpose, mechanism, scope, or result.
- Mix sentence length naturally; split overloaded sentences and combine only
  logically dependent ideas.
- Use adjectives/adverbs only for exact structural, mathematical,
  computational, temporal, statistical, or evidential properties.

Sentence diversity is not a quota. Reject decorative inversions, unnecessary
passive voice, dangling participles, connector stuffing, and long sentences
whose internal logic is harder to follow.

## W5: nonempty purpose — 语句与段落忌空洞

Give every retained paragraph one role: define, motivate, derive, explain,
compare, delimit, connect, or prepare a later task. Record:

- the single theme/purpose;
- why it is necessary;
- its relation to the previous paragraph;
- macro-to-micro level;
- reader-facing output or later dependency.

Every sentence must advance that purpose. Delete or merge a paragraph if its
removal does not break a definition, dependency, argument, derivation,
limitation, or necessary transition. Reject generic importance, module
inventories, drafting commentary, repeated summaries, unsupported promotional
language, and text retained only to increase length.

## W6: symbol consistency — 符号一致性

- **首处解释:** explain every symbol in the Introduction-end Notation paragraph
  or at exact first use.
- **同义同符号:** use one registered symbol for one meaning throughout prose,
  equations, algorithms, appendices, captions, and LaTeX schematics.
- **同类同字体:** use consistent typography for the same object type—italic
  scalars, registered bold vectors/matrices, calligraphic mappings, and
  blackboard-bold sets/spaces according to the context.
- **一符一义:** do not reuse the exact glyph or base family for an unrelated
  meaning merely by changing case, font, subscript, or decoration.

Check meaning, semantic family, naming basis, object type, dimension,
typography, first-definition route, and scope against the frozen registry. A
new object or conflict requires Stage 2 registration, global rename, refreeze,
and rerun—not end-of-paper cleanup.

## W7: specialized terminology — 名词专有化

- Prefer the established term in the target field and journal.
- Verify unfamiliar or contested terms in current primary literature and the
  task-matched case; do not infer terminology from literal translation.
- Introduce an abbreviation only when it will recur enough to improve reading.
- Give `full term (ABBR)` at first use and use the same abbreviation thereafter.
- Keep one term for one concept; reconcile Chinese/English terms in bilingual
  output.
- Do not coin a method name, acronym, “mechanism,” “framework,” or technical
  phrase merely for novelty, brevity, or avoiding repetition.
- Remove obsolete, ambiguous, colloquial, metaphorical, or promotional labels.

## W8: formula rigor — 公式严谨性

- Definitions and assumptions precede use.
- State every index/range, dimension, partition, operator, norm, evaluation
  point, initial condition, data source, and designability status.
- Check dimensional compatibility, algebraic transitions, recursion, equality
  and inequality conditions, boundary cases, and intermediate steps.
- Explain the decisive derivation rather than skipping it in prose.
- Restrict conclusions to the proved local/global, one-/multi-step,
  fixed/time-varying, empirical/theoretical, and conditional scope.
- Combine related multi-line relations under one equation number when they form
  one logical object.

## W9: schematic completeness — 示图完整性

For each figure need in the context, determine whether the technical draft
requires a principle schematic, model structure diagram, or task workflow
diagram. Produce a compileable LaTeX/TikZ sketch when required; record why a
category is unnecessary otherwise.

The schematic must:

- fix scientific topology, major groups, input/output interfaces, arrow
  direction, and exact manuscript symbols;
- have no whole-image title inside the graphic;
- title every major group;
- avoid decorative modules and invented relationships;
- remain simple enough to revise with the manuscript.

Stage 3 checks completeness and scientific meaning, not polished layout,
icons, final palette, or experimental plots. Stage 4 performs visual production.

## W10: manuscript alignment — 全文对齐性

- Keep the central problem count, principal contribution count, order, body
  results, and task roles identical to the frozen context.
- Use one numbered item per central problem in `Problem Formulation` or
  `Problem Description`, normally two or three.
- Keep title, result-neutral abstract, Introduction contributions, roadmap,
  theory, method, task application, schematics, and Conclusion draft aligned.
- Remove an alleged contribution that only names a module, loss, ordinary
  implementation detail, or expected experiment.
- Do not insert experimental findings into Stage 3. Mark result-dependent
  abstract/Conclusion sentences for Stage 4 completion.

## Failure, repetition, and done

Normal progression:

`DRAFT -> CHECK -> PASS -> NEXT`

Failed writing check:

`DRAFT -> CHECK -> FAIL -> REVISE -> CHECK`

Conception deviation:

`CHECK -> CONTEXT_DEVIATION -> STAGE 2 REVISE/REFREEZE -> INVALIDATE -> CHECK`

Run W1–W10 after every substantive subsection revision, after a context
revision, before the Stage 4 handoff, and before any Stage 3 delivery. Record
the actual evidence, conflict, revision, and affected location; a bare `PASS`
does not count.

Stage 3 is complete only when every retained paragraph has a purpose/progression
record, every substantive subsection has current W1–W10 rows, all symbols and
terms match the frozen context, formulas and schematics pass, the technical
draft compiles, and no result-dependent statement masquerades as an observed
experimental finding.
