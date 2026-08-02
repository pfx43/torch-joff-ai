# Stage 2: Paper conception

Convert a confirmed research route into one frozen paper-specific baseline.
Apply every Stage 2 rule only to
`MANUSCRIPT_CONTEXT.md` inside the active revision folder; do not draft final paragraphs,
equations, experimental prose, or polished figures in this stage.

## Contents

- Entry and baseline identity
- Loop overview
- C1: background and task
- C2: contribution correspondence
- C3: technical main line
- C4: model-definition order
- C5: notation planning
- C6: terminology planning
- C7: chapter precedence
- C8: narrative blueprint
- C9: cross-check and freeze
- Repetition and output

## Entry and baseline identity

Enter only after Stage 1 confirms the central task, closest-work delta,
information boundary, assumptions, realizable route, analytical scope, and two
or three candidate contribution themes.

Read `references/artifact-naming.md`. Create the draft folder
`<short-baseline-id>-r0/` and place `MANUSCRIPT_CONTEXT.md`, created from
`assets/templates/manuscript-context.md`, inside it. The first frozen snapshot
becomes the sibling folder `<baseline-id>-r1/`; every later refreeze creates the
next sibling revision folder without overwriting the prior frozen snapshot.

- **Baseline ID** identifies the conception lineage of one paper. Use a short,
  recognizable slug of one to three terms, preferably no more than 16
  characters and never more than 24, for example `koop-fd`, and keep it
  unchanged while refining that same paper.
- **Context revision** identifies the version inside that lineage. Start at
  `1` for the first freeze and increase it by one after every later refreeze.
- **FROZEN** means the recorded revision passed C1–C9 and is permitted as a
  Stage 3/4 input. The exact snapshot is the pair `<Baseline ID, revision>`.
- **Revision folder** is `<baseline-id>-r<revision>`. Files inside it use short
  role names; the folder supplies their paper/revision identity.
- A changed central task or contribution structure that amounts to another
  paper receives a new baseline ID and normally a new paper case.

## Loop overview

| ID | 概括词语 | 简述 |
|---|---|---|
| C1 | 背景与任务定调 | 先讲为什么需要研究，再明确论文究竟解决什么 |
| C2 | 贡献逐项对应 | 每个中心问题只对应一个主要贡献主题和明确结果 |
| C3 | 技术主线成链 | 把问题、方法需要、机制、结果和任务作用连成因果链 |
| C4 | 模型定义有序 | 先整体映射，再前向结构、局部模块、损失优化和任务应用 |
| C5 | 符号提前规划 | 所有计划符号先登记含义、类型、字体、维度和首次解释位置 |
| C6 | 名词提前规划 | 采用领域专有名词，判断是否需要缩写并禁止杜撰术语 |
| C7 | 章节先后明确 | 理论、模型、优化和应用按科学依赖安排，不按模块数量拆章节 |
| C8 | 行文逻辑预演 | 预先规划每节和每段为什么存在、如何从宏观推进到局部 |
| C9 | 基线交叉冻结 | 定位、技术、符号、术语、结构和叙事无冲突后才冻结 |

## C1: background and task — 背景与任务定调

Complete the background/task baseline in this order:

1. application or scientific setting;

2. concrete failure, limitation, or unmet requirement;

3. why that limitation matters;

4. why existing method families remain insufficient under this paper's
   information, modeling, or computational setting;

5. the exact task solved and adjacent tasks excluded;

6. two or three central paper questions.

Do not begin with a module list or generic importance statement. A reader
should be able to retell both the reason for the paper and its exact task before
seeing the proposed architecture.

## C2: contribution correspondence — 贡献逐项对应

Map every central question to one principal contribution theme, one
problem-facing construction, one nontrivial result/capability, one body
location, and one admissible evidence need. Keep identical count and order
through the context.

Use two or three principal themes. If the venue needs three to five concrete
construction, analysis, and validation statements, nest them under those
themes. Reject contributions that merely name modules, losses, training tricks,
implementation details, or unsupported superiority.

## C3: technical main line — 技术主线成链

Write one retellable main line and one dependency table following:

`problem/limitation -> method need -> overall construction -> mechanism -> analytical or algorithmic consequence -> task role -> evidence need -> scoped conclusion`

Every component, theorem, loss, subsection, schematic, and later experiment
must serve one link. Merge or remove branches that add complexity without a
necessary mechanism, result, or explanatory role.

## C4: model-definition order — 模型定义有序

Plan the technical description in this order unless a proven dependency
requires another sequence:

1. **theoretical object and task:** system/data relation, information setting,
   assumptions, and formal problem;

2. **overall model mapping:** complete input-to-output or state-to-result map;

3. **forward structure:** main modules and their interfaces in computation
   order;

4. **local design:** internal layers, transformations, constraints, and
   required variables;

5. **loss and optimization:** objectives, regularizers, design variables,
   update order, and offline computation;

6. **task application:** how the learned/designed output performs estimation,
   diagnosis, prediction, control, or another declared task.

Record inputs, outputs, known/unavailable quantities, parameters, dimensions,
initialization, and interfaces as conception facts. This is a definition-order
baseline, not a repeated “model completeness” prose score and not an
experimental protocol.

## C5: notation planning — 符号提前规划

Register every planned mathematical object in the context Markdown table. For
each row record semantic family, meaning, naming basis, object type, dimension,
typography, first-definition route, and scope.

Choose symbols by field/journal convention, then mathematical convention, then
a meaningful English initial or mnemonic, and only then a justified
project-specific choice. Search exact glyphs and case/font/base-family variants.
Resolve meaning, object-type, semantic-family, and typography conflicts before
freezing. Plan every symbol to be explained either in the Introduction-end
Notation paragraph or at first use.

## C6: terminology planning — 名词提前规划

Create a terminology registry for every important method, task, state,
quantity, result, and regime.

- Prefer terms established in the target field and verified in current primary
  literature or the task-matched case.
- Introduce an abbreviation only when the term recurs enough to justify it;
  record the full form and first-use location.
- Use `full term (ABBR)` at first use and the same abbreviation thereafter.
- Do not coin a term or acronym merely to avoid repetition or make the method
  sound novel.
- Record prohibited, obsolete, ambiguous, or project-rejected alternatives.

## C7: chapter precedence — 章节先后明确

Start from the compact non-experimental sequence:

1. Introduction;

2. Preliminaries and Problem Formulation;

3. theoretical analysis needed before design;

4. Proposed Method: overall mapping, forward structure, local design, and
   loss/optimization;

5. task application or a genuinely independent second task;

6. Conclusion draft, with result-dependent claims deferred to Stage 4;

7. appendices for long proofs.

Stage 4 later adds Experiments and finalizes result-dependent front/back matter.
For every section/subsection record one question, input, output, dependency,
problem/contribution ID, and justification for deviation. Search for duplicate
responsibilities and merge, move, or delete them. A network layer or one
equation is not by itself a section.

## C8: narrative blueprint — 行文逻辑预演

Plan paragraph purposes and transitions without writing final prose:

- Introduction: need -> exact task -> method-family synthesis -> shared
  limitation -> questions -> overall response -> contributions -> roadmap;
- theory/method: overall problem/mapping -> prerequisite analysis -> forward
  structure -> local mechanism/equation -> loss/optimization -> task use;
- each paragraph: cause, condition, limitation, or purpose -> response ->
  consequence or bridge;
- each transition: established output -> unresolved requirement -> reason for
  the next paragraph/section.

Record macro-to-micro movement and the specific reader-facing purpose of each
planned paragraph. Empty “background,” “module overview,” or “summary” roles do
not pass unless they perform a necessary argumentative function.

## C9: cross-check and freeze — 基线交叉冻结

Run six passes and record evidence/revisions:

1. positioning: background, task, closest-work limitation, and questions;

2. contribution: problem–contribution–result order and scope;

3. technical: theoretical dependency, model-definition order, loss, and task
   application;

4. notation and terminology: symbols, fonts, full terms, abbreviations, and
   first-use routes;

5. structure: chapter responsibilities and prerequisites;

6. narrative: paragraph purpose, causal order, and macro-to-micro progression.

Resolve every blocking issue, set `Context status: FROZEN`, retain the stable
baseline ID, set a positive revision, record the freeze date, and list evidence
checked. A bare `FROZEN` is invalid.

## Repetition and output

Use:

`DRAFT_CONTEXT -> CHECK -> REVISE -> CHECK -> FROZEN`

Reopen Stage 2 whenever Stage 3 or 4 discovers a conflict in task, contribution,
technical route, core notation/terminology, chapter responsibility, or causal
story. Increase the revision and invalidate dependent downstream checks. A new
novelty, assumption, proof, or feasibility question returns first to Stage 1.

Stage 2 produces exactly one active frozen `MANUSCRIPT_CONTEXT.md` inside the
selected revision folder while preserving older sibling revision folders as
history. It does not produce manuscript
paragraphs, experimental-result text, polished figures, a separate notation
ledger, or a separate section-role matrix.
