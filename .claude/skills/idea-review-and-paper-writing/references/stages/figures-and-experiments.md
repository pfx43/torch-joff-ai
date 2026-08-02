# Stage 4: Figures and experiments

Convert stable Stage 3 LaTeX schematics into polished scientific figures,
integrate reviewed quantitative plots produced by the dedicated external Python
plotting library, write evidence-grounded experimental prose, and finalize
result-dependent manuscript parts.

## Contents

- Entry and inputs
- Loop overview
- F1: evidence readiness
- F2: figure-type match
- F3: hierarchical composition
- F4: acyclic arrow topology
- F5: academic visual vocabulary
- F6: information density
- F7: symbol synchronization
- F8: reference and provenance
- F9: palette and readability
- E1: plot–result consistency
- E2: figure-grounded narration
- E3: conclusion restraint
- D1: figure–caption–text closure
- Repetition and output

## Entry and inputs

Require:

- one current frozen `<Baseline ID, revision>`;
- the Stage 3 technical draft and completed W1–W10 records;
- compileable LaTeX schematics or documented reasons why particular figure
  types are unnecessary;
- actual experimental outputs, metric tables, run metadata, or reviewed plot
  exports before writing result claims;
- target-journal figure requirements when available.

Read `references/artifact-naming.md`. Keep every Stage 4 log, figure plan,
editable source, export, integrated plot, PDF, and delivery archive inside the
current revision folder `<short-Baseline-ID>-r<revision>` and use the short
role filenames defined there.

Do not use Stage 4 to redesign the method. A topology, symbol, task, or technical
claim change returns to Stage 2/3 before figure production continues.

## Loop overview

| ID | 概括词语 | 简述 |
|---|---|---|
| F1 | 证据输入就绪 | 没有真实结果、来源和图件需求时不开始结果叙述 |
| F2 | 图型选择匹配 | 区分原理图、模型结构图、任务流程图和定量数据图 |
| F3 | 大框套小框 | 用有标题的大框组织阶段/模块，用小框表达必要细节 |
| F4 | 箭头单向无环 | 箭头必须从形状指向形状，方向清楚、无悬空和视觉环路 |
| F5 | 图形学术化 | 网络层可用形状或克制 icon，避免商务、海报和装饰性图标 |
| F6 | 信息不过载 | 模块说明完整但不冗余，限制图中文字和公式数量 |
| F7 | 图文符号同步 | 图中术语、符号、上下标和方向与 context/正文完全一致 |
| F8 | 参考来源清楚 | 原理和构图参考顶刊/顶会案例并记录来源、许可和借鉴边界 |
| F9 | 配色可读一致 | 每图只用一条配色，冷灰/冷色为主，亮色具有固定语义 |
| E1 | 数据图结果一致 | 数据图必须来自专门 Python 库和已审查结果，不手工改点或补数 |
| E2 | 看图描述有据 | 先说明图检验什么，再报告现象、比较、机制解释和边界 |
| E3 | 实验总结克制 | 只总结数据支持的发现，不把趋势写成证明或因果保证 |
| D1 | 图注正文闭合 | 图、图注、实验段落、摘要和结论相互引用且结论范围一致 |

## F1: evidence readiness — 证据输入就绪

For conceptual figures, require a stable Stage 3 schematic, method equations,
and exact figure purpose. For quantitative plots, require the actual export,
underlying metric/result summary, data split/condition, aggregation, and
uncertainty definition needed to interpret it.

Missing evidence may be listed as a blocking item or a labeled placeholder. It
must not be reconstructed from expected behavior, a paper narrative, or a
visually plausible curve.

## F2: figure-type match — 图型选择匹配

Route by purpose:

- **principle diagram:** explains a scientific mechanism or relation; no fixed
  template, so inspect current related primary papers before composition;
- **model structure diagram:** explains the overall model, internal modules,
  losses, interfaces, and computational direction;
- **task workflow diagram:** explains operational steps such as acquisition and
  preprocessing, offline modeling, and online monitoring/use;
- **quantitative data plot:** reports measured or computed results and is
  produced by the separate Python plotting library, not manually redrawn here.

Do not force one layout onto another figure type.

## F3: hierarchical composition — 大框套小框

Use `references/figure-composition-rules.md` as the authoritative layout guide.
The default organization is major titled frames containing smaller components.

- A model diagram's major frames are overall model modules or loss/optimization
  modules; contained shapes show internal operations, layers, and complete
  input/output interfaces.
- A workflow diagram's major frames are ordered steps; contained shapes form a
  one-way local chain.
- Allow scientifically motivated local zoom-ins and repeated-module stacks.
- Do not place an overall figure title inside the graphic. Every major frame
  must have a concise title.

## F4: acyclic arrow topology — 箭头单向无环

Every arrow must start at one visible shape/port and end at another. Arrows may
cross major-frame boundaries but must remain single-directional, semantically
defined, and visually acyclic. Remove dangling arrows, decorative arrows,
ambiguous bidirectional connectors, and crossings that obscure direction.

If the method is genuinely recurrent or iterative, preserve scientific truth by
unrolling time/iterations or using explicit delayed/update notation rather than
drawing a closed visual loop or falsely deleting the recurrence.

## F5: academic visual vocabulary — 图形学术化

Do not render every small component as the same rectangle. Use restrained
academic icons or schematic motifs when they clarify data, sensors, matrices,
neural layers, loss aggregation, or deployment. Keep icons editable, licensed,
stylistically consistent, and semantically necessary.

For downloaded icons, read `assets/icons/README.md` and use only registry rows
marked `APPROVED`. Alibaba Iconfont and EmojiAll are candidate search sources,
not blanket licenses. When interactive page state or download controls are
needed, use the Codex Chrome browser plugin and retain source/license metadata.

Reject clip-art, glossy 3-D effects, stock-business illustrations, dashboards,
poster aesthetics, mascots, and decorative technology icons. Basic geometric
shapes remain preferable when an icon adds no scientific meaning.

## F6: information density — 信息不过载

Make each small module complete enough to identify its role, input, output, and
essential transformation, but remove prose explanations already provided by the
caption/body. Prefer short noun phrases and symbols to sentences. Include only
the formula that uniquely identifies a mechanism; move derivations to the text.

At final publication size, labels, subscripts, arrows, frames, repeated layers,
and local zoom-ins must remain legible without turning the figure into a poster.

## F7: symbol synchronization — 图文符号同步

Compare every label against the frozen context registry and final manuscript.
Require identical glyph, font family, bold/calligraphic treatment, subscript,
superscript, term, abbreviation, and meaning. A figure cannot introduce a new
symbol or synonym. Any necessary change returns to Stage 2 registration and
invalidates affected Stage 3/4 checks.

## F8: reference and provenance — 参考来源清楚

For principle figures and unfamiliar compositions, search current top-tier or
field-leading primary papers and select references by scientific function, not
surface beauty. Use one of the three category-level cases under
`cases/figure-exemplars/` and add each selected reference as one inventory row
with source, venue/year, figure number, link/file, layout abstraction,
transferable pattern, nontransferable content, and rights status. Do not create
a separate case directory for every reference figure.

Do not copy an unlicensed figure, distinctive complete composition, or paper-
specific scientific claim. Store a local source image only when user-provided,
openly licensed, or otherwise authorized; otherwise store metadata, a source
link, and an independently written structural analysis.

## F9: palette and readability — 配色可读一致

Read `references/scientific-figure-palettes.md`. Select exactly one recorded
strip per figure, use cool gray/blue-gray/desaturated blue-green for ordinary
structure, and reserve one brighter swatch for one defined contribution,
anomaly, intervention, warning, or comparison.

Keep color semantics stable and preserve meaning in grayscale and for common
color-vision deficiencies through line style, labels, markers, or grouping.
Check final-size text, contrast, fonts, strokes, spacing, clipping, and white
space.

## E1: plot–result consistency — 数据图结果一致

Quantitative plots are edited by the dedicated external Python plotting
library, outside this skill. Stage 4 consumes reviewed exports and their result
metadata; it does not recreate the plotting library, manually move points,
invent curves, or smooth away inconvenient observations.

Confirm axes, units, groups, sample size, operating condition, aggregation,
uncertainty type, event markers, baseline identity, and metric definition before
writing. A requested plot correction returns to the plotting workflow and is
reimported after review.

## E2: figure-grounded narration — 看图描述有据

Write each experimental paragraph in this order:

`claim/property under test -> data/protocol context -> observed pattern or value -> relevant comparison -> mechanism-facing interpretation -> limitation`

Do not begin with only “Fig. X shows...”. Name what is being tested and why the
figure is relevant. Distinguish observation from interpretation, and use exact
values only when they are available and correctly scoped.

## E3: conclusion restraint — 实验总结克制

Do not convert empirical separation into theoretical detectability, a trend
into causality, one dataset into generality, or a best mean into statistical
significance. State negative, mixed, or condition-dependent findings when they
occur. Match every adjective to a defined metric, comparison, test, or practical
effect.

Finalize result-dependent abstract, Introduction, Conclusion, and contribution
support only after the experiment text is stable.

## D1: figure–caption–text closure — 图注正文闭合

Each caption defines panels, abbreviations, styles, shading, bounds, conditions,
and data aggregation needed to read the figure. Introduce every figure near its
placement, cite every panel, and state which conclusion follows and within what
boundary. Avoid duplicating the same evidence in a large table unless each form
answers a different question.

Synchronize figure number, source filename, LaTeX reference, caption, body
description, abstract result, and Conclusion. Rerun Stage 3 W6/W7/W10 after
figure and result integration.

## Repetition and output

Run F1–F9 for every conceptual figure revision; E1–E3 for every quantitative
plot/result subsection revision; and D1 after any figure, caption, result,
abstract, or Conclusion change.

Run `python -X utf8 scripts/audit_figures.py <figure-root>`, inspect editable
sources and final exports, compile the manuscript, render the PDF, and rerun
manuscript audits. Run
`python -X utf8 scripts/audit_artifact_names.py <revision-folder> --require-pdf --require-stage4`
before delivery. Stage 4 delivers the editable/reproducible scientific figure
sources, reviewed exports, captions, evidence-grounded experimental prose, final
result-bearing front/back matter, compiled PDF, provenance, and unresolved
limitations.
