# 研究构思评审与论文写作

[简体中文](README.md) | [English](README.en.md)

当前版本：**3.5.0**

本仓库包含一个两阶段 Codex 技能，用于发展研究构思，并将经过确认的研究工作转化为配有可发表科学图件的 IEEE 风格期刊论文。它不仅用于语言润色，还会审查创新性、假设、信息可用性、理论适用范围、计算可实现性、证据、符号、LaTeX 和图件完整性。

本技能尤其面向模糊系统、非线性系统、Koopman 模型、观测器、故障诊断、神经网络、稳定性分析及相关数据驱动控制主题。领域专用指南采用模块化设计，因此可以继续加入模型预测控制、数据补全等研究方向，而不会将某个项目的假设变成全局规则。

纯文本 [`VERSION`](VERSION) 文件是本仓库语义版本号的来源。发布 GitHub Release 时，请使用匹配的 `vMAJOR.MINOR.PATCH` 标签；不要将版本元数据写入 `SKILL.md` 的 frontmatter，因为 Codex 技能发现机制只接受 `name` 和 `description`。

## 目录

- [从这里开始](#从这里开始)
- [两阶段闭环规则](#两阶段闭环规则)
- [规则层级](#规则层级)
- [仓库结构](#仓库结构)
- [典型任务路径](#典型任务路径)
- [可编辑科学图件](#可编辑科学图件)
- [官方 LaTeX 模板](#官方-latex-模板)
- [验证与前向测试](#验证与前向测试)
- [版本与发布](#版本与发布)

## 从这里开始

本技能严格分为两个工作阶段。首先请选择当前所处的阶段。

1. 阅读 [`SKILL.md`](SKILL.md)，了解两阶段边界和路由规则。

2. 对于阶段 1，阅读 [`references/stages/idea-exploration.md`](references/stages/idea-exploration.md)。

3. 对于阶段 2，阅读 [`references/stages/journal-paper-writing-and-figures.md`](references/stages/journal-paper-writing-and-figures.md)，并执行 [`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md)。

4. 在任一阶段，都要阅读 [`references/living-user-rules.md`](references/living-user-rules.md) 以及 `SKILL.md` 路由到的支持文档。

5. 只读取匹配的研究领域和准确对应的论文案例。不要将无关领域或案例的内容作为权威依据。

6. 阅读当前项目的 `MANUSCRIPT_CONTEXT.md`。如果该文件不存在且项目已经较为成熟，请基于 [`assets/templates/manuscript-context.md`](assets/templates/manuscript-context.md) 创建。

## 两阶段闭环规则

本节是 GitHub 首页的执行摘要。阶段 1 的完整权威规则位于
[`references/stages/idea-exploration.md`](references/stages/idea-exploration.md)，
阶段 2 的完整权威规则位于
[`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md)。
README 不建立另一套独立规则；若摘要与权威文件出现差异，以权威文件为准并同步修正摘要。

### 阶段 1：构思探索闭环

阶段 1 不把未经确认的构思直接写成论文贡献。每轮必须依次回答三个问题：

| 关口 | 核心问题 | 通过要求 | 未通过时 |
|---|---|---|---|
| Gate A：现有工作重合 | 构思或结果是否已被已有工作实质完成？ | 基于最新一手文献记录检索日期、来源、检索式、最接近工作、实质重合和精确创新差量 | 标记为 `unresolved` 或 `exploratory`；不得使用“首次”等绝对创新表述 |
| Gate B：假设与可实现性 | 所需信息、假设、训练、数值计算、在线运行和验证是否闭合？ | 明确可测量量、训练期量、在线可得量和不可得量；说明可训练性、离线/在线计算、复杂度和验证路线 | 修正假设、降级为探索路线，或把新增结构拆成独立备选方案；不得暗中改变原信息设定 |
| Gate C：模型专用理论 | 针对特定模型的分析是否构成非平凡贡献？ | 结论必须实质依赖模型结构，具有完整假设和可检查推导，并产生可计算条件、设计规则、界、稳定性、可检测性或其他可验证后果 | 缩小结论范围、补全证明或改为性质/分析结果；不能仅凭模型组合独特而宣称理论创新 |

在以下时点必须重新执行 Gate A、B、C：

- 构思首次提出时；
- 模型结构或信息设定改变时；
- 选择核心贡献、把探索路线提升为 `confirmed`，或起草定理、命题和核心分析结论之前；
- 最近文献更新后，或实验设计表明某项结论不可验证时；
- 最终确定题目、摘要、Introduction 贡献和 Conclusion 之前。

每轮结果写入 `idea-assessment.md` 或匹配案例，并同步到
`MANUSCRIPT_CONTEXT.md`。状态闭环为：

`候选构思 -> Gate A/B/C 检查 -> confirmed -> 阶段 2`

若任一关口失败：

`候选构思 -> 检查失败 -> 降级/修正假设/拆分备选/重定义贡献 -> 重新检查`

只有当技术问题、信息边界、最接近文献、两到三个候选贡献主题、假设与可实现性、选定路线、分析结果和验证计划足够明确，且核心路线已达到 `confirmed`，才进入阶段 2。次要未决事项可以保留，但不得出现在题目、摘要、贡献或形式化结论中冒充已完成结果。

### 阶段 2：论文写作闭环

阶段 2 使用 `SECTION_ROLE_MATRIX.md` 和 `NOTATION_LEDGER.md` 作为控制产物，并按 Gate 0–5 循环：

| 关口 | 执行动作 | 阻断条件 |
|---|---|---|
| Gate 0：初始化 | 阅读完整论文和 `MANUSCRIPT_CONTEXT.md`，建立或更新章节职责矩阵与符号台账 | 控制产物缺失、过期，或存在未标记的章节、符号、维度和问题—贡献冲突 |
| Gate 1：章节职责 | 为每节确定一个主要科学问题和一个读者可见输出；检查紧凑章节顺序、先决关系、重复职责及问题—贡献—正文结果映射 | 章节重复、依赖顺序错误、无依据偏离 skill 章节结构，或中心问题与贡献数量/顺序不一致 |
| Gate 2：符号登记 | 先采用领域/期刊惯例，其次数学惯例、英文首字母或语义助记符；登记语义、命名依据、类型、维度、字体、首次定义位置和作用域 | 符号未先登记，或存在释义、类型、语义家族、字体和首次定义冲突 |
| Gate 3：逐小节检查 | 每次实质新增或修改小节后，重读该小节、核对职责和符号，并完成下面七项检查 | 任一项缺少具体证据、修订动作或仍处于 `FAIL`/`BLOCKED` |
| Gate 4：全文主线对齐 | 对齐摘要、Problem Formulation/Description、两到三个主要贡献主题、正文结果和 Conclusion 的数量、顺序与结论范围 | 各位置陈述不同任务数、不同顺序、不同技术结论，或把模块/实现细节冒充贡献 |
| Gate 5：交付审计 | 运行写作闭环和论文审计，编译并检查 PDF/图件，复核最终符号台账、章节矩阵、所有小节记录和 Gate 4 | 存在未处理警告、错误、过期检查，或交付状态夸大完成度 |

Gate 3 的七项小节检查缺一不可：

1. **章节和小节安排**：核对 skill 规定的紧凑结构、前置依赖、每节唯一职责及任何偏离的期刊或科学理由。

2. **语句间逻辑性**：每句话必须具有明确作用、指代对象和与前后句的逻辑依赖，不允许无解释跳转、含混指代或重复结论。

3. **叙事因果性**：建立“问题/局限—方法需求—结构作用—推导结果—证据检验”的因果链，不能用模块排列或时间先后冒充机制解释。

4. **符号一致性**：所有符号必须先登记，在 Introduction 末尾的 Notation 或首次出现处解释，并保持唯一释义、命名依据、对象类型、字体、维度和作用域。

5. **公式严谨性**：检查定义和假设的先后顺序、索引、维度、算子、初始条件、数据来源、代数步骤、成立条件、边界情况和结论范围。

6. **模型描述完整性**：写清输入、输出、物理/潜在状态、已知/未知量、扰动或故障、固定/学习映射、参数、假设、更新顺序、时间索引和模块接口。

7. **训练/验证/测试/部署清晰性**：分别说明数据来源和划分、输入与标签、预处理拟合范围、目标与约束、优化、模型选择、测试专用操作、指标、不确定性、离线/在线计算和信息泄漏防护。

每个小节的状态转换为：

`DRAFT -> CHECK -> PASS -> NEXT`

若检查失败：

`DRAFT -> CHECK -> FAIL -> REVISE -> CHECK`

只写一个没有证据的 `PASS` 不算完成。每轮必须记录检查时间、检查证据、发现的冲突、修订动作、受影响位置和最终状态。后续修改若改变章节顺序、因果链、符号、公式、模型接口、训练/测试流程、中心问题或贡献结论，所有依赖该内容的既有 `PASS` 立即失效并必须重跑。

阶段 2 只有在章节矩阵、符号台账、每个小节的七项记录、摘要—问题—贡献—正文对齐、自动审计、编译和视觉检查全部通过，或仅剩已明确披露的非阻断限制时，才能标记为完成并交付。

## 规则层级

不同文档具有不同的权威性和适用范围。

| 层级 | 用途 | 适用条件 |
|---|---|---|
| 用户当前明确指令 | 用户对当前任务的决定 | 始终具有最高优先级 |
| `MANUSCRIPT_CONTEXT.md` | 单篇论文中已确认和探索中的决定 | 仅适用于该论文 |
| 匹配的案例文件 | 单篇已识别论文的长期讨论历史、备选方案和未决问题 | 仅适用于该篇论文 |
| 领域指南 | 某一研究方向的有效模式、注意事项和备选路径 | 仅当论文匹配该领域且选择了相应路径时适用 |
| 持续维护的用户规则 | 可跨论文沿用的长期偏好和推理规则 | 适用于每项论文任务 |
| `SKILL.md` 默认规则 | 工作流和面向 IEEE 的默认设置 | 除非被更高优先级的指令或项目决定覆盖 |
| 典型错误与参考模式 | 负面检查和基于文献的结构指南 | 由 `SKILL.md` 路由时适用 |

领域文档中记录的构思不会自动成为论文的论断、假设或选定方法。当前项目必须将重要决定分类为：

- `confirmed`：已采纳，可以作为论文依据；
- `exploratory`：值得分析，但尚不能作为论文论断；
- `alternative`：竞争性方案，不得与已选路径混用；
- `rejected`：已经考虑并明确排除；
- `unresolved`：需要证据、推导或用户明确决定。

不得在未说明的情况下，将探索性或备选构思提升为已经确认的论文假设。

## 仓库结构

### 核心工作流

- [`SKILL.md`](SKILL.md)：必须遵守的两阶段路由、共享规则、参考资料使用方式和交付要求。
- [`agents/openai.yaml`](agents/openai.yaml)：调用本技能的界面元数据。

### 阶段 1：构思探索

- [`references/stages/idea-exploration.md`](references/stages/idea-exploration.md)：针对现有工作重合度、假设与实际可实现性，以及模型专用理论贡献的重复检查。

### 阶段 2：期刊论文写作与图件

- [`references/stages/journal-paper-writing-and-figures.md`](references/stages/journal-paper-writing-and-figures.md)：论文构建、符号、形式化论断、实验、科学图示、定量图表、图注、视觉完整性、可复现图件源文件、LaTeX 和 PDF 验证。
- [`references/manuscript-quality-gates.md`](references/manuscript-quality-gates.md)：章节安排、逐句逻辑、因果叙事、符号命名依据与一致性、公式严谨性、模型完整性、训练/验证/测试/部署流程，以及摘要—问题—贡献对齐的强制闭环。
- [`references/latex-template-workflow.md`](references/latex-template-workflow.md)：IEEE 和《控制理论与应用》官方模板的选择、验证下载、仅编辑正文的边界、依赖项处理和完整性审计。

### 共享支持规则

- [`references/living-user-rules.md`](references/living-user-rules.md)：长期偏好、符号约定、贡献表述规范、定理分类、论文语气、回复语言和更新协议。
- [`references/user-writing-requirements-and-preferences.md`](references/user-writing-requirements-and-preferences.md)：网页版写作记忆的完整清单，保留通用规则、项目专用规则、非论文写作偏好和明确未知边界。
- [`references/source-rule-coverage.md`](references/source-rule-coverage.md)：最早版 `ieee-english-paper-polish` 与网页版清单的迁移覆盖、冲突兼容和来源记录。
- [`references/technical-validity-and-implementation.md`](references/technical-validity-and-implementation.md)：从第一性原理检查可观测性和信息可用性、非线性论断范围、可训练性、可计算性、硬约束、离线与在线分离、复杂度、实验和评审。
- [`references/rule-scope-map.md`](references/rule-scope-map.md)：维护索引，标明各类规则的权威文件和适用范围，用于避免重复和更新错位。
- [`references/typical-errors.md`](references/typical-errors.md)：禁止重复出现的模式及其替代方式。
- [`assets/templates/manuscript-context.md`](assets/templates/manuscript-context.md)：当前项目上下文模板。
- [`assets/templates/idea-assessment.md`](assets/templates/idea-assessment.md)：用于应用通用构思检查的阶段 1 评估产物。
- [`assets/templates/section-role-matrix.md`](assets/templates/section-role-matrix.md)：阶段 2 章节职责和问题—贡献映射模板。
- [`assets/templates/notation-ledger.md`](assets/templates/notation-ledger.md)：阶段 2 符号语义、命名依据、对象类型、维度、首次定义路径和作用域注册表。
- [`assets/templates/figure-plan.md`](assets/templates/figure-plan.md)：用于规划和验证单幅科学图件的阶段 2 产物。

### IEEE TFS 参考指南

- [`references/tfs-reference-patterns.md`](references/tfs-reference-patterns.md)：可复用的“模型—证明”链条、TFS 叙事模式、动态阈值指南和引用规范。适用于 TFS 故障诊断、T–S/IT2 模糊观测器、残差阈值、事件触发诊断、知识蒸馏模糊诊断及相关需要参考文献支撑的工作。

### 论文专用案例库

- [`cases/README.md`](cases/README.md)：案例库边界、权威性、命名、同步和提升规则。
- [`assets/templates/paper-case.md`](assets/templates/paper-case.md)：单篇论文讨论记录的模板。
- [`cases/fault-diagnosis/`](cases/fault-diagnosis/)：各篇故障诊断论文案例。
- [`cases/soft-sensing/`](cases/soft-sensing/)：各篇软测量和非线性观测器论文案例。

每篇论文对应一个案例文件。案例用于保留局部讨论和备选方案，但不会为其他论文创建规则。

### 研究领域

- [`references/domains/README.md`](references/domains/README.md)：领域路由规则和新增研究方向的说明。
- [`references/domains/fault-diagnosis.md`](references/domains/fault-diagnosis.md)：未知故障的信息边界、Koopman–T–S–注意力建模、与测量解耦的正常参考、联合残差、动态阈值、后处理滤波、结构化升维扰动基和在线实现。
- [`references/domains/soft-sensing-and-observers.md`](references/domains/soft-sensing-and-observers.md)：潜在状态解释、非线性观测器稳定性、可测与不可测状态划分、压缩层、长记忆网络和质量变量预测。

模型预测控制和数据补全被保留为未来的领域模块。只有在具备具体且可复用的规则时才创建相应文档，不要用虚构的默认规则填充它们。

## 典型任务路径

### 阶段 1 任务

新构思、文献创新性检查、模型选择、假设分析、可行性、理论路径探索和实验规划从 `references/stages/idea-exploration.md` 开始。使用技术有效性文档和领域文档作为辅助检查。案例可以记录单篇论文的讨论，但不包含通用的阶段 1 规则。

### 阶段 2 任务

语言与结构修改、定理呈现、完整论文起草、LaTeX 交付、架构图、工作流、图表、图注和 PDF 视觉质量检查使用 `references/stages/journal-paper-writing-and-figures.md`。如果阶段 2 暴露出尚未解决的技术问题，请将该问题退回阶段 1。

### 新增研究方向

遵循 [`references/domains/README.md`](references/domains/README.md)。将可复用的领域知识放入新的领域文件，将单篇论文的选择放入 `MANUSCRIPT_CONTEXT.md`，仅将真正可跨论文复用的规则提升到 `living-user-rules.md`。

### 新增论文案例

遵循 [`cases/README.md`](cases/README.md)。在匹配的任务分组下创建一个文件，将它链接到当前的 `MANUSCRIPT_CONTEXT.md`，且不得将其作为另一篇论文的权威依据。

## 可编辑科学图件

对于架构图、机理图和工作流图，首选交付物是可编辑源文件以及用户要求的发表用导出文件。支持的源文件形式包括 `.drawio`、结构化 SVG、可编辑 PowerPoint `.pptx`、TikZ 和可复现绘图脚本。位图预览不能替代可编辑源文件。

安装并可调用可选的 [Draw.io Scientific Illustrator](https://github.com/icebird1998/drawio-scientific-illustrator) 集成后，阶段 2 指南会将适合的图件交给其实时 draw.io 工作流，并保留 `.drawio` 源文件。该集成的 MIT 许可证覆盖其代码，但不会使第三方参考图件免受版权限制。

绘图前请使用 [`assets/templates/figure-plan.md`](assets/templates/figure-plan.md) 定义图件含义、视觉编码、源文件格式、论文中的放置位置和验证要求。

## 官方 LaTeX 模板

官方压缩包注册表为 [`assets/latex-templates/sources.json`](assets/latex-templates/sources.json)。下载的压缩包保存在已被 git 忽略的本地缓存中，因为两家出版机构的再分发条款不同。

除非用户明确选择其他期刊模板，否则所有中文论文均使用已登记的《控制理论与应用》模板；在尚未选定更具体的目标模板时，英文期刊论文默认使用已登记的 IEEE 期刊模板。两个已登记模板的论文正文都是双栏；中文模板在进入双栏正文前使用单栏前置信息区。

```powershell
python -X utf8 scripts/fetch_latex_templates.py
python -X utf8 scripts/create_latex_project.py --template ieee-journal --destination <project-directory>
python -X utf8 scripts/create_latex_project.py --template control-theory-and-applications --destination <project-directory>
python -X utf8 scripts/audit_latex_template.py <project-directory>
```

只初始化当前使用的期刊模板。在复制出的主文件中撰写论文内容，不要更改文档类、页眉、宏包、参考文献样式、页面布局、字体和其他模板控制项。初始化程序会保留源文件字节，并记录 `TEMPLATE_LOCK.json` 完整性基线。

## 验证与前向测试

每次修改结构或规则后，都要运行仓库验证程序：

```powershell
python -X utf8 scripts/validate_skill.py .
```

阶段 2 还包含用于 LaTeX 源文件审计、可编辑图件审计、编译和渲染页面检查的独立辅助程序：

```powershell
python -X utf8 scripts/audit_manuscript.py <project-root>
python -X utf8 scripts/audit_figures.py <figure-root>
python -X utf8 scripts/compile_manuscript.py <main-tex>
python -X utf8 scripts/audit_latex_template.py <project-root>
```

[`tests/forward-tests.json`](tests/forward-tests.json) 包含以下回归场景：阶段 1 的三个重复检查关口、可编辑矢量文件交付、官方模板保留，以及跨案例迁移边界。测试夹具记录提示词、必需行为和禁止行为；评估器应让本技能处理每条提示，并根据这些字段对回复评分。

## 版本与发布

使用语义化版本：

- 对不兼容的工作流、路由或规则归属变更，增加 `MAJOR`；
- 对向后兼容的领域指南、模板、工具或能力变更，增加 `MINOR`；
- 对不改变预期行为的修正，增加 `PATCH`。

发布前，请运行仓库验证程序和相关前向测试，更新 [`VERSION`](VERSION)，提交经过评审的源代码状态，创建匹配的 Git 标签（例如 `v3.0.0`），并发布该准确提交。创建标签或 GitHub Release 属于单独的外部操作，只有在用户明确要求时才能执行。
