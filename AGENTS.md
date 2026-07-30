# AGENTS.md

本文件是自动化代理在本仓库中的唯一根级协作入口。项目状态与技术事实应引用下述权威文档，不在多个代理入口之间复制维护。

## 1. 仓库性质与权威来源

本仓库同时承载两条工作线：

1. **Joff**（`src/joff/`）：spec-first 的 PyTorch 过程数据实验工具包，主链为“配置 -> 数据 -> 模型 -> 训练 -> 评估 -> 产物”。
2. **论文项目**（`paper/`）：面向 IEEE TFS 的受保护 Attention-Koopman-T-S 模糊故障检测与结构化隔离论文。Joff 是实验底座，论文专用方法目前尚未完整实现。

进入仓库后按以下优先级读取：

1. `docs/01-文档导航.md`：项目文档的唯一入口。
2. `MANUSCRIPT_CONTEXT.md`：英文论文的项目级写作上下文；处理 `paper/` 前必须整读。
3. `docs/03-当前工作记录.md`：当前阶段、阻塞、验证结果和近期决策。
4. `docs/04-技术架构.md`：运行时分层、调用链、扩展接口和架构不变量。
5. `docs/05-目录结构与文件数.md`：仓库目录、核心文件地图、文件统计口径和源码阅读顺序。

`docs/旧文档/` 是被新方案取代或归档的历史材料，不是当前实现规格。需要追溯旧推导时可读取，但不得用它覆盖 `MANUSCRIPT_CONTEXT.md`、`paper/main.tex` 或最新工作记录中的决策。

项目文档使用中文；`docs/` 下 Markdown 使用中文文件名。根级标准文件（如 `README.md`、`AGENTS.md`）保留标准英文名。

## 2. 常用命令

```bash
# 可编辑安装与常用开发/论文依赖
python -m pip install -e ".[dev,paper,hdf5]"

# 测试；pyproject.toml 已配置 testpaths=tests、pythonpath=src、默认 -q
python -m pytest
python -m pytest tests/test_phase1_core.py -q
python -m pytest "tests/test_phase1_core.py::test_unknown_config_key_raises_helpful_error" -q
python -m pytest --collect-only -q

# Lint；line-length=100、target py310、排除 datasets/raw
python -m ruff check .

# 冒烟路径；只验证代码路径，数值不能作为论文结果
python examples/fd_cstr.py --smoke
python examples/data_preset_cstr_fd.py

# 当前英文论文
latexmk -cd -pdf -interaction=nonstopmode -file-line-error -halt-on-error paper/main.tex

# 仅在明确维护归档中文设计稿时使用
pwsh docs/旧文档/编译PDF.ps1
```

截至 2026-07-30，pytest 可从 21 个测试文件收集 305 个测试；这只是收集结果，不代表全套测试已经通过。执行验证后只报告实际运行过的范围。MiKTeX 若报 `Support package 'expl3' too old`，先执行 `mpm --update`；若首次安装设置未完成，不得声称 PDF 已重新编译或目检。

## 3. 运行时架构与不变量

主调用链：

`配置解析(core) -> 实验编排(experiments) -> 数据(data) -> 模型构建(models/layers) -> 训练(training) -> 评估(evaluation) -> 产物(artifacts)`

`plotting`、`tracking`、`console` 和 `xai` 是支持模块，不拥有实验业务状态。数据有两条进入路径：

- preset：`DatasetRegistry -> DatasetAdapter -> CanonicalDataset -> DataModule`；
- 直接文件：source readers 读取 CSV、Excel、MAT、NPY 或 NPZ，再交给 `DataModule`。

修改代码时必须维持：

- 所有 Pydantic 配置严格拒绝未知字段；每次解析保留字段 provenance 和 16 位配置哈希。
- schema 与 task 决定列语义，模型不得猜测原始列；拟合型预处理只能从允许的正常训练 split 学参数。
- 模型只负责 `forward(batch)`、`compute_loss(...)` 和模型自身配置；模型不创建运行目录，`Trainer` 不读取原始数据文件。
- `import joff` 无副作用：不读数据、不建目录、不启动追踪、不修改 Matplotlib 全局状态。
- 新模型、评估器和数据集走显式注册表（`register_model`、`register_evaluator`、`DatasetAdapter` + registry），由 `core.factory` 按 `type` 构建，不使用任意动态导入。
- 故障检测评估器保持 `fit(normal_scores)` / `evaluate(test_scores, labels)` 两阶段接口，使正常拟合与冻结测试在接口上分开。
- 单次 `Experiment` 负责组合下层模块；`Study` 只展开配置和聚合结果。
- 论文结果必须可追溯到解析配置、数据摘要、随机种子、checkpoint 和逐次运行产物。
- 可选绘图、追踪和数据依赖不得使基础包导入失败。

分层职责、扩展方式和当前限制以 `docs/04-技术架构.md` 为准，不在本文件重复完整代码地图。

## 4. 数据与论文硬约束

论文遵守“仅正常数据”纪律：训练、结构选择、超参数选择、窗口选择、阈值校准、anchor gate 和状态策略冻结都不得使用故障数据。故障数据只能进入协议冻结后的最终测试。

当前理论要求五个互斥的正常数据阶段；前四段用于拟合与两次独立校准，第五段是冻结正常测试。
真实冻结故障测试是五段之外的独立封存范围，只能在整个正常协议冻结后访问。五个正常段之间保留隔离带：

`正常训练 -> 正常估计 -> 检测校准 -> 归因校准 -> 冻结正常测试`

随后才允许进入独立的 `冻结故障测试`。这一口径与 `paper/main.tex` 的 Learning and
Calibration Protocol 一致，不能把故障测试混写成正常数据第五段。

检测校准和归因校准 episode 不得复用。通用 `DataModule` 不拥有论文五段业务状态；正式
五段抽象已经由 `data.paper_protocol` 的 `FiveStageNormalSplitter` / `PaperDataBundle`
实现，不能在示例脚本里临时切数组，也不能退回旧 `FourWayNormalSplitter` 规格。

闭环 CSTR preset `cstr_closed_loop_fd` 的物理协议问题已经闭环，但数据许可仍未闭环：

- 测试 episode 按真实 onset=200 逐行标注；
- `u=(Ci,Ti,Tci)`、`y=(C,T,Tc,Qc)` 已进入物理 schema；
- MathWorks File Exchange #66189 版本 1.1.0.1 的上游模型 BSD-3-Clause URL、许可证文本
  hash、本地模型 hash 和完整第三方 notice 已进入数据卡和公共 adapter 来源摘要；
- 上述证据只覆盖上游模型，仓库内 normal/fault MAT 的生成链和再分发许可仍为
  `to_verify`，不得把模型许可扩张为数据许可。
- paper development/frozen 配置必须绑定精确的数据集卡 SHA-256；入口还会交叉核对卡片中的
  raw 路径、许可状态和原始文件 SHA-256。未来把许可改为 `verified` 时，卡片必须同时绑定
  已存在且 hash 匹配的 MAT 生成记录和许可证据。仅修改配置或卡片状态不能解锁数据访问；
  卡片与证据只能位于仓库或声明的数据根，normal/fault 子路径不得逃离数据根。软件校验
  只能固定证据身份和路径边界，不能替代人工法律与所有权审查。

当前 formal 同时受两类门禁阻塞：本地 MAT 生成/许可链尚未核实，以及仓库没有真实
interval/verified-quadrature 与 full nonlinear 认证 provider。任一条件未满足时，
development-only runtime 都不得创建正式 manifest/claim 或读取冻结故障值。

老师稿的两处 critical 修正已经进入当前英文方案但仍需导师最终确认：推论 1 使用可观测超额得分校准；隔离层使用堆叠受保护多步残差与加权响应矩阵配对。最新边界以 `MANUSCRIPT_CONTEXT.md` 和 `paper/main.tex` 为准。

不得把 smoke、占位符或未冻结结果写成论文实验结论；不得宣称当前方法已完整实现、理论已饱和或结果可投稿。`MANUSCRIPT_CONTEXT.md` 记录的 `Pause / Not theoretically saturated` 状态在新证据出现前保持有效。

## 5. 数据集与产物边界

- OA 原始数据放在 `datasets/raw/oa/**`，公开数据集卡片放在 `datasets/cards/oa/**`，公开清单是 `datasets/manifest.public.yaml`。
- private 原始数据、数据集卡片、manifest 和 private 示例不得入库；遵守 `.gitignore` 中的 private 边界，不得为方便实验而强制添加。
- `runs/`、`outputs/`、checkpoint、追踪目录和 LaTeX 中间文件是生成产物，不纳入版本控制。
- 不修改或删除不属于当前任务的工作树变更。归档、教师稿、数据和论文文件可能很大，操作前先确认精确路径。

## 6. 论文写作与交付

- 当前正式稿为英文 `paper/main.tex`，参考文献为 `paper/refs.bib`，体例为 IEEEtran 双栏、IEEE TFS 风格。
- 处理英文稿前必须按 `ieee-english-paper-polish` 技能要求整读 `MANUSCRIPT_CONTEXT.md`。
- 实验节允许完整协议和明确占位符，不允许编造数值、硬件条件、统计显著性或已完成验证。
- 修改 LaTeX 后必须本地编译；交付前检查错误、未定义引用、超宽行并逐页目检 PDF。若环境阻塞，应报告阻塞和未完成的检查。
- 编辑 `.tex` 应使用能保真反斜杠的文件编辑工具，避免 shell heredoc；中文材料使用全角中文引号，避免 ASCII 直引号造成排版错误。

新增或重命名文档必须登记到 `docs/01-文档导航.md`。只有改变研究判断、阶段、阻塞或关键证据的里程碑才写入 `docs/03-当前工作记录.md`；普通调试运行留在运行产物中。

## 7. 文献检索与 Zotero 工作流（强制）

**任何进入论文的文献，必须先入 Zotero，再被引用。** 严禁凭模型记忆编写参考文献；`.bib` 条目只能来自 Zotero 导出或权威数据库返回的已核验元数据。

标准流程不可跳步：

1. **检索**：用 `/paper-triage` 排序候选，用 `/expand-references` 从 1-3 篇种子扩展，用 `/trace-citations` 追踪前向/后向引用。仅在查询项目主页、代码仓库、数据集或会议通知等非论文信息时使用普通 Web 搜索。
2. **入库**：确认相关后立即用 `zotero_add_by_doi`，或按需使用 `zotero_add_by_url` / `zotero_add_by_bibtex`，并设置集合与标签。未入库论文不得进入正文、相关工作或 `.bib`。
3. **精读**：通过 Zotero MCP 检索全文和 PDF 标注，不重复下载、上传或创建重复条目。
4. **引用**：从 Zotero 导出 BibTeX。投稿前核对标题、作者、年份、DOI，去除编造条目，并确认预印本与正式版本没有混淆。

若写作时在 Zotero 中查不到引用，必须回到检索和入库步骤，不能凭印象补条目。

集合按论文结构划分：

`01-综述与基础` / `02-Koopman` / `03-T-S模糊` / `04-Attention与编码` / `05-故障检测` / `06-故障隔离` / `07-正常数据建模` / `08-实验与数据集` / `09-待精读` / `10-已引用`

Zotero 服务配置在 `.mcp.json`（Claude Code）和 `~/.codex/config.toml`（Codex）。本地读取需要 Zotero 客户端运行、允许其他应用通信，并使用 `ZOTERO_LOCAL=true`；Web API 写入依赖用户级环境变量 `ZOTERO_API_KEY` 与 `ZOTERO_LIBRARY_ID`。`SEMANTIC_SCHOLAR_API_KEY` 可提高检索限额。所有密钥只放系统环境变量，绝不写入仓库、日志、事项或文档。

## 8. Agent 协作约定

### 解释与沟通方式（强制）

默认把用户视为刚开始接触本项目和相关理论的初学者。说明代码、论文、数学、实验或
开发决策时，必须以详细、易懂、可以跟着执行为目标，不能假设用户已经理解专业背景。

- 先用通俗中文说明“这是什么、为什么需要、会影响什么”，再介绍实现细节或公式。
- 专业术语、英文缩写和符号第一次出现时必须解释；例如首次使用 FAR 时，应同时说明它
  是“误报率”，以及它在本实验中具体统计什么。
- 复杂流程按前置条件、输入、操作步骤、输出、验收方法和常见失败原因逐项讲清。
- 给出建议时同时解释推荐理由、其他选项的代价，以及用户当前需要确认的具体决定。
- 代码说明应指出文件位置、数据如何流动、每个关键对象负责什么，并用小例子解释张量
  形状、时间窗口、数据切分和状态变化。
- 数学说明先给直觉和最小例子，再给公式；不得只复述公式或用更多未解释的术语解释术语。
- 不得用“显然”“简单”“同理可得”等表达跳过对初学者必要的推理步骤。
- 信息较多时分阶段讲解，每一阶段都给出明确结论和下一步，避免一次堆出无法执行的大清单。

### 事项追踪器

事项和规格以 Markdown 文件存放在 `.scratch/` 下；目录按需创建。工单属于工作产物，创建后应纳入版本控制，不要加入 `.gitignore`。格式和状态更新见 `docs/agents/事项追踪器.md`。

### 分诊标签

使用五个规范化分诊标签；名称和含义见 `docs/agents/分诊标签.md`。

### 领域文档

领域上下文和 ADR 的发现顺序见 `docs/agents/领域文档.md`。约定文件不存在时静默继续，不为填空而创建空文档。

### Skill 目录

`.claude/skills/` 与 `.agents/skills/` 是同一批技能面向不同工具的镜像。修改任一侧后必须同步另一侧并验证内容一致，不能让两份拷贝分叉。

论文相关项目技能：

- `ieee-english-paper-polish`：IEEE 英文稿写作与润色；
- `paper-triage`、`expand-references`、`trace-citations`：基于 Semantic Scholar 的文献检索与引用网络追踪。

## 9. 文件说明与代码注释规范（强制）

### 9.1 文件顶部说明

以后新建或修改任何**由项目维护的代码文件**时，必须在文件顶部补充或更新详细的中文说明。Python 文件必须使用模块级中文 docstring；其他代码文件按语言的原生注释形式书写。已有代码文件缺少顶部说明时，本次修改必须一并补齐，不能以“不是本次功能的一部分”为由跳过。

顶部说明至少包含：

- **文件用途**：该文件解决什么问题、服务于哪条运行或研究流程；
- **主要职责**：文件内部承担哪些职责，明确不承担哪些职责；
- **关键输入与输出**：重要配置、数据结构、返回值、持久化产物或外部接口；
- **依赖与副作用**：关键模块依赖、文件读写、网络访问、随机性、全局状态等；
- **重要约束**：数据泄漏边界、调用顺序、不变量、兼容性和修改风险。

根据编程语言使用原生注释或文档字符串：

| 代码类型 | 顶部说明形式 |
|---|---|
| Python | 模块级中文 docstring，放在 shebang 和编码声明之后、import 之前 |
| PowerShell、Shell | 文件顶部连续 `#` 中文注释；shebang 必须保持第一行 |
| JavaScript、TypeScript、C/C++、Java 等 | 文件顶部块注释，放在语言要求的固定声明之后 |

Markdown、LaTeX、YAML、JSON、锁文件、普通配置、数据文件、模型权重、图片、PDF 和其他非代码文件不要求添加顶部用途说明。自动生成代码和第三方 vendored 代码也不得为了满足本规范而手工改写。

### 9.2 中文代码注释

所有新建或修改的实现必须提供充分、准确的中文注释或中文 docstring。注释重点解释“为什么这样做、受什么约束、失败时会怎样”，不能只把代码逐行翻译成自然语言。

以下位置必须详细注释：

- 模块、公开类、公开函数和重要内部函数的职责、参数、返回值、异常与副作用；
- 非显然算法、数学公式、张量/数组形状变化和数值稳定性处理；
- 配置优先级、注册表解析、状态机转移、缓存、并发和生命周期逻辑；
- 数据切分、正常数据限定、校准、阈值拟合等可能造成数据泄漏的位置；
- 文件系统、网络、Zotero、追踪后端和可选依赖的失败路径；
- 兼容旧格式、性能优化或看似可简化但实际不能改动的代码。

测试文件顶部说明测试范围、关键场景和不覆盖的风险；复杂 fixture、参数化数据和回归用例需说明其业务含义。示例脚本说明它是 smoke、演示还是论文实验，并明确其输出能否作为论文结果。

### 9.3 注释质量与维护

- 中文注释保留必要的英文标识符、公式和标准术语，不生硬翻译接口名。
- 注释必须与代码同步更新；行为改变时同时修改顶部说明、docstring 和相关行内注释。
- 禁止保留与实现冲突、已经过期或描述未来计划为既成事实的注释。
- 避免“给变量赋值”“调用函数”这类无信息量注释；详细不等于重复代码。
- 注释不能代替清晰接口、类型标注、验证和测试。复杂逻辑应先改善结构，再用中文说明关键设计理由。
