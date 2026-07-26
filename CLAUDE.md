# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

一个仓库、两条线：

1. **Joff**（`src/joff`）：spec-first 的 PyTorch 过程数据实验工具包（配置 → 数据 → 模型 → 训练 → 评估 → 产物）。
2. **论文项目**：受保护 Attention–Koopman–T–S 模糊故障检测论文。Joff 是它的实验底座；论文方法模块**尚未实现**，模块落点见 `docs/技术架构与代码地图.md` 第 12 节与 `docs/论文方法完整设计.tex` 第 21 节。

项目文档全部为中文，唯一入口是 `docs/文档导航.md`。`docs/` 下 Markdown 用中文文件名；`README.md`、`AGENTS.md`、`CLAUDE.md` 保留标准英文名。协作代理约定（事项追踪器、分诊标签、技能目录同步）见 `AGENTS.md`。

## 常用命令

```bash
# 安装（可编辑 + 常用可选依赖）
python -m pip install -e ".[dev,paper,hdf5]"

# 测试（pytest 配置在 pyproject.toml：testpaths=tests，pythonpath=src，默认 -q）
python -m pytest                                   # 全套（9 个文件共 128 个测试）
python -m pytest tests/test_phase1_core.py -q      # 快速核心测试（约 16 秒）
python -m pytest "tests/test_phase1_core.py::test_unknown_config_key_raises_helpful_error" -q  # 单个测试

# Lint（line-length=100，target py310，排除 datasets/raw）
python -m ruff check .

# 冒烟示例（合成/预置数据，只验证代码路径；数值不是论文结果）
python examples/fd_cstr.py --smoke
python examples/data_preset_cstr_fd.py

# 编译 docs/ 下中文 LaTeX（MiKTeX + latexmk；中间文件放 %TEMP%，只拷回 PDF 并打印质量检查）
pwsh docs/编译PDF.ps1
```

换机器后若 ctex 报 `Support package 'expl3' too old`，先执行 `mpm --update`。

## 架构（大图）

主调用链：`配置解析(core) → 实验编排(experiments) → 数据(data) → 模型构建(models/layers) → 训练(training) → 评估(evaluation) → 产物(artifacts)`；`plotting`/`tracking`/`console`/`xai` 是支持模块。分层职责、扩展接口与核心文件地图见 `docs/技术架构与代码地图.md`（唯一权威架构文档，此处不重复）。

改代码时必须维持的不变量：

- 严格 Pydantic 配置：未知字段直接报错；每次解析产生字段 provenance 与 16 位配置哈希。
- 拟合型预处理（归一化、异常值处理等）只能从训练 split 学习参数。
- 模型只实现 `forward(batch)`（返回 Tensor 或 dict）与 `compute_loss`，不创建运行目录；Trainer 不读原始数据文件。
- `import joff` 无副作用（不读数据、不建目录、不改 Matplotlib 全局状态、不启动 tracking）。
- 新模型/评估器/数据集走显式注册表（`register_model` / `register_evaluator` / `DatasetAdapter` + registry），由 `core.factory` 按 `type` 构建，不用动态 import。
- 故障检测评估器遵循 `fit(normal_scores)` / `evaluate(test_scores, labels)` 两阶段接口，正常拟合与在线评价在 API 上分开。

数据两条进入路径：preset（`DatasetRegistry` → adapter → `CanonicalDataset`；OA 原始数据在 `datasets/raw/oa/**`，卡片在 `datasets/cards/oa/**`）或直接文件路径（source readers）。private 数据一律不入库（.gitignore 已强制）。

## 论文项目的硬约束

- **仅正常数据**：模型训练、结构与超参选择、阈值校准只能用正常数据；故障数据只能在协议冻结后作最终测试。任何实验代码不得违反——禁止用故障数据调参、拟合阈值、选择模型或窗口。
- 数据划分需要"正常训练 / 正常验证 / 正常校准 / 冻结故障测试"四分边界（现有 `DataModule` 只有 train/test 抽象，需正式扩展，勿在脚本里临时切数组）。
- 闭环 CSTR（preset `cstr_closed_loop_fd`）已知坑：适配器把测试段整段标为故障（真实 onset=200，须逐行标签）；u/y 角色（u=Ci,Ti,Tci；y=C,T,Tc,Qc）只在数据说明 txt 中、未进 schema；数据许可 `to_verify`。
- 对老师理论稿的两处 critical 修正（推论 1 改校准可观测超额得分、隔离层残差—响应矩阵配对）已写入设计文档但**尚未经导师确认**，实现时以 `docs/论文方法完整设计.tex` 第 16 节的修正清单为准。

## 文献检索与 Zotero 工作流（强制）

**铁律：任何进入论文的文献，必须先入 Zotero，再被引用。** 严禁直接凭模型记忆写参考文献——`.bib` 条目只能来自 Zotero 导出或权威数据库返回的元数据，不得由模型自由生成。

标准流程（四步，不可跳步）：

1. **检索**：用 `/paper-triage`（模糊查询 → 排序候选）、`/expand-references`（1–3 篇种子论文 → 分类扩展）、`/trace-citations`（引用网络前向/后向追踪）。这三个技能走 Semantic Scholar，返回带 ID 的结构化 JSON。仅当需要项目主页、代码仓库、数据集、会议通知这类非论文信息时才用 Tavily。
2. **入库**：确认相关的论文立刻用 `zotero_add_by_doi`（或 `zotero_add_by_url` / `zotero_add_by_bibtex`）写进 Zotero，并打好集合与标签。**未入库的论文不得进入正文、相关工作或 `.bib`。**
3. **精读**：读文献一律经 Zotero MCP（`zotero_search_items` / 全文 / PDF 标注），不要重复下载或重复上传 PDF。
4. **引用**：`.bib` 从 Zotero 导出（BibTeX）。投稿前跑一次引用核验，确认标题/作者/年份/DOI 与数据库一致、无编造条目、预印本与正式出版版本未混淆。

写作时若发现某条引用在 Zotero 中查不到，正确处理是回到第 1 步补检索并入库，而不是照着印象把条目补进 `.bib`。

集合按论文结构划分（`01-综述与基础` / `02-Koopman` / `03-T-S模糊` / `04-Attention与编码` / `05-故障检测` / `06-故障隔离` / `07-正常数据建模` / `08-实验与数据集` / `09-待精读` / `10-已引用`）。

工具落点与凭据：MCP 服务 `zotero` 在 `.mcp.json`（Claude Code）与 `~/.codex/config.toml`（Codex）各配一份；读走本地 API（`ZOTERO_LOCAL=true`，需 Zotero 客户端运行且在"设置 → 高级"里允许其他应用通信），写入走 Web API，需要用户级环境变量 `ZOTERO_API_KEY` 与 `ZOTERO_LIBRARY_ID`。**密钥只放系统环境变量，绝不写进仓库任何文件。** 检索技能在无 `SEMANTIC_SCHOLAR_API_KEY` 时共享匿名配额，容易 429；持续检索前先设该变量。

## 文档与交付约定

- 新增或重命名文档必须登记进 `docs/文档导航.md`；改变研究判断的里程碑写进 `docs/当前工作记录.md`（按其自带模板追加，勿逐次运行都改）。
- 方法/理论类交付物用中文 LaTeX（ctexart + XeLaTeX，定理环境用中文名）；交付前必须本地编译通过（`pwsh docs/编译PDF.ps1`），PDF 与 `.tex` 一起入库、随实质修改同批提交。
- 写 `.tex` 一律用 Write/Edit 工具，不要用 bash heredoc（shell 转义会吞 `\`，制造假语法错误）；中文引号用全角（ASCII 直引号 `"` 会被 TeX 排成右引号）。
