# 研究构思评审与论文写作

[简体中文](README.md) | [English](README.en.md)

当前版本：**5.0.0**

本技能采用四阶段闭环：idea 审查、论文构思、非实验技术正文写作、科研图件与实验结果整合。每个阶段先用“概括词语 + 简述”说明 loop，再由对应阶段文件展开细则。

## 目录

- [四阶段总览](#四阶段总览)
- [Baseline ID、revision 与 FROZEN](#baseline-idrevision-与-frozen)
- [Stage 1 loops](#stage-1-loops)
- [Stage 2 loops](#stage-2-loops)
- [Stage 3 loops](#stage-3-loops)
- [Stage 4 loops](#stage-4-loops)
- [科研图构图规则](#科研图构图规则)
- [参考图案例库](#参考图案例库)
- [文件职责与选读](#文件职责与选读)
- [验证与版本](#验证与版本)

## 四阶段总览

| 阶段 | 核心任务 | 主要产物 | 明确不做 |
|---|---|---|---|
| Stage 1：idea 审查 | 查重合、创新差量、任务/信息边界、假设、理论和落地性 | `idea-assessment.md`、论文 case | 不把未确认想法写成论文结论 |
| Stage 2：论文构思 | 定调背景任务、贡献、技术主线、模型定义顺序、符号、名词、章节和叙事 | revision 文件夹内唯一的 `MANUSCRIPT_CONTEXT.md` | 不写最终正文、实验结果或正式图件 |
| Stage 3：论文写作 | 依据冻结 context 写非实验技术正文、公式、证明、方法和 LaTeX 简图 | `manuscript.tex/.pdf`、`WRITING_LOOP_LOG.md`、LaTeX 简图 | 不写实验结果，不制作最终科研图 |
| Stage 4：图件与实验 | 制作科研图、导入外部 Python 数据图、写实验描述并完成全文整合 | 可编辑图源、实验文字、最终 PDF | 不改造技术方法，不编造数据图 |

```text
Stage 1 confirmed idea
  -> Stage 2 FROZEN <Baseline ID, revision>
  -> Stage 3 non-experimental technical draft + LaTeX schematics
  -> Stage 4 scientific figures + actual-result narration + final delivery
```

## Baseline ID、revision 与 FROZEN

- `Baseline ID`：同一篇论文构思谱系的稳定、可读名称，例如 `koopman-fault-diagnosis`。只要仍是同一篇论文就不变。
- `Context revision`：该构思的版本号。第一次冻结为 `1`，每次修改冻结内容并重新冻结时加一。
- `FROZEN`：表示当前 revision 已通过 Stage 2 全部检查，可以作为下游写作依据。
- `<Baseline ID, revision>`：唯一指向一次具体构思快照，例如 `<koop-fd, 3>`。
- `Baseline ID` 应短而可辨认，通常为一至三个词或公认缩写，建议不超过 16 个字符，硬上限为 24 个字符。
- revision 文件夹：`<baseline-id>-r<revision>`，例如 `koop-fd-r3/`。
- 如果中心任务和贡献结构已经变成另一篇论文，应新建 Baseline ID 和论文 case，而不是继续增加原 revision。

版本身份只放在文件夹名中；内部采用 `MANUSCRIPT_CONTEXT.md`、
`manuscript.tex/.pdf`、`WRITING_LOOP_LOG.md`、
`STAGE4_FIGURE_EXPERIMENT_LOG.md` 等短职责名。详细契约见
[artifact-naming.md](references/artifact-naming.md)。日志必须同时绑定 ID、
revision 和 revision 文件夹，防止拿旧版构思的 `PASS` 继续使用。

## Stage 1 loops

详细规则：[idea-exploration.md](references/stages/idea-exploration.md)

| ID | 概括词语 | 简述 |
|---|---|---|
| I1 | 现有工作重合 | 最接近工作是否已完成相同构造、机制或结果 |
| I2 | 任务边界清楚 | 解决什么、不解决什么，哪些信息真实可得 |
| I3 | 假设能够落地 | 训练、计算、在线使用和验证是否能实现 |
| I4 | 理论并非平凡 | 是否产生依赖模型结构的可证明、可计算或可检验结果 |
| I5 | 贡献主线集中 | 两到三个问题—贡献主题是否集中且相互衔接 |
| I6 | 成熟度不夸大 | 区分 confirmed、exploratory、alternative、rejected 和 unresolved |

## Stage 2 loops

Stage 2 的所有规则只作用于当前
revision 文件夹内的 `MANUSCRIPT_CONTEXT.md`。详细规则：[paper-conception.md](references/stages/paper-conception.md)

| ID | 概括词语 | 简述 |
|---|---|---|
| C1 | 背景与任务定调 | 先讲为什么研究，再说论文解决什么 |
| C2 | 贡献逐项对应 | 中心问题、主要贡献、结果和证据需求一一对应 |
| C3 | 技术主线成链 | 问题—需要—构造—机制—结果—任务作用连成主线 |
| C4 | 模型定义有序 | 理论到设计、整体到局部、前向到损失、设计到应用 |
| C5 | 符号提前规划 | 预登记含义、类型、字体、维度和首次解释位置 |
| C6 | 名词提前规划 | 使用领域专有名词，合理缩写，不杜撰术语 |
| C7 | 章节先后明确 | 按科学依赖安排章节，禁止按模块数量机械拆分 |
| C8 | 行文逻辑预演 | 预先规定每节/每段的目的、因果关系和宏观到微观推进 |
| C9 | 基线交叉冻结 | 定位、技术、符号、名词、结构和叙事无冲突后冻结 |

```text
DRAFT_CONTEXT -> CHECK -> REVISE -> CHECK -> FROZEN
```

## Stage 3 loops

详细规则：[manuscript-writing-loop.md](references/manuscript-writing-loop.md)

| ID | 概括词语 | 简述 |
|---|---|---|
| W1 | 基线一致性 | 正文不能静默改变冻结构思 |
| W2 | 章节先后性 | 理论到模型、整体到局部、前向到损失、设计到应用 |
| W3 | 叙事逻辑性 | 语句/段落真实衔接，先因或目的再给响应和结果 |
| W4 | 表达忌生硬 | 避免陈述句堆砌，按逻辑自然改变句式 |
| W5 | 语句忌空洞 | 每句每段有必要功能，不写无意义背景、重复或凑字数内容 |
| W6 | 符号一致性 | 首处解释、同义同符号、同类同字体、一符一义 |
| W7 | 名词专有化 | 标准名词、必要缩写、首处全称，不杜撰名词/缩写 |
| W8 | 公式严谨性 | 定义、维度、索引、条件、推导和结论范围完整 |
| W9 | 示图完整性 | 需要时给出原理、模型结构、任务流程的 LaTeX/TikZ 简图 |
| W10 | 全文对齐性 | 问题、贡献、正文、简图、摘要和结论保持一致 |

“模型描述完整性”和“训练/验证/测试/部署清晰性”不再作为 Stage 3 loop 标题。前者转化为 Stage 2 的模型定义顺序和 Stage 3 的章节/公式规则；后者的实验性内容转入 Stage 4。

```text
DRAFT -> CHECK -> PASS -> NEXT
DRAFT -> CHECK -> FAIL -> REVISE -> CHECK
CONTEXT_DEVIATION -> STAGE 2 REVISE/REFREEZE -> INVALIDATE -> CHECK
```

## Stage 4 loops

详细规则：[figures-and-experiments.md](references/stages/figures-and-experiments.md)

| ID | 概括词语 | 简述 |
|---|---|---|
| F1 | 证据输入就绪 | 没有真实结果和稳定简图时不开始最终制作和结果叙述 |
| F2 | 图型选择匹配 | 区分原理图、模型结构图、任务流程图和定量数据图 |
| F3 | 大框套小框 | 有标题的大框组织阶段/模块，小框表达必要设计细节 |
| F4 | 箭头单向无环 | 从形状指到形状，可跨大框，无悬空、双向和视觉闭环 |
| F5 | 图形学术化 | 形状/icon 克制、可编辑、有授权，禁止海报或商务风格 |
| F6 | 信息不过载 | 模块完整但不冗余，限制图中文字和公式 |
| F7 | 图文符号同步 | 图中符号、术语和字体与 context/正文一致 |
| F8 | 参考来源清楚 | 检索顶刊/顶会并记录来源、借鉴边界和版权状态 |
| F9 | 配色可读一致 | 每图一条配色，冷灰/冷色为主，亮色只承担一种语义 |
| E1 | 数据图结果一致 | 数据图来自外部专门 Python 库及真实结果，不手工改点 |
| E2 | 看图描述有据 | 说明检验目标、观察、比较、机制解释和边界 |
| E3 | 实验总结克制 | 不把趋势写成证明、因果性、显著性或普适性 |
| D1 | 图注正文闭合 | 图、图注、实验文字、摘要和结论相互对应 |

## 科研图构图规则

完整规则：[figure-composition-rules.md](references/figure-composition-rules.md)

共同要求：图内不放整张图标题；每个大框必须有标题；箭头单向、视觉无环且从形状指向形状；允许跨大框；术语和符号必须与正文一致。

### 模型结构图

- 大框表示整体模型模块或损失/优化模块；
- 小框表示具体变换、网络层、输入、输出和接口；
- 顺序为输入—整体前向—局部细节—任务输出，损失放在其所消费变量之后或旁侧；
- 可用局部放大和重复层堆叠表达贡献相关细节。

### 任务流程图

- 大框表示数据采集与预处理、离线建模、在线监测/应用等 step；
- 每个 step 内部由单向箭头连接的形状组成；
- 明确区分离线和在线信息，不能把测试标签或未来信息放入在线流程。

### 原理图

没有固定模板。必须检索同领域当前权威论文，判断别人使用几何关系、信息分离、状态演化、变换、边界等哪种方式表达原理，再针对本论文独立构图。

真正的递归/反馈通过时间或迭代展开、重复模块或更新公式表示，不能画视觉闭环，也不能为了无环而删除真实递归关系。

## 参考图案例库

入口：[cases/figure-exemplars/README.md](cases/figure-exemplars/README.md)

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

案例边界是上述三种图类，不是单张参考图。每一类在同一个
`references.md` 中用一行记录一张参考图；只有获得授权的图片才进入该类的
`images/`。无权本地保存的图片只记录论文、DOI/URL、图号、构图分析和版权状态。

配色资产集中在 [`assets/palettes/`](assets/palettes/)，权威色值和使用规则在 [`references/scientific-figure-palettes.md`](references/scientific-figure-palettes.md)。

学术 icon 资产集中在 [`assets/icons/`](assets/icons/README.md)。可从
[Alibaba Iconfont](https://www.iconfont.cn/) 和
[EmojiAll](https://www.emojiall.com/zh-hans) 检索，但必须逐项核对作者、许可和
目标出版用途，并登记到 `icon-registry.md`。动态页面、登录态或交互下载需要时
使用 Codex Chrome 浏览器插件；网站可下载不等于自动获得论文出版许可。

## 文件职责与选读

| 文件/目录 | 唯一职责 | 读取时机 |
|---|---|---|
| `SKILL.md` | 四阶段简要路由 | 每次触发 |
| `references/stages/idea-exploration.md` | Stage 1 loops | idea/理论/可实现性审查 |
| `references/stages/paper-conception.md` | Stage 2 loops | 建立或修改 context |
| `references/stages/manuscript-writing.md` | Stage 3 成文 | 非实验技术写作 |
| `references/manuscript-writing-loop.md` | Stage 3 W1–W10 | 每次实质写作修改 |
| `references/stages/figures-and-experiments.md` | Stage 4 loops | 图件、实验结果和最终整合 |
| `references/figure-composition-rules.md` | 科研图详细构图 | Stage 4 概念图件 |
| `references/artifact-naming.md` | 短 ID/revision 文件夹与内部职责名契约 | Stage 2–4 |
| `references/detail-preservation-and-refactoring.md` | skill 重构时无细节损失、去重和迁移证据 | 维护 skill 时必读 |
| `<short-baseline-id>-r<revision>/MANUSCRIPT_CONTEXT.md` | 唯一构思基线 | Stage 2 创建，Stage 3/4 必读 |
| `<revision-folder>/WRITING_LOOP_LOG.md` | Stage 3 检查证据 | Stage 3 |
| `<revision-folder>/STAGE4_FIGURE_EXPERIMENT_LOG.md` | Stage 4 图件、数据和整合证据 | Stage 4 |
| `cases/<task-group>/` | 单篇论文历史/写作范本 | 准确匹配论文任务时 |
| `cases/figure-exemplars/<三类之一>/` | 图类案例和多张参考图清单 | 准确匹配图型时 |
| `assets/icons/` | 已核验许可的可复用 icon 和登记表 | Stage 4 制图时 |

## 验证与版本

```powershell
python -X utf8 scripts/validate_skill.py .
python -X utf8 scripts/audit_artifact_names.py <revision-folder>
python -X utf8 scripts/audit_writing_loops.py <project-root>
python -X utf8 scripts/audit_manuscript.py <project-root>
python -X utf8 scripts/audit_figures.py <figure-root>
```

[`tests/forward-tests.json`](tests/forward-tests.json) 覆盖四阶段进入条件、Stage 3 十项 loop、Stage 4 图件类型和实验结果边界、符号/术语冲突以及案例迁移。

纯文本 [`VERSION`](VERSION) 是语义版本来源。7.0.0 将短 Baseline
ID/revision 放入 revision 文件夹名，内部改用简短职责名，并加入
`skill-creator` 重构时的无细节损失契约。只有用户明确要求发布时才提交、打标签或推送。
