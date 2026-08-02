# Section Role Matrix

## Manuscript

- Working title: Measurement-Decoupled Attention–Koopman–T–S Fuzzy Monitoring via
  Structured Basis-Perturbation Residual Learning
- Target journal: IEEE Transactions on Fuzzy Systems
- Main source:
  `teacher_latest_manuscript/Measurement_Decoupled_Attention_Koopman_TS_Fuzzy_Monitoring_Skill_Revised(1).pdf`
  (SHA-256:
  `e47deefbdee57d2f62ea5de2eb107f25bb7cac52d4ac8830c30f23c252a6e090`)
- Last synchronized: 2026-07-31（第二次全文重写：基于老师最新版优化语言表达与行文逻辑，保持理论内容不变，经三轮子代理审核与13项修复）

## Chapter arrangement conformance

- Skill default sequence: Introduction; Preliminaries and Problem Formulation;
  Proposed Method; optional genuinely independent second task; Experiments;
  Conclusion.
- Actual top-level sequence: Introduction; Preliminaries and Problem Formulation;
  Proposed Method; Input-Conditioned Detection and Set-Valued Isolation; Conclusion;
  Appendices; References.
- Target-journal or scientific justification for each deviation: Section IV is the
  genuinely independent second task that consumes the residual constructed in Section
  III and produces the detection/isolation decision. The teacher manuscript contains no
  experiment section because V2 software and frozen fault evaluation do not yet exist;
  adding results or an experiment-design section would exceed the authorized
  language-and-logic-only revision and could turn a planned protocol into claimed
  evidence. This omission remains an explicit readiness limitation rather than a hidden
  completed result.
- Status: PASS（章节次序符合依赖关系；实验缺失作为证据状态限制保留）

## Problem–contribution alignment

Use one row per central subproblem. Keep the problem and contribution counts
identical and preserve their order.

| Problem ID | One central subproblem | Contribution ID | Problem-facing construction | Explicit result or output | Body location | Validation or evidence | Status |
|---|---|---|---|---|---|---|---|
| P1 | 在新测量仍用于残差评价时，怎样阻止其继续修正正常参考，并让训练、短视野和长参考使用同一残差对象？ | C1 | 严格因果 attention–Koopman–T–S 参考、统一 start-indexed 残差和延迟锚点确认 | 精确非线性残差递推与条件式锚点接受命题 | III-A、III-B、Theorem 1、Proposition 1 | 老师稿公式 (4)–(20) 与 Appendix A；尚无 V2 实验 | PASS |
| P2 | 怎样用适合一阶网络训练的量控制完整 attention–Koopman–T–S 映射的有限视野正常传播？ | C2 | 完整模糊提升状态 Jacobian 分解、逐元素硬参数化和无穷范数递推 | 可实现 Jacobian 上界、有限视野正常残差界和可靠参考视野 | III-C、Theorem 2、Assumption 1、Corollary 1 | 老师稿公式 (21)–(27) 与 Appendix B；界的 V2 软件尚未实现 | PASS |
| P3 | 在没有物理故障方向和故障样本时，怎样学习可重复的模型坐标响应并只在证据支持时输出唯一组？ | C3 | `u/y/z` 结构化 basis rollout、工况条件标准化、多尺度分数与模式库 | 残差空间检测充分条件、唯一组或兼容集合输出 | III-D、IV-A 至 IV-D、Propositions 2–3 | 老师稿公式 (28)–(42)；正式故障评价仍受 V2 实现与数据许可限制 | PASS |

## Section responsibilities

Use one row per section or subsection. Assign one primary question and one
reader-facing output. State the causal transition into the section and whether
its location conforms to the skill sequence.

| Section / subsection | Single primary question | Required input | Reader-facing output | Problem ID | Contribution ID | Depends on | Narrative dependency / causal transition | Skill-sequence conformance or justified deviation | Status |
|---|---|---|---|---|---|---|---|---|---|
| Abstract | 在 150–220 词内怎样按相同顺序概括信息边界、三个构造、条件结果和证据范围？ | 最终正文与三条主线 | 问题–方法–结果–范围摘要 | P1–P3 | C1–C3 | 全部正文 | 由最终正文反向压缩，不增加正文没有的保证 | 符合 | PASS |
| Introduction | 为什么 normal-data-only 监测形成一个相互依赖的三问题设计，而不是模块清单？ | 领域背景、信息边界和正文结果 | 技术张力、已有路线限制、三个贡献及全文路线 | P1–P3 | C1–C3 | 摘要最终回写 | 从闭环异常测量污染参考的矛盾，推进到传播与残差空间决策 | 符合；不得展示公式 | PASS |
| II. Preliminaries and Problem Formulation | 在什么信息边界和标准背景下，本文的三个问题才是可计算且互相对齐的？ | 物理系统、可用在线量、正常数据纪律、Koopman 标准概念 | 信息边界、有限历史提升表示和三个编号问题 | P1–P3 | C1–C3 | Introduction gap | 先定义可用信息，再给标准提升背景，最后提出三个问题 | 符合 | PASS |
| II-A. Information Boundary and Plant Description | 在线可测、条件已知和不可用的物理量分别是什么？ | 闭环系统与三类故障入口 | Plant mapping、控制命令与实际作用区别、正常-only 数据约束、严格过去历史 | P1–P3 | C1–C3 | Introduction | 信息可得性决定后续参考和残差不能使用的量 | 符合；只给问题背景 | PASS |
| II-B. Finite-History Lifted Representation | 有限历史学习坐标与理想 Koopman observable 之间是什么受限关系？ | 标准 Koopman composition | 不假设有限不变子空间或物理状态包含关系的提升表示 | P1–P2 | C1–C2 | II-A | 因真实状态和精确 observable 不可用，转向有限历史预测坐标 | 符合；不放论文专用 encoder 结构 | PASS |
| II-C. Problem Formulation | 前述信息边界具体产生哪三个中心问题？ | II-A 与 II-B | 三个与贡献顺序一致的编号子问题 | P1–P3 | C1–C3 | II-A、II-B | 由可用信息和表示限制自然导出 reference、propagation、decision | 符合 | PASS |
| III. Proposed Method | 怎样依次构建残差对象、控制正常传播并学习结构化响应？ | II 的三个问题 | 完整离线模型、参考、传播界和 basis 学习方法 | P1–P3 | C1–C3 | II-C | 按 P1→P2→P3 的因果顺序实现问题定义 | 符合 | PASS |
| III-A. Attention–Koopman–T–S Normal Model | 怎样用严格过去历史形成可自由 rollout 的模糊 Koopman 正常模型？ | 严格过去历史、控制命令、外生量 | causal attention encoder、metric T–S memberships、local/interpolated model、decoder、硬参数化与自由 rollout loss | P1–P2 | C1–C2 | II-B、II-C | 先给总体正常映射，再拆 encoder、membership、local model 和训练细节 | 符合 top-down 顺序 | PASS |
| III-B. Measurement-Decoupled Reference and Unified Residual | 怎样从同一 rollout 对象得到短视野与长参考残差而不回注新测量？ | III-A 的冻结模型与自由 rollout | start-indexed residual、delay-confirmed anchor、joint residual、精确传播递推 | P1 | C1 | III-A | 因训练与部署必须共享残差，先统一起点，再定义 anchor 生命周期和递推 | 符合 | PASS |
| Proposition 1 | 延迟确认在什么条件下拒绝故障发生后的锚点候选？ | warning delay 假设 | 条件式 delayed-anchor acceptance 结论 | P1 | C1 | III-B | 锚点规则需要明确其可覆盖和不可覆盖范围 | 置于构造后、传播前 | PASS |
| Theorem 1 | start-indexed 残差的精确非线性递推是什么？ | 可微 transition、统一残差 | 一步递推与有限视野展开 | P1 | C1 | III-B | 统一残差后才能分析 normal propagation | 置于对应构造之后 | PASS |
| III-C. Fuzzy-Structure-Dependent Normal Propagation | 完整 T–S 模糊结构怎样影响正常参考误差传播？ | Theorem 1、III-A 的 membership 与 local model | 完整 Jacobian 分解、无穷范数上界、forcing 假设、递推界和可靠视野 | P2 | C2 | III-A、III-B、Theorem 1 | 因测量解耦暴露自由 rollout drift，必须量化完整 fuzzy mapping | 符合 | PASS |
| Theorem 2 | 完整模糊提升状态 Jacobian 如何分解且怎样得到可实现上界？ | local dynamics、normalized memberships、scheduling map | local-dynamics term 与 membership-variation term 的精确分解及 bound | P2 | C2 | III-C definitions | 先解释为什么局部矩阵加权不足，再给完整导数 | 置于动机后 | PASS |
| Assumption 1 and Corollary 1 | 在有限正常事件上，forcing 与 Jacobian bound 怎样形成可靠参考视野？ | Theorem 1、Theorem 2、held-out normal calibration | 条件式有限视野残差界与可靠 horizon | P2 | C2 | Theorems 1–2 | 结构上界不能覆盖有限数据中的全部 remainder，因此分离校准接口 | 置于结构结果后 | PASS |
| III-D. Structured Basis-Perturbation Residual Learning | 仅减小 normal drift 不足时，怎样在无物理故障样本下学习可分离响应？ | III-B residual、III-C normal-side bound | `u/y/z` basis rollout、scaled sensitivity、unit-L1 pattern 与联合损失 | P3 | C3 | III-B、III-C | 先指出 normal-side objective 的不足，再引入 diagnostic-side signal | 符合 | PASS |
| IV. Input-Conditioned Detection and Set-Valued Isolation | 冻结 residual 和 response 后怎样形成在线检测与诚实的集合值决策？ | III 的 residual、bound 和 patterns | condition-aware standardization、multiscale threshold、detection 与 group decision | P3 | C3 | III-D | 由学习对象转入独立决策任务 | 合理的独立第二任务 | PASS |
| IV-A. Input-Conditioned Residual Standardization | 工况、命令变化、fuzzy region 与 reference age 怎样进入正常尺度和阈值？ | joint residual、normal calibration blocks | mixture mean/scale、standardized residual、multiscale score 与 dynamic threshold | P3 | C3 | III-B、III-C | 因 raw residual 分布随工况变化，先冻结条件尺度再比较分数 | 符合 | PASS |
| IV-B. Sufficient Detection Condition | 为什么 normal propagation 与 structured response learning 必须同时成立？ | IV-A threshold、III-D learned response | residual-space detection sufficient condition 与 amplitude implication | P3 | C3 | III-D、IV-A | 将较低 normal threshold 与较高 learned response 合并为可检验条件 | 符合 | PASS |
| IV-C. Unique-or-Set-Valued Isolation | 何时可以输出唯一组，何时必须保留兼容集合？ | standardized observed pattern、stored response patterns | compatible set、unique margin 与 sufficient group separation | P3 | C3 | IV-A、IV-B | 检测后再讨论证据强度，避免把匹配强制转成物理 singleton | 符合 | PASS |
| IV-D. Offline and Online Algorithms | 训练、校准和部署分别访问什么数据、冻结什么对象、执行什么计算？ | III 与 IV-A 至 IV-C | 可复现的 offline/online algorithms 和计算边界 | P1–P3 | C1–C3 | 全部方法 | 将分散定义收束为阶段化实现，防止数据泄漏和在线重训练 | 符合 | PASS |
| V. Conclusion | 在不扩大老师稿结论的前提下，三条主线最终得到什么、仍不支持什么？ | 全部正文 | 与摘要、问题、贡献同序的结论和物理解释限制 | P1–P3 | C1–C3 | 全部章节 | 汇总已推导结果并明确没有 unique physical fault identification | 符合 | PASS |
| Appendix A | Theorem 1 的关键等式怎样由加减项和 integral mean-value identity 得到？ | Theorem 1 | 完整但紧凑的 proof | P1 | C1 | III-B | 正文给结论，附录保留证明细节 | 符合 | PASS |
| Appendix B | Theorem 2 的 Jacobian 分解和 induced-norm bound 怎样得到？ | Theorem 2 | differentiating membership interpolation 后的 proof | P2 | C2 | III-C | 正文给结构意义，附录保留代数证明 | 符合 | PASS |
| References | 哪些经核验来源支撑标准背景、先行方法和有限样本解释？ | 老师稿 15 条 reference 与现有 BibTeX 元数据 | 与 citation key 一致的 bibliography | P1–P3 | C1–C3 | 全文 citations | 只保留老师稿来源，不新增文献论点 | 符合；正文/BibTeX 一致性已通过，Zotero 入库状态因接口不可用未核验 | PASS（正文一致性；Zotero 核验未闭环） |

## Subsection writing-loop record

Use one row after every substantive subsection draft or revision. Record
evidence and revisions; do not enter a bare `PASS`.

| Subsection | Chapter arrangement | Sentence-to-sentence logic | Narrative causality | Symbol consistency | Formula rigor | Model completeness | Training / validation / testing / deployment clarity | Evidence inspected and revision action | Last checked | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Abstract | 位于标题后且作为全文反向压缩；顺序与 P1→P2→P3 一致 | 单段依次承担问题、三类构造、条件结果和范围限定，代词均有明确指向 | 从 measurement coupling 推到 decoupled reference、propagation bound、basis response 与 set-valued decision | 未引入正文外符号；术语与 ledger 的 reference、residual、group 含义一致 | 无展示公式；每项结果可回指 (13)–(42)，未扩大充分条件 | 同时覆盖严格过去 history、冻结 rollout、joint residual、threshold 和 compatible set | 明示 normal-data-only 且不声称实验结果、物理 fault singleton 或已完成验证 | 对照老师 PDF 摘要、`main.tex` Abstract、P1/C1–P3/C3 和 docs/03；Claude 重写后复核词数约 178 | 2026-07-30 | PASS |
| Introduction | 背景→现有路线限制→三重张力→三项贡献→全文路线，未提前放公式 | 段首分别命名 normal-only 困难、标准背景、传统 fault-model 限制、三目标耦合和贡献，句式含主被动、从句与非谓语结构 | 异常测量污染 reference 是起因，free-rollout drift 与诊断灵敏度冲突随后导出三项设计 | Koopman、T–S、reference、residual 等沿 ledger 使用；`u/y/z` 未在此提前物理化 | 无新公式；贡献中的“exact”“sufficient”均保留对应条件和局部范围 | 说明可测命令/输出与未知物理 fault 信息，贡献按模型→传播→决策闭合 | 只陈述 normal-only 训练事实，不写硬件、数值或完成的 frozen test | 对照 `main.tex` Introduction、老师稿引言与 15 个 citation key；删除模块清单式叙述并重排为 gap→need→result | 2026-07-30 | PASS |
| II-A. Information Boundary and Plant Description | 只给标准 plant、在线信息边界、normal-only 纪律和严格过去 history，论文专用构造留到 III | 各段先说明对象或限制，再给式 (1)–(2) 及其后果；`u_k` 与 `u_k^a` 的指代持续明确 | 因实际 action 和 fault direction 不可得，模型只能条件化于记录命令；因此历史排除当前输出 | 核对 ledger 中 $u,u^a,x,y,\xi,f^a,f^p,f^s,\mathcal H_k^-$ 的类型、维数和首次定义 | 式 (1)–(2) 的索引、维数、加性 fault 入口和严格过去边界与老师稿一致 | 输入、物理状态、测量、外生量、未知映射、噪声及不可用量均明确 | 明示训练与 threshold calibration 仅用正常轨迹，模型和 calibration quantities 在测试冻结 | 对照老师 PDF (1)–(2)、`main.tex` II-A、ledger 与 docs/03 数据纪律；修订命令/实际作用混淆 | 2026-07-30 | PASS |
| II-B. Finite-History Lifted Representation | 位于信息边界之后、Problem Formulation 之前，只提供受限 Koopman 背景 | 先给理想 observable 关系，再否定有限不变子空间和物理状态嵌入假设，最后转到 finite-history encoder | 真实状态与精确 observable 不可得是采用预测性有限历史坐标的原因 | $\psi,\mathbb Z,m_z,z_k^{\rm enc}$ 已按 observable/set/vector 家族登记且未复义 | 式 (3) 的 composition、域和值域完整；learned coordinate 只承诺近似 (3) | 区分 ideal observable、learned lifted coordinate 与物理 state | 不涉及训练结果；只限定后续 normal-model representation 的可用输入 | 对照老师 PDF (3)、`main.tex` II-B 与 ledger；保留“不假设物理状态包含关系”限定 | 2026-07-30 | PASS |
| II-C. Problem Formulation | 三个编号问题在背景定义完成后集中提出，与三项贡献一一对应 | 引导句后每项都用问题名开头，并在同一项内给约束和目标，无模块堆砌 | reference construction 先解决污染，再由其 drift 导出 propagation，最后由残差质量导出 decision | 不新增公式符号；reference、normal propagation、residual-space decision 用义与后文一致 | 每项都能回指 III-B、III-C、III-D/IV 的显式结果 | 三个问题覆盖 reference、complete mapping propagation 和 conditional decision | normal data alone 是三项共同前提；未把 fault test 写入选择或 calibration | 对照 `main.tex` II-C、Introduction 三项贡献和 Conclusion；核对数量=3、顺序相同 | 2026-07-30 | PASS |
| III-A. Attention–Koopman–T–S Normal Model | 先给 causal encoder 与总体 mapping，再给 membership、local model、硬参数化和 free-rollout objective | 每段以“形成什么对象/为何需要”为主题，公式后紧接性质或作用说明，避免逐式陈述堆叠 | 严格过去信息先形成 $z^{enc}$，其后 scheduling 和 local interpolation 形成 transition，最后 free rollout 对齐部署 reference | 对照 ledger 搜索 $\varpi,\chi,\rho,\omega,\Pi,c,A,B,W,\mathcal T,\mathcal D$ 全部视觉变体；attention 权重未误写为 $\omega$ | 式 (4)–(12) 数量、编号、初值、归一化、逐元素约束和 $\ell_1$ loss 与老师稿一致 | encoder 输入、context、membership、local/interpolated transition、decoder 和 rollout update order 完整 | normal split 用 free rollout 训练；没有 future measurement teacher forcing、fault labels 或 test adaptation | 对照老师 PDF (4)–(12)、`main.tex` III-A、ledger；修复 $\varpi/\omega$、黑体/花体并将长式仅做等价拆行 | 2026-07-30 | PASS |
| III-B. Measurement-Decoupled Reference and Unified Residual | 位于 normal model 后，先定义统一 residual，再给 anchor 生命周期、命题和精确递推 | 段落按 unified object→short/long starts→anchor rule→innovation/context terms→Theorem 1 推进 | 训练/部署 residual 不一致的风险先导出 start-indexed object；measurement contamination 再导出冻结 reference 和延迟确认 | 核对 $e^z,e^y,e,z^0,s_k^0,\nu,\delta^\chi,\overline{\Gamma},\Phi$；状态残差不再误用输出符号 | 式 (13)–(20)、Proposition 1 条件、积分平均 Jacobian 与乘积次序逐项对齐老师稿 | 同一 rollout 对象覆盖 free training、short window、long reference，encoded/rollout context 接口明确 | 在线新输出只评价 residual，不更新 reference；alarm 后等待 $\ell_h$，未宣称不可检测类可被覆盖 | 对照老师 PDF (13)–(20)、Proposition 1、Theorem 1 和 Appendix A；修复 barred Jacobian 与 anchor 限定 | 2026-07-30 | PASS |
| III-C. Fuzzy-Structure-Dependent Normal Propagation | 紧接 Theorem 1，先分解 complete fuzzy Jacobian，再给 bound、forcing、corollary 和可靠 horizon | 先解释局部矩阵加权为何不足，随后定义梯度并给结构结果；经验 remainder 在结构项之后单独限定 | measurement decoupling 暴露 free-rollout drift，因此需要 complete mapping bound；有限数据不能给全局 forcing，故转入 held-out normal calibration | 核对 $\gamma^\rho,\overline{\gamma}^\rho,\Gamma_{z,k},\kappa_k,\bar\varepsilon_k,\beta,\ell_{\rm ref},\mathcal L_{\rm nr}$ | 式 (21)–(27)、Theorem 2 两项分解、induced norm、Assumption 1 与 Corollary 1 的条件范围一致；式 (23) 只做等价换行 | local dynamics、membership variation、scheduling sensitivity、forcing remainder 和 horizon 接口齐全 | bound 和 remainder 均为 normal-side；V2 provider/正式验证未实现，正文没有写成确定全局保证 | 对照老师 PDF (21)–(27)、Theorem 2、Assumption/Corollary、Appendix B 与 docs/06；修复 complete Jacobian 记号 | 2026-07-30 | PASS |
| III-D. Structured Basis-Perturbation Residual Learning | 在 normal propagation 后引入独立 diagnostic-side signal，先总体 group set 再拆三分支与 losses | 首段先说明仅减小 drift 不足，随后按 insertion site、rollout、response、objective 逐层细化 | normal-side objective 可能同时压低异常响应，因此在无 physical fault 样本时引入可重复 model-coordinate perturbations | $u/y/z$ 仅登记为 insertion-location groups；核对 $q_j^g,\alpha,\Delta e,\eta^{rsp},p,\mathcal L_{\rm sen},\mathcal L_{\rm iso}$ | 式 (28)–(33) 的三分支、堆叠 horizon、$\ell_1$ scaling、separation loss 和总 loss 与老师稿一致 | 每个分支的注入点、共享冻结模型、nominal subtraction 和 minibatch sampling 均说明 | 训练只用 synthetic model-coordinate perturbation 与 normal data；未声称它们对应 actuator/sensor/process faults | 对照老师 PDF (28)–(33)、`main.tex` III-D、ledger 与 docs/03 claim boundary；删除物理 fault class 暗示 | 2026-07-30 | PASS |
| IV-A. Input-Conditioned Residual Standardization | 作为独立决策任务入口，先给 condition descriptor，再给 mixture moments、score、calibration 和 threshold | 每段主题依次为条件依赖、局部统计、标准化、多尺度统计和 threshold 组成，定义前均说明原因 | raw residual 随工况与 reference age 变化，因此需条件尺度；传播项和 calibration remainder 再合成动态阈值 | 核对 $\upsilon,\zeta,\mu_i,\sigma_i,\widetilde e,\varsigma,\tau_{\rm cal},\tau_{\rm prop},\tau_{k,\ell}$ | 式 (34)–(39) 的关系顺序、逐元素平方、非重叠 block calibration 和最大分数定义一致 | fuzzy region、command、variation、exogenous input、reference age 与 local statistics 接口完整 | 只在 held-out normal blocks 校准并冻结；全文未声称五阶段 V2 protocol 已执行或 frozen fault test 已解封 | 对照老师 PDF (34)–(39)、`main.tex` IV-A、ledger 和 docs/03/06；修复式 (34) 顺序与式 (35) elementwise square | 2026-07-30 | PASS |
| IV-B. Sufficient Detection Condition | threshold 构造后给 residual-space sufficient condition，再解释其可控因素 | 动机句先指出 propagation 与 response learning 必须联合；命题、证明依据、amplitude implication 和范围限定顺序完整 | 较低 normal threshold 与较高 learned response 共同产生 detection margin，而非由任一模块单独保证 | 核对 $\widetilde e^0,\widetilde{\Delta e},\varepsilon_g,\eta_g$，未与 physical fault amplitude 混用 | 式 (40)–(41) 由 reverse triangle inequality 支撑，条件保持 strict inequality 和 residual-model scope | observed/learned response、mismatch bound、window threshold 的角色清楚 | 明确不是 minimum physical fault claim；没有故障数据参与 threshold 或 model selection 的陈述 | 对照老师 PDF Proposition 2 与 (40)–(41)、`main.tex` IV-B；保留 sufficient 和 residual-space 限定 | 2026-07-30 | PASS |
| IV-C. Unique-or-Set-Valued Isolation | detection 后再讨论 evidence strength，compatible set 先于 unique margin 与 Proposition 3 | 段落从距离定义到集合输出，再给 singleton 条件和失败时解释，避免强制分类 | 因 pattern evidence 可能不足，默认返回 compatible set；只有 separation 条件成立才收缩为 unique group | 核对 $\delta_g,\tau_{\rm iso},\mathbb G_k^{cmp},\delta_{\rm obs}$；`u/y/z` 仍是 model-coordinate labels | 式 (42) 与 Proposition 3 的 $2\delta_{\rm obs}$ 条件和 triangle-inequality 结论一致 | pattern library、distance、compatible set、ambiguity margin 和输出语义完整 | 不把集合输出解释为物理 fault identification；测试阶段不在线更新 pattern 或 threshold | 对照老师 PDF (42)、Proposition 3、`main.tex` IV-C 和 ledger；补强 set-valued honesty 限定 | 2026-07-30 | PASS |
| IV-D. Offline and Online Algorithms | 在所有定义和命题后汇总实现次序，Algorithm 1 先于 Algorithm 2，Conclusion 在其后 | 引导段分别说明 offline 与 online 目的；算法后段说明计算边界，浮动体最终按阅读顺序排版 | 先训练/校准并冻结对象，在线只执行 forward、standardization 与查表，由此阻断 test-time leakage | 算法中的 $\mathcal H,\mathcal T,\mathcal D,e,\zeta,\varsigma,\mathbb G$ 与正文和 ledger 一致 | 每一步引用 (8)、(10)、(12)、(16)、(23)、(27)、(28)、(33)–(39)、(42) 均存在且次序可执行 | offline inputs/outputs、冻结点、anchor management、alarm 后等待和 online non-update 完整 | normal training 与 nonoverlapping calibration blocks 分离；完整五阶段实验 protocol 属 V2 实现，正文没有伪装为已运行 final test | 对照老师 PDF Algorithms 1–2、`main.tex` IV-D、docs/03/06/07；固定 Algorithm 2 浮动位置并复核 7 页 PDF | 2026-07-30 | PASS |
| V. Conclusion | 位于方法和决策后，按 P1→P2→P3 回收主线，不新增 future-work 或实验结论 | 单段按 reference、propagation、structured response、decision、claim boundary 递进 | 三个问题的构造与条件结果回收到同一 normal-data-only 监测论点，末句限定物理解释 | 不引入新符号或新术语；`model-coordinate perturbations` 与 ledger 一致 | 没有超出 Theorems 1–2、Propositions 1–3 和 Corollary 1 的保证 | 汇总 reference、complete Jacobian、response 和 set-valued output | 明确不声称 unique physical fault identification，也不声称实验已完成 | 对照老师 PDF Conclusion、`main.tex` Conclusion、Abstract 和三项贡献；删除任何结果数值或投稿 readiness 暗示 | 2026-07-30 | PASS |
| Appendices A–B | 分别紧随 Conclusion 支撑 Theorems 1–2，不重复展开主文模块 | 每个 proof 先说明操作，再给关键 identity/derivative，最后回指对应 equation | Appendix A 从加减项导出 residual recursion；Appendix B 从 membership differentiation 导出 complete Jacobian 和 bound | 核对 $\overline{\Gamma}_{k|s},\Phi,\Gamma_{z,k},v_i,\omega_i,W_z$ 与主文一致 | 式 (43) 及两段 proof 的代数依赖、积分 mean-value identity、induced norm 与老师稿一致 | proof 使用主文已定义 mapping、state、context 和 membership，无新增接口 | 不涉及数据访问或实验主张 | 对照老师 PDF Appendices、`main.tex` Appendices、Theorems 1–2 和 ledger；修复 barred Jacobian 后重编译 | 2026-07-30 | PASS |
| References | 位于附录后；只列正文实际引用的老师稿 15 条来源 | Citation clusters 分别支撑 Koopman/T–S 背景、fault-model 先行路线和 finite-sample calibration 解释 | 引用只承担背景与范围证据，不被用来替代本文 derivation | citation key 与 `refs.bib` 匹配，无未定义 citation | 15 条引用不改变公式；编译后编号 [1]–[15] 连续 | 不适用模型接口审计；文献元数据与老师稿条目对照 | 不适用数据 split；未用引用暗示实验完成 | 对照老师 PDF References、`main.aux/main.bbl`、`refs.bib` 与最终 PDF；10 个老师稿缺失 BibTeX 条目已补齐，但 Zotero 本地/写接口不可达，入库状态未核验 | 2026-07-30 | PASS（正文一致性；Zotero 核验未闭环） |

## Gate record

| Gate | Last run | Evidence inspected | Conflict | Revision action | Affected sections | Result |
|---|---|---|---|---|---|---|
| Gate 1: section responsibilities | 2026-07-30 | 老师 PDF 目录、三个 problem items、三个 contributions、docs/03/06/07 | 无重复章节职责；缺少 experiment evidence | 保留老师稿理论结构并把实验缺失登记为 readiness limitation | 全文结构 | PASS |
| Gate 3: seven mandatory subsection audits | 2026-07-30 | 最终 `main.tex`、老师 PDF、`NOTATION_LEDGER.md`、docs/03/06/07、公式 (1)–(43)、命题/定理/算法/图 | 首轮发现 $\overline{\Gamma}_{k\mid s}$ 台账对象和 overline semantic family 不一致；其余小节审计无未决冲突 | 修正积分平均 Jacobian 的符号与装饰家族登记；按上表为 Abstract 至 References 逐项记录七类证据与修改 | 全部实质小节、附录和参考文献 | PASS |
| Gate 4: abstract–problem–contribution alignment | 2026-07-30 | 最终 Abstract、Introduction 三项 contributions、II-C 三项 problems、III/IV explicit outputs、Conclusion | 三项数量、顺序、技术范围和限定条件一致；Abstract 约 178 词 | 完成 content-level 对齐，固定 P1/C1、P2/C2、P3/C3 映射并核对 set-valued/physical-fault 限定 | Abstract、Introduction、II-C、III–IV、Conclusion | PASS |
| Gate 5: manuscript-wide audit | 2026-07-30 | 写作闭环审计 0/0、manuscript 审计 0/0、figure 审计 0 error/1 disposed warning、`latexmk`、最终 log、`pdfinfo`、`pdffonts`、7 页 150-dpi 渲染图 | Figure warning 仅指老师外部 PDF 无同名 editable source；既有 IEEEtran 工程不是 skill template 下载项目，故 `TEMPLATE_LOCK.json` 检查不适用；10 个老师稿缺失 BibTeX 条目已补齐，但 Zotero 本地/写接口不可达，入库状态未核验 | 加载 `amsthm`；仅对长公式等价拆行；固定 Algorithm 2 浮动顺序；确认无 error、undefined citation/reference、overfull box，字体全部嵌入且逐页无溢出/重叠；Zotero 核验作为独立环境阻塞保留 | 全文 source、Figure 1、Algorithms 1–2、compiled PDF、15 条 references | PASS（正文/LaTeX/PDF；Zotero 核验未闭环） |
