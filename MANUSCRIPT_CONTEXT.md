# MANUSCRIPT_CONTEXT

英文论文（IEEE TFS 目标）的项目级写作上下文。`ieee-english-paper-polish` skill 的每次稿件任务开始前必须整读本文件；只有项目级决策变化时才更新对应条目。创建：2026-07-26（P0）；2026-07-26 P1–P4 版本曾完成本地编译（IEEEtran 双栏 11 页）。2026-07-27 八轮理论深化已大幅改写 `paper/main.tex`，旧 PDF 不再代表当前源文件；当前 MiKTeX 因 fresh-installation setup 未完成而阻断重新编译和 PDF 目检。实验节仍为协议 + 数值占位（见第 3 节纪律）。

## 1. 论文核心思想

闭环非线性系统、仅正常数据条件下的故障检测与结构化隔离。核心构造：把“读取实际系统的数据支路”与“生成正常参照的受保护支路”分离；受保护支路只在记录命令固定时阻断 post-anchor 测量 token 的直接回注，不声称阻断 $y\to u\to\hat z$ 的闭环因果路径。在该支路的堆叠多步残差上建立**有限视野签名/扰动 LTV 分解**（一个对象、三组算子），据此给出三大核心模块：

1. 堆叠残差后滤波器：白化坐标中的锚点谱商 + 被保留锚点半径定价 + 未零化 guard + matched/omnibus 支路库；奇异值阈值 $\tau$ 对应 Gram 投影阈值 $\tau^2$，全部统计量、阈值和签名共享同一部署算子 $L_b$；

2. 输入与运行状态调度的动态阈值发生器：路径条件联合集合传播 + 锚点/确定性/统计三层分账 + 正阈值 floor + 有限 episode maximum conformal 校准；anchor gate 和状态转移在最终校准前冻结，检测与 full-normal 归因使用互不重用的校准 episode；

3. 三类故障结构化隔离：在完整有限监视器前态与 raw window 上建立 actuator/sensor/process/mixed/Normal 联合 explanation graph；mask 是反事实重算而非新物理观测；候选集保留所有可行解释，只有非正常 singleton 才给标签，真值与覆盖事件均提升为支持轨迹。

详细技术路线与诊断依据归档在 `docs/旧文档/论文核心重构与写作计划.md`；更早的旧方案归档在 `docs/旧文档/论文方法完整设计.tex`（其第 10、14 节由新方案取代，其余大部分保留为素材：状态机、精确传播定理、完整模糊 Jacobian、有限时间界、掩码机制、命题 1/4 等）。

## 2. 建模与数据边界

- 对象：离散时间闭环非线性系统；执行器故障加性进输入、传感器故障加性进线性输出方程 $y_k=Cx_k$（主版本）、过程故障加性进状态方程。$u_k$ 为控制器记录指令（可信）；$u^{\mathrm{act}}$、$x$、故障量不可测；外生工况 $\xi_k$ 假定不受故障影响。
- **硬边界（不可违反）**：训练、结构/超参选择、阈值校准仅用正常数据；故障数据只进冻结后的最终测试。当前理论要求五段正常数据：训练 / 估计 / 检测校准 / 归因校准 / 冻结测试，块间留隔离带；检测与归因校准 episode 不得重用。
- 实验对象：闭环 CSTR（preset `cstr_closed_loop_fd`，u=(Ci,Ti,Tci)，y=(C,T,Tc,Qc)，fault01–08、onset=200，故障 1–2 过程 / 3–5 执行器 / 6–8 传感器）。已知前置修补：逐行 onset 标签、u/y schema、许可 to_verify。
- 老师稿两处 critical 修正已采纳（待导师最终确认）：推论 1 改校准可观测超额得分；隔离配对改堆叠受保护多步残差 ↔ $\Phi$ 加权响应矩阵。

## 3. 语言、体例与文件

- 英文、`IEEEtran` 期刊双栏、IEEE TFS 风格；定理环境按 skill 分类学（目标：2 个中心定理，其余 Lemma/Proposition/Corollary/Remark）。
- 文件：`paper/main.tex` + `paper/refs.bib`（P3 创建）；编译 `latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error`；交付前 PDF 逐页目检。
- 参考文献只收录逐条核实的真实条目；实验数值一律占位符，不发明结果。

## 4. 章节大纲（P0 版）

I Introduction（无编号公式；4 条贡献）· II Preliminaries and Problem Formulation（受控 Koopman 算子 + 独立有限维受控预测器、T–S 隶属度条件、三类故障；末尾 Problem Formulation 小节 2 个中心目标：有限 episode 联合误报控制的检测、带认证边界和集合值拒识的结构化隔离）· III Proposed Method（A 受保护残差生成；B 签名/扰动分解与后滤波器；C 动态阈值发生器；D 结构化隔离；E 学习与损失；F 端到端算法与复杂度）· IV Experiments（协议 + 占位）· V Conclusion · Appendix（长证明）。

## 5. 符号账本（种子，P3 定稿）

已决定的族约定（继承 living-user-rules）：

- $k$ 采样时刻，$s$/$t_0$ 锚点，$u$ 输入，$m_\bullet$ 维度前缀，$f$/$\bm F$ 族一律故障语义（含三类签名算子 $\bm F_{\mathrm a},\bm F_{\mathrm s},\bm F_{\mathrm p}$），$\varepsilon$ 族噪声/扰动，$W/\omega$ 族白化与隶属权重，粗体向量/矩阵、花体映射、黑板体仅数域/空间（$\mathbb H^\infty$ 记 Hilbert 空间）。
- 禁用：`\operatorname{col}`（写显式括号堆叠）、$y^m$ 上标、黑板体数据变量、`H` 用于物理输出映射（保留给神经/历史侧）。
- 待定与冲突监视：正常侧扰动算子族临时记 $\bm G_\bullet$（与度量参数化 $G_i$、得分分散度 $\Sigma_\varsigma$ 查冲突后定稿）；$T$（统计量）与 $\mathcal T$（可能的 Toeplitz 记号）分离；Koopman 算子与 Attention 键符号保持分立。
- 中文设计文档的符号表（`docs/旧文档/论文方法完整设计.tex` 第 3 节）是维度与对象清单的历史底本，英文稿逐符号换轨时在此登记。

## 6. 已确认决策

| 日期 | 决策 |
|---|---|
| 2026-07-26 | 老师反馈后重构三核心模块（后滤波器/动态阈值/结构化隔离），采用"一个堆叠残差对象、三组算子"统一框架 |
| 2026-07-26 | 最终交付改为英文 IEEE TFS 论文原文（本 skill 体例）；中文设计文档降为决策记录与素材库 |
| 2026-07-26 | 实验节写完整协议 + 数值占位，不发明结果 |
| 2026-07-27 | 八轮理论对抗审核后仍有 R97--R99，按规格暂停并标记未理论饱和；所有强结论降级到完整观测、有限 episode、轨迹级覆盖事件 |

## 6.2 八轮理论深化后的当前决策（2026-07-27）

| 决策 | 约束 |
|---|---|
| 后滤波在 white space 选择锚点谱子空间，$L_0=Q_w^\mathsf TW$；partial quotient 的锚点残留逐支路定价，guard 保留 | 不得把可逆白化写成信息增益；$\tau$ 奇异值切分必须在 Gram 上用 $\tau^2$ |
| 动态阈值使用 certified operator enclosure、联合 support-function 外包和 $\epsilon_\Gamma>0$ | 普通 power iteration 不能作安全上界；零阈值不能进入普通 gauge |
| FAR 只对预指定有限 episode 的 detection maximum 成立 | 分位使用有限样本秩；分辨率不足时 $q=+\infty$；无限首报警无保证 |
| anchor gate、重锚、mode、hysteresis 与 score state path 在估计块冻结 | 最终 $q_\alpha/q_\beta$ 不得反馈生成自身校准分数的状态轨迹 |
| Normal 归因使用 full-normal oracle：$\mathbb A_0^{det}\cup\overline{\mathbb E}_0^{full}(q_\beta)$ | 先用 detection-calibration 集冻结 $q_\alpha$，再用独立 attribution-calibration 集校准 $q_\beta$ |
| explanation graph 的充分观测包含 $(\mathsf m_{k^-},D_k)$、$L/T/\Gamma$、算子 enclosure 与全部 mask 输出 | raw-identical/full-state-identical 解释必须得到相同算法输出；潜在健康 regime 不能冒充观测量 |
| fault-vs-fault 的对称半径特例是 $\operatorname{dist}>r_c+r_d$，Normal 直接做 full-set 分离 | 因子 2 只属于两个完整单位对称 fault tube |
| H1 风险预算赋给支持轨迹 $\bm c^\star$ 上的 episode-level signature/nuisance 同时事件 | 逐窗口覆盖、$\alpha$、$\beta$ 均不是标签 PPV/FDR/后验置信度 |

完整裁定见 `.scratch/理论深化饱和审计/最终裁定矩阵.md`；第一至第八轮账本保留每一项反例与降级来源。

## 6.1 对抗性审查后的定稿设计决策（2026-07-26，已写入 paper/main.tex）

| 决策 | 来源 |
|---|---|
| 支路库加**未零化守卫支路**（原始堆叠范数 + 保守半径阈值）——覆盖零化器精确吞掉的"锚点后首步"故障方向（$\mathrm{Range}(\bm F_{\mathrm a}^{(1)})\subseteq\mathrm{Range}(\bm G_0)$ 结构定理） | 反驳 A3 |
| 匹配支路一律用**低维持续故障参数化**（$\bm 1_N\otimes I$，dim $m_u$/通道级/$m_z$），全类子空间增益不随 $N$ 增长 | 反驳 A4 + B5 |
| 正常数据**四分块**：训练 / 估计（包络、$\Sigma_e$、支路尺度）/ 校准（仅 conformal 分位）/ 冻结测试；重锚策略在校准段重放且与告警无关 | 反驳 B1 |
| 保证归属定稿：FAR ← 块可交换性（定理 1，包络错不毁）；可检测性 ← 故障时输入处包络有效性（附录假设 A5′） | 反驳 B1/B2 |
| $\sigma_{\min}$ 限持续故障子空间；TH-2 比较降级为 Remark（诚实双侧） | 反驳 B4/B5 |
| 上下文抵消引理显式化（$\nu+\delta^\chi$ 的 $\chi^E$ 依赖精确抵消）；$\delta^\chi$ 单独不作隔离特征 | 反驳 C1 |
| 过程类弃用 $\mathcal A\cap\mathcal S_{\mathrm a}^\perp$（结构性全空间），改**持续过程子空间** + 结构外能量检验（uncertified Process-side）；定理 2 只对真子空间类主张 | 反驳 C4 |
| 掩码命题带完整实现条件（全流水线重算）；豁免检验**单侧**（回落支持 Sensor-j，不回落不排除）；Mixed 经**掩码消去法**判定且先于裕度拒识 | 反驳 C2/C6 |
| 分解携带显式有效域（度量锐度 × 包络小），在线隶属度分散门控，破门降级守卫支路 | 反驳 A1 |
| 检测统计量每步无条件运行并无条件校准（避免告警门控后选择偏差）；隔离签名 JVP 仅告警后装配 | 反驳 A6 |

## 7. 未决问题

- 理论饱和未达成：第七、八轮仍有实质新增，八轮上限后按规格 Pause；若继续审核，需用户另行授权新一轮规格。
- certified segment-integral enclosure、full-normal outer oracle、monotone refinement cache 与五段数据切分尚未实现。
- $\mathcal E_{\mathrm{sig},\bm c^\star}^{\mathrm{ep}}/\mathcal E_{\mathrm{nuis},\bm c^\star}^{\mathrm{ep}}$ 的物理概率预算尚无外部证据。
- 导师确认：设计文档第 24 节 10 问 + 计划第 12 节新增 3 问 + 本稿全文。
- 实验数值：待方法实现（设计文档 §21 落点）与冻结评估后回填 Tables I–II；作者信息与投稿格式化（现为 Anonymous 占位）。
- CSTR 数据许可 to_verify。
- 反驳报告中记录的实现风险（锐规则区求积节点数、$\Sigma_e$ 样本量、慢采样假设）在实现阶段逐项落实。
