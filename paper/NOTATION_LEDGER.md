# Notation Ledger

## Manuscript

- Working title: Measurement-Decoupled Attention–Koopman–T–S Fuzzy Monitoring via
  Structured Basis-Perturbation Residual Learning
- Main source:
  `teacher_latest_manuscript/Measurement_Decoupled_Attention_Koopman_TS_Fuzzy_Monitoring_Skill_Revised(1).pdf`
  (SHA-256:
  `e47deefbdee57d2f62ea5de2eb107f25bb7cac52d4ac8830c30f23c252a6e090`)
- Last synchronized: 2026-07-30（全文重写后终稿符号审计）

## Registry

Use one row per mathematical object. Register an object before it appears in
the manuscript.

| Symbol | Semantic family | Meaning | Naming basis / convention | Object type | Dimension | Typography | First definition | Scope |
|---|---|---|---|---|---|---|---|---|
| $k$ | 离散时间索引 | 当前离散采样时刻 | field standard: discrete-time index | index | scalar integer | italic lowercase | II-A before (1) | 全文 |
| $s$ | rollout 起点 | start-indexed rollout 的起始样本 | English initial: start | index | scalar integer | italic lowercase | III-A before (11) | III–IV |
| $i$ | 模糊规则索引 | T–S 规则或局部模型索引 | mathematical convention: local summation index | index | scalar integer | italic lowercase | III-A before (7) | III–IV |
| $j$ | 局部索引 | past token、局部求和或 basis 坐标的局部索引 | mathematical convention: local index | index | scalar integer | italic lowercase | III-A before (4) | III–IV，作用域由各公式限定 |
| $q$ | 局部选择/方向 | attention head 的局部索引 | mathematical convention: local selector | index | scalar integer | italic lowercase | III-A before (4) | III-A |
| $g$ | 模型坐标组 | `u/y/z` basis response group 索引 | English initial: group | index | scalar categorical | italic lowercase | III-D before (28) | III-D–IV |
| $m_u$ | 维度 | 控制输入维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | II-A before (1) | 全文 |
| $m_x$ | 维度 | 物理状态维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | II-A after (1) | 全文 |
| $m_y$ | 维度 | 测量输出维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | II-A after (1) | 全文 |
| $m_\xi$ | 维度 | 已测外生变量维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | II-A after (1) | 全文 |
| $m_z$ | 维度 | 学习提升坐标维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | II-B after (3) | 全文 |
| $m_a$ | 维度 | attention key/query 缩放维数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | III-A in (4) | III-A |
| $m_{\mathrm{hd}}$ | 维度 | multihead attention 的 head 数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with roman subscript | III-A before (4) | III-A |
| $m_r$ | 维度 | T–S 模糊规则数 | mathematical convention: dimension prefix | constant | positive integer | italic lowercase with subscript | III-A before (7) | III–IV |
| $\boldsymbol u_k$ | 输入/命令 | 监测器可获得的 controller command | field standard: control input | vector | $m_u$ | bold italic lowercase | II-A before (1) | 全文 |
| $\boldsymbol u_k^a$ | 输入/命令 | 实际施加到 plant 的 action | field standard: actuator-applied input | vector | $m_u$ | bold italic lowercase with superscript | II-A before (1) | II–IV |
| $\boldsymbol f_k^a$ | 故障 | actuator-side fault vector | field standard: actuator fault | vector | $m_u$ | bold italic lowercase with superscript | II-A in (1) | II |
| $\boldsymbol f_k^p$ | 故障 | process-side fault vector | field standard: process fault | vector | plant-defined | bold italic lowercase with superscript | II-A in (1) | II |
| $\boldsymbol f_k^s$ | 故障 | sensor-side fault vector | field standard: sensor fault | vector | $m_y$ | bold italic lowercase with superscript | II-A in (1) | II |
| $\boldsymbol x_k$ | 物理状态系统 | plant physical state | field standard: state vector | vector | $m_x$ | bold italic lowercase | II-A in (1) | II |
| $\boldsymbol y_k$ | 测量输出系统 | measured plant output | field standard: measured output | vector | $m_y$ | bold italic lowercase | II-A in (1) | 全文 |
| $\boldsymbol \xi_k$ | 外生条件 | measured exogenous variable | field standard: exogenous input | vector | $m_\xi$ | bold Greek lowercase | II-A in (1) | 全文 |
| $\boldsymbol \varepsilon_k^x$ | 误差/数值裕量 | physical-state disturbance or modeling error | field standard: additive error | vector | $m_x$ | bold Greek lowercase with superscript | II-A in (1) | II |
| $\boldsymbol \varepsilon_k^y$ | 误差/数值裕量 | measurement disturbance or noise | field standard: measurement noise | vector | $m_y$ | bold Greek lowercase with superscript | II-A in (1) | II |
| $\mathbb X$ | 物理状态系统 | admitted physical-state domain | mathematical convention: state space | set/space | subset of $\mathbb R^{m_x}$ | blackboard bold capital | II-A after (1) | II |
| $\mathbb Y$ | 测量输出系统 | admitted measured-output domain | mathematical convention: output space | set/space | subset of $\mathbb R^{m_y}$ | blackboard bold capital | II-A after (1) | II |
| $\mathbb Z$ | 提升坐标系统 | learned lifted-coordinate domain | mathematical convention: latent space | set/space | subset of $\mathbb R^{m_z}$ | blackboard bold capital | II-B after (3) | II–IV |
| $\mathcal X_0$ | 物理状态系统 | unknown healthy state-transition mapping | field standard: nonlinear state mapping | mapping | plant-defined | calligraphic capital | II-A in (1) | II |
| $\mathcal Y_0$ | 测量输出系统 | unknown healthy physical-output mapping | field standard: nonlinear output mapping | mapping | plant-defined | calligraphic capital | II-A in (1) | II–III |
| $\mathcal H_k^-$ | 历史/编码器 | strictly past causal history at sample $k$ | English initial: history | set/space | finite sequence | calligraphic capital with minus superscript | II-A in (2) | II–III |
| $\ell_h$ | 时间视野 | causal history length | English initial: length of history | constant | positive integer | italic script lowercase with subscript | II-A in (2) | II–IV |
| $\boldsymbol \psi$ | 提升坐标系统 | finite vector of Koopman observables | field standard: Koopman observable | vector | $m_z$ | bold Greek lowercase | II-B before (3) | II |
| $\boldsymbol h_j$ | 历史/编码器 | embedded token formed from one history element | English initial: history token | vector | embedding dimension | bold italic lowercase | III-A before (4) | III-A |
| $\varpi_{k j}^{(q)}$ | attention 权重 | strictly causal attention coefficient from token $j$ to sample $k$ in head $q$ | field standard: normalized attention weight | scalar | nonnegative | Greek lowercase | III-A in (4) | III-A |
| $\boldsymbol W_Q^{(q)}$ | 学习线性权重/映射 | query projection matrix of head $q$ | field standard: attention query matrix | matrix | compatible with $m_a$ | bold capital with roman subscript | III-A in (4) | III-A |
| $\boldsymbol W_K^{(q)}$ | 学习线性权重/映射 | key projection matrix of head $q$ | field standard: attention key matrix | matrix | compatible with $m_a$ | bold capital with roman subscript | III-A in (4) | III-A |
| $\boldsymbol W_V^{(q)}$ | 学习线性权重/映射 | value projection matrix of head $q$ | field standard: attention value matrix | matrix | compatible with token embedding | bold capital with roman subscript | III-A in (5) | III-A |
| $\boldsymbol W_O$ | 学习线性权重/映射 | output projection after head concatenation | field standard: attention output matrix | matrix | compatible with context dimension | bold capital with roman subscript | III-A in (5) | III-A |
| $\boldsymbol \chi_k$ | attention 上下文 | concatenated causal attention context | field standard: context vector | vector | context dimension | bold Greek lowercase | III-A in (5) | III |
| $\mathcal H_\Theta$ | 历史/编码器 | finite-history encoder including attention | English initial: history encoder | mapping | history to $\mathbb Z$ | calligraphic capital with parameter subscript | III-A in (5) | III |
| $\Theta$ | 模型参数 | complete learned parameter set | field standard: trainable parameters | constant | model-defined | Greek capital | III-A in (5) | III–IV |
| $\boldsymbol z_k^{\mathrm{enc}}$ | 提升坐标系统 | encoded predictive lifted coordinate | field standard: latent state | vector | $m_z$ | bold italic lowercase with roman superscript | II-B after (3) | III–IV |
| $\boldsymbol \rho_k$ | 模糊调度 | scheduling vector for rule memberships | field standard: premise/scheduling variable | vector | scheduling dimension | bold Greek lowercase | III-A in (6) | III–IV |
| $\boldsymbol W_z$ | 学习线性权重/映射 | linear map from lifted coordinate to scheduling vector | semantic mnemonic: lifted-state scheduling weight | matrix | compatible | bold capital with subscript | III-A in (6) | III |
| $\boldsymbol W_\chi$ | 学习线性权重/映射 | linear map from attention context to scheduling vector | semantic mnemonic: context scheduling weight | matrix | compatible | bold capital with Greek subscript | III-A in (6) | III |
| $\boldsymbol W_u$ | 学习线性权重/映射 | linear map from controller command to scheduling vector | semantic mnemonic: input scheduling weight | matrix | compatible | bold capital with subscript | III-A in (6) | III |
| $\boldsymbol W_\xi$ | 学习线性权重/映射 | linear map from exogenous variable to scheduling vector | semantic mnemonic: exogenous scheduling weight | matrix | compatible | bold capital with Greek subscript | III-A in (6) | III |
| $\varphi_i$ | 模糊调度 | negative metric energy for rule $i$ | mathematical convention: radial log-weight | scalar | scalar-valued function | Greek lowercase | III-A in (7) | III |
| $\omega_i$ | 模糊调度 | normalized firing strength of rule $i$ | field standard: T–S normalized membership | scalar | nonnegative and sums to one | Greek lowercase | III-A in (7) | III–IV |
| $\boldsymbol \Pi_i$ | 模糊度量 | positive diagonal scheduling-space metric of rule $i$ | field standard: metric matrix | matrix | scheduling by scheduling | bold Greek capital | III-A in (7) | III |
| $\boldsymbol c_i$ | 模糊度量 | center of rule $i$ in scheduling space | field standard: cluster/rule center | vector | scheduling dimension | bold italic lowercase | III-A in (7) | III |
| $\underline \pi$ | 模糊度量 | positive lower elementwise metric bound | mathematical convention: lower bound | constant | positive scalar | underlined Greek lowercase | III-A in (8) | III |
| $\overline \pi$ | 上界/加权派生量 | finite upper elementwise metric bound | mathematical convention: upper bound | constant | scalar greater than lower bound | overlined Greek lowercase | III-A in (8) | III |
| $\widetilde{\boldsymbol \pi}_i$ | tilde 变换量 | unconstrained trainable metric parameter | field standard: raw parameter before bounded transform | vector | scheduling dimension | bold Greek lowercase with tilde | III-A in (8) | III |
| $\widetilde{\boldsymbol c}_i$ | tilde 变换量 | unconstrained trainable rule-center parameter | field standard: raw parameter before bounded transform | vector | scheduling dimension | bold italic lowercase with tilde | III-A in (8) | III |
| $\boldsymbol v_i$ | 局部模型 | local lifted next-state mapping value | English initial: local value | vector | $m_z$ | bold italic lowercase | III-A in (9) | III |
| $\overline{\boldsymbol v}_k$ | 上界/加权派生量 | membership-weighted local model value | mathematical convention: weighted mean | vector | $m_z$ | bold italic lowercase with overline | III-C before (21) | III |
| $\boldsymbol A_i$ | 局部模型 | local lifted-state transition matrix | field standard: state matrix | matrix | $m_z$ by $m_z$ | bold capital | III-A in (9) | III |
| $\boldsymbol B_i$ | 局部模型 | local command-input matrix | field standard: input matrix | matrix | $m_z$ by $m_u$ | bold capital | III-A in (9) | III |
| $\mathcal T_\Theta$ | 正常模型映射 | interpolated attention–Koopman–T–S transition | English initial: transition | mapping | lifted/input/context to lifted state | calligraphic capital | III-A in (9) | III–IV |
| $\mathcal D_\Theta$ | 正常模型映射 | learned decoder from lifted state to measured output | English initial: decoder | mapping | lifted/input/exogenous to $\mathbb Y$ | calligraphic capital | III-A in (9) | III–IV |
| $\boldsymbol y^{\mathrm{dec}}$ | 测量输出系统 | decoded output of the learned model | English initial: decoded output | vector | $m_y$ | bold italic lowercase with roman superscript | III-A in (9) | III |
| $\kappa_A$ | 传播界 | hard infinity-norm budget for local state matrices | semantic mnemonic: bound for $A$ | constant | nonnegative scalar | Greek lowercase with subscript | III-A in (10) | III |
| $\kappa_B$ | 传播界 | hard infinity-norm budget for local input matrices | semantic mnemonic: bound for $B$ | constant | nonnegative scalar | Greek lowercase with subscript | III-A in (10) | III |
| $\kappa_W$ | 传播界 | hard infinity-norm budget for scheduling state map | semantic mnemonic: bound for $W$ | constant | nonnegative scalar | Greek lowercase with subscript | III-A in (10) | III |
| $\widetilde{\boldsymbol A}_i$ | tilde 变换量 | unconstrained raw parameter for $\boldsymbol A_i$ | field standard: raw parameter before hard transform | matrix | $m_z$ by $m_z$ | bold capital with tilde | III-A in (10) | III |
| $\widetilde{\boldsymbol B}_i$ | tilde 变换量 | unconstrained raw parameter for $\boldsymbol B_i$ | field standard: raw parameter before hard transform | matrix | $m_z$ by $m_u$ | bold capital with tilde | III-A in (10) | III |
| $\widetilde{\boldsymbol W}_z$ | tilde 变换量 | unconstrained raw parameter for $\boldsymbol W_z$ | field standard: raw parameter before hard transform | matrix | compatible | bold capital with tilde | III-A in (10) | III |
| $\ell_{\mathrm{tr}}$ | 时间视野 | free-rollout training horizon | English initial: training length | constant | positive integer | italic script lowercase with roman subscript | III-A before (11) | III |
| $\ell_{\mathrm{sh}}$ | 时间视野 | short online residual window length | English initial: short horizon length | constant | positive integer | italic script lowercase with roman subscript | III-B after (14) | III–IV |
| $\ell_b$ | 时间视野 | structured basis-response rollout horizon | English initial: basis horizon length | constant | positive integer | italic script lowercase with subscript | III-D before (30) | III–IV |
| $\ell_k^0$ | 时间视野 | age of the latest accepted normal reference | semantic mnemonic: reference age | scalar | nonnegative integer | italic script lowercase | III-B in (15) | III–IV |
| $\ell_{\mathrm{cf}}$ | 时间视野 | confirmation delay before committing an anchor | English initial: confirmation length | constant | positive integer | italic script lowercase with roman subscript | III-B before Proposition 1 | III–IV |
| $\ell_{\mathrm{det}}$ | 时间视野 | assumed maximum warning delay for detectable class | English initial: detection delay | constant | positive integer | italic script lowercase with roman subscript | Proposition 1 | III |
| $\ell_{\mathrm{ref}}$ | 时间视野 | reliable reference horizon under frozen tolerance | English initial: reference horizon | scalar | nonnegative integer | italic script lowercase with roman subscript | III-C in (26) | III–IV |
| $\boldsymbol z_{t\mid s}^{\mathrm{fr}}$ | 提升坐标系统 | free-rollout lifted state started at $s$ | semantic mnemonic: free rollout state | vector | $m_z$ | bold italic lowercase with roman superscript | III-A in (11) | III |
| $\boldsymbol \chi_{t\mid s}^{\mathrm{fr}}$ | attention 上下文 | predicted causal context used by free rollout | semantic mnemonic: free rollout context | vector | context dimension | bold Greek lowercase with roman superscript | III-A in (11) | III |
| $\boldsymbol y_{t\mid s}^{\mathrm{fr}}$ | 测量输出系统 | decoded free-rollout output | semantic mnemonic: free rollout output | vector | $m_y$ | bold italic lowercase with roman superscript | III-A in (11) | III |
| $\mathcal L_{\mathrm{pred}}$ | 损失函数 | normal free-rollout prediction loss | English initial: prediction loss | scalar | nonnegative | calligraphic capital | III-A in (12) | III |
| $\lambda_z$ | 损失权重 | lifted-state prediction loss weight | semantic mnemonic: weight for lifted state | constant | nonnegative scalar | Greek lowercase with subscript | III-A in (12) | III |
| $\lambda_y$ | 损失权重 | output prediction loss weight | semantic mnemonic: weight for output | constant | nonnegative scalar | Greek lowercase with subscript | III-A in (12) | III |
| $\boldsymbol e_{k\mid s}^{z}$ | 残差 | start-indexed lifted residual | field standard: residual/error in lifted space | vector | $m_z$ | bold italic lowercase with superscript | III-B in (13) | III–IV |
| $\boldsymbol e_k^y$ | 残差 | output-consistency residual | field standard: output residual | vector | $m_y$ | bold italic lowercase with superscript | III-B in (14) | III–IV |
| $s_k^0$ | rollout 起点 | latest delay-confirmed normal anchor index | semantic mnemonic: accepted start | index | scalar integer | italic lowercase | III-B before (15) | III–IV |
| $\boldsymbol z_k^0$ | 提升坐标系统 | long-horizon measurement-decoupled normal reference state | field standard: normal/reference state | vector | $m_z$ | bold italic lowercase with superscript | III-B in (15) | III–IV |
| $\boldsymbol e_k$ | 残差 | joint output/short-lifted/long-lifted residual | field standard: stacked residual | vector | $m_y+2m_z$ | bold italic lowercase | III-B in (16) | III–IV |
| $\boldsymbol \nu_k$ | 残差传播 | one-step encoded-state innovation or normal-model inconsistency | field standard: innovation | vector | $m_z$ | bold Greek lowercase | III-B in (17) | III |
| $\boldsymbol \delta_{k\mid s}^{\chi}$ | 差异/距离 | transition difference caused by finite attention context mismatch | field standard: difference term | vector | $m_z$ | bold Greek lowercase with superscript | III-B in (17) | III |
| $\overline{\boldsymbol \Gamma}_{k\mid s}$ | 上界/加权派生量 | integral mean Jacobian along the residual segment | mathematical convention: Jacobian/transition matrix | matrix | $m_z$ by $m_z$ | bold Greek capital with overline | Theorem 1 in (19) | III |
| $\boldsymbol \Gamma_{z k}$ | 模糊 Jacobian/梯度 | complete fuzzy lifted-state Jacobian | mathematical convention: Jacobian matrix | matrix | $m_z$ by $m_z$ | bold Greek capital | III-C in (22) | III |
| $\boldsymbol \Phi$ | 残差传播 | ordered product of start-indexed Jacobian matrices | mathematical convention: state-transition product | matrix | $m_z$ by $m_z$ | bold Greek capital | Theorem 1 after (20) | III |
| $\boldsymbol \gamma_{i k}^{\rho}$ | 模糊 Jacobian/梯度 | gradient of rule metric energy with respect to scheduling vector | mathematical convention: gradient | vector | scheduling dimension | bold Greek lowercase | III-C in (21) | III |
| $\overline{\boldsymbol \gamma}_k^\rho$ | 上界/加权派生量 | membership-weighted scheduling gradient | mathematical convention: weighted mean gradient | vector | scheduling dimension | bold Greek lowercase with overline | III-C in (21) | III |
| $\kappa_k$ | 传播界 | computable complete-Jacobian infinity-norm upper bound at sample $k$ | semantic mnemonic: propagation bound | scalar | nonnegative | Greek lowercase | Theorem 2 in (23) | III–IV |
| $\overline \varepsilon_k$ | 上界/加权派生量 | calibrated nonnegative bound on unresolved normal forcing | mathematical convention: error upper bound | constant | nonnegative scalar | overlined Greek lowercase | Assumption 1 in (24) | III–IV |
| $\beta_{k\mid s}$ | 传播界 | recursively propagated lifted-residual upper bound | mathematical convention: scalar bound sequence | scalar | nonnegative | Greek lowercase | Corollary 1 in (25) | III–IV |
| $\overline \beta$ | 上界/加权派生量 | allowed frozen reference-error tolerance | mathematical convention: upper tolerance | constant | nonnegative scalar | overlined Greek lowercase | Corollary 1 before (26) | III |
| $\mathcal L_{\mathrm{nr}}$ | 损失函数 | empirical normal-reference propagation regularizer | English initial: normal-reference loss | scalar | nonnegative | calligraphic capital | III-C in (27) | III |
| $\beta^{\mathrm{emp}}$ | 传播界 | empirical counterpart of propagated normal bound | English initial: empirical bound | scalar | nonnegative | Greek lowercase with roman superscript | III-C in (27) | III |
| $\mathbb G_b$ | 模型坐标组 | set containing the `u/y/z` perturbation groups | English initial: group set | set/space | three elements | blackboard bold capital | III-D in (28) | III–IV |
| $\boldsymbol q_j^u$ | 局部选择/方向 | standard basis vector inserted into the input branch | mathematical convention: basis direction | vector | $m_u$ | bold italic lowercase with superscript | III-D after (28) | III-D |
| $\boldsymbol q_j^y$ | 局部选择/方向 | standard basis vector inserted into the measurement branch | mathematical convention: basis direction | vector | $m_y$ | bold italic lowercase with superscript | III-D after (28) | III-D |
| $\boldsymbol q_j^z$ | 局部选择/方向 | standard basis vector inserted into the lifted branch | mathematical convention: basis direction | vector | $m_z$ | bold italic lowercase with superscript | III-D in (29) | III-D |
| $\alpha$ | 模型坐标组 | signed amplitude of a structured basis perturbation | field standard: perturbation amplitude | scalar | scalar | Greek lowercase | III-D after (28) | III-D–IV |
| $\Delta\boldsymbol e$ | 模型坐标响应 | stacked difference between perturbed and nominal residual rollouts | mathematical convention: finite difference | vector | stacked residual dimension | bold italic lowercase with delta prefix | III-D in (30) | III-D–IV |
| $\eta^{\mathrm{rsp}}$ | 模型坐标响应 | perturbation-amplitude-scaled L1 residual sensitivity | English initial: response sensitivity | scalar | nonnegative | Greek lowercase with roman superscript | III-D in (31) | III-D–IV |
| $\boldsymbol p$ | 模型坐标响应 | unit-L1 response pattern | field standard: normalized pattern vector | vector | stacked residual dimension | bold italic lowercase | III-D in (31) | III-D–IV |
| $\varepsilon_0$ | 误差/数值裕量 | positive numerical regularizer in response scaling | mathematical convention: numerical stabilizer | constant | positive scalar | Greek lowercase with subscript | III-D after (31) | III-D |
| $\overline \eta_g$ | 上界/加权派生量 | target or lower reference sensitivity for group $g$ | mathematical convention: group sensitivity reference | constant | nonnegative scalar | overlined Greek lowercase | III-D in (32) | III-D–IV |
| $\eta_{\mathrm{iso}}$ | 模型坐标响应 | intergroup pattern-separation margin | English initial: isolation margin | constant | nonnegative scalar | Greek lowercase with roman subscript | III-D in (32) | III-D–IV |
| $\mathcal L_{\mathrm{sen}}$ | 损失函数 | structured response sensitivity loss | English initial: sensitivity loss | scalar | nonnegative | calligraphic capital | III-D in (32) | III |
| $\mathcal L_{\mathrm{iso}}$ | 损失函数 | intergroup response separation loss | English initial: isolation loss | scalar | nonnegative | calligraphic capital | III-D in (32) | III |
| $\mathcal L_{\mathrm{bal}}$ | 损失函数 | minimum rule-occupancy balance penalty | English initial: balance loss | scalar | nonnegative | calligraphic capital | III-D after (33) | III |
| $\mathcal L$ | 损失函数 | complete training objective | field standard: total loss | scalar | nonnegative | calligraphic capital | III-D in (33) | III |
| $\lambda_{\mathrm{nr}}$ | 损失权重 | weight of normal-reference regularizer | semantic mnemonic: normal-reference weight | constant | nonnegative scalar | Greek lowercase with roman subscript | III-D in (33) | III |
| $\lambda_{\mathrm{sen}}$ | 损失权重 | weight of sensitivity loss | semantic mnemonic: sensitivity weight | constant | nonnegative scalar | Greek lowercase with roman subscript | III-D in (33) | III |
| $\lambda_{\mathrm{iso}}$ | 损失权重 | weight of separation loss | semantic mnemonic: isolation weight | constant | nonnegative scalar | Greek lowercase with roman subscript | III-D in (33) | III |
| $\lambda_{\mathrm{bal}}$ | 损失权重 | weight of occupancy penalty | semantic mnemonic: balance weight | constant | nonnegative scalar | Greek lowercase with roman subscript | III-D in (33) | III |
| $\boldsymbol \upsilon_k$ | 条件标准化 | command variation $\boldsymbol u_k-\boldsymbol u_{k-1}$ | field standard: input increment | vector | $m_u$ | bold Greek lowercase | IV-A in (34) | IV |
| $\boldsymbol \zeta_k$ | 条件标准化 | condition descriptor containing fuzzy region, command, command variation, exogenous variable, and reference age | field standard: conditioning descriptor | vector | descriptor dimension | bold Greek lowercase | IV-A in (34) | IV |
| $\boldsymbol \omega_k^0$ | 模糊调度 | firing-strength vector evaluated along the normal reference | field standard: T–S membership vector | vector | $m_r$ | bold Greek lowercase with superscript | IV-A after (34) | IV |
| $\boldsymbol \mu_i$ | 条件标准化 | rule-local conditional normal residual mean | field standard: mean vector | vector | joint residual dimension | bold Greek lowercase | IV-A before (35) | IV |
| $\boldsymbol \mu_k$ | 条件标准化 | mixture conditional residual mean | field standard: mean vector | vector | joint residual dimension | bold Greek lowercase | IV-A in (35) | IV |
| $\boldsymbol \sigma_i$ | 条件标准化 | positive rule-local elementwise residual scale | field standard: scale vector | vector | joint residual dimension | bold Greek lowercase | IV-A before (35) | IV |
| $\boldsymbol \sigma_k$ | 条件标准化 | mixture elementwise residual scale | field standard: scale vector | vector | joint residual dimension | bold Greek lowercase | IV-A in (35) | IV |
| $\varepsilon_\sigma$ | 误差/数值裕量 | positive scale floor in conditional variance | mathematical convention: numerical stabilizer | constant | positive scalar | Greek lowercase with subscript | IV-A in (35) | IV |
| $\widetilde{\boldsymbol e}_k$ | tilde 变换量 | elementwise standardized joint residual | field standard: standardized residual | vector | joint residual dimension | bold italic lowercase with tilde | IV-A in (36) | IV |
| $\ell$ | 时间视野 | one member of the predeclared multiscale window family | English initial: window length | constant | positive integer | italic script lowercase | IV-A before (37) | IV |
| $\varsigma_{k\ell}$ | 多尺度检测 | L1 residual score for window length $\ell$ ending at $k$ | field standard: scalar monitoring statistic | scalar | nonnegative | Greek lowercase | IV-A in (37) | IV |
| $\tau_{\mathrm{cal}}$ | 多尺度检测 | held-out normal calibration quantile for a declared condition/window cell | English initial: calibrated threshold | scalar | nonnegative | Greek lowercase with roman subscript | IV-A before (38) | IV |
| $\tau_{\mathrm{prop}}$ | 多尺度检测 | standardized propagation contribution to the online threshold | English initial: propagation threshold component | scalar | nonnegative | Greek lowercase with roman subscript | IV-A before (38) | IV |
| $\tau_{k\ell}$ | 多尺度检测 | input/region/age-conditioned online threshold | field standard: detection threshold | scalar | nonnegative | Greek lowercase | IV-A in (38) | IV |
| $\varsigma_k^{\max}$ | 多尺度检测 | maximum score over the predeclared finite window family | mathematical convention: maximum statistic | scalar | nonnegative | Greek lowercase with roman superscript | IV-A in (39) | IV |
| $\widetilde{\boldsymbol e}^{0}$ | tilde 变换量 | normal standardized residual block in Proposition 2 | field standard: normal residual block | vector | windowed residual dimension | bold italic lowercase with tilde | Proposition 2 | IV |
| $\varepsilon_g$ | 误差/数值裕量 | mismatch bound between actual and learned group response | mathematical convention: response error bound | constant | nonnegative scalar | Greek lowercase with subscript | Proposition 2 | IV |
| $\eta_g$ | 模型坐标响应 | learned lower response sensitivity used in the amplitude condition | English initial: group sensitivity | constant | nonnegative scalar | Greek lowercase with subscript | IV-B after (40) | IV |
| $\delta_g$ | 差异/距离 | smallest L1 distance from observed response to stored patterns of group $g$ | field standard: distance | scalar | nonnegative | Greek lowercase with subscript | IV-C before (42) | IV |
| $\tau_{\mathrm{iso}}$ | 多尺度检测 | calibrated compatibility tolerance for pattern matching | English initial: isolation tolerance | constant | nonnegative scalar | Greek lowercase with roman subscript | IV-C in (42) | IV |
| $\mathbb G_k^{\mathrm{cmp}}$ | 模型坐标组 | compatible set of model-coordinate groups at sample $k$ | English initial: compatible group set | set/space | subset of $\mathbb G_b$ | blackboard bold capital with roman superscript | IV-C in (42) | IV |
| $\delta_{\mathrm{obs}}$ | 差异/距离 | admitted distance from an observed response to its generating stored pattern | English initial: observation mismatch | constant | nonnegative scalar | Greek lowercase with roman subscript | Proposition 3 | IV |

Recommended object-type values are `scalar`, `vector`, `matrix`, `mapping`,
`operator`, `set/space`, `index`, and `constant`. The typography entry records
the applied convention; the authoritative convention is in
`references/living-user-rules.md`.

Use one of these naming-basis forms: `field standard: <object or source>`,
`mathematical convention: <object>`, `English initial: <word>`,
`semantic mnemonic: <concept>`, or `project-specific: <why no stronger convention is
usable>`. Prefer the first applicable category in that order. A meaningful
initial is not acceptable when it conflicts with an established convention or
an existing symbol family. In `First definition`, write either
`Introduction Notation` or the exact section, equation, algorithm, table, or figure location
where the object is defined before use.

## Reserved and rejected families

Record manuscript-specific reservations and rejected alternatives. Do not use
this table to override cross-manuscript rules.

| Base family | Reserved meaning | Rejected competing meaning | Resolution |
|---|---|---|---|
| $f/F$ | physical fault quantities | state-transition mapping | Healthy transition uses $\mathcal X_0$; learned transition uses $\mathcal T_\Theta$ |
| $n/N$ | normal or nominal status if needed | dimension prefix | All dimensions use $m$ |
| $H$ | history and neural encoder family | Hilbert space or physical output matrix | History uses $\mathcal H$; no $H$-Hilbert-space object is introduced |
| $u$ | control-command and applied-input family | unrelated uncertainty variable | Uncertainty and exogenous variables use $\xi$ or $\varepsilon$ |
| $z$ | learned lifted-coordinate family | physical state | Physical state remains $\boldsymbol x$ |
| $W$ | learned weight and linear-map matrices | disturbance/noise | Noise and remainder use $\varepsilon$ |
| $G$ | model-coordinate group set | generator network | No generator network is present |
| $\omega$ | T–S normalized firing strengths | attention coefficient | Attention coefficients use $\varpi$ |

## Conflict log

| Date | Collision or mismatch | Affected locations | Resolution | Status |
|---|---|---|---|---|
| 2026-07-30 | 老师稿的 $\varepsilon$ family 同时承载 physical error、normal forcing、numerical floor 和 response mismatch bound | (1)、(24)、(31)、(35)、Proposition 2 | 因用户要求公式完全不变，本轮不重命名；统一登记为“误差/数值裕量”语义家族，并要求 prose 在每次首次出现时明确对象 | ACCEPTED SOURCE CONSTRAINT |
| 2026-07-30 | $j$ 在 attention token、rule summation 和 basis coordinate 中局部复用 | (4)、(7)、(21)、(28)–(32) | 保留老师公式；台账把 $j$ 限定为公式局部索引，并要求每个作用域显式说明 | ACCEPTED SOURCE CONSTRAINT |
| 2026-07-30 | 自动审计会把 `\overline` 和 `\widetilde` 装饰命令本身当作 base family；老师稿又分别用这些装饰表示 weighted/bounded、raw/standardized 对象 | (8)、(10)、(21)、(24)–(26)、(32)、(36)、Proposition 2 | 不改公式；台账用“上界/加权派生量”和“tilde 变换量”登记装饰家族，Meaning 列保留每个对象的精确含义 | ACCEPTED SOURCE CONSTRAINT |
| 2026-07-30 | $\boldsymbol W$ family 同时用于 attention projection 和 scheduling maps | (4)–(6) | 二者均属于 learned linear weights/maps，统一登记为“学习线性权重/映射”，并由下标区分接口 | PASS |
| 2026-07-30 | 当前 source 没有 Introduction-end Notation paragraph | Introduction 与首次公式 | 为避免改变老师稿结构和增加未授权内容，采用“精确首次定义”路线；每个对象在首次公式前后解释 | PASS |
| 2026-07-30 | `u/y/z` labels 容易被误写成 actuator/sensor/process fault classes | III-D、IV-C、Conclusion | 统一登记为 model-coordinate perturbation groups；没有额外映射时禁止物理 singleton 表述 | PASS |
