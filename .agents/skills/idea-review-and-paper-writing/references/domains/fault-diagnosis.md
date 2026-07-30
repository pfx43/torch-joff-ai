# Fault diagnosis

Read this document for fault detection, isolation, estimation, residual generation, dynamic thresholds, post-filtering, Koopman lifting, T–S fuzzy diagnosis, and related unknown-fault research. It contains both domain-wide constraints and route-specific options. Confirm the selected route in the active `MANUSCRIPT_CONTEXT.md`.

## Contents

- Unknown-fault information boundary
- Koopman lifting and output relations
- T–S premises, memberships, and attention
- Measurement-decoupled normal references
- Joint latent and output residuals
- Dynamic thresholds
- Post-filtering and robust fault gain
- Structured lifted disturbance bases
- Normal attenuation and fault preservation
- Fault channels and control-input definitions
- Sliding-mode route
- Multi-step training
- Offline and online implementation
- Experiments and project context

## Information boundary for unknown faults

In a normal-data-only or unknown-fault setting, the following are unknown unless an additional assumption explicitly supplies them:

- fault class;
- fault direction or physical distribution matrix;
- fault magnitude;
- occurrence time;
- temporal waveform;
- physical component location.

Online detection may use known control commands, real-time measurements, a model trained from permitted data, and offline-selected design parameters. It must not differentiate an unavailable fault variable, train against an unknown fault direction, maximize the gain of a direction that has not been assumed, or use fault labels contrary to the declared training setting.

During Stage 1, report the following as two explicit, mutually exclusive routes whenever both are considered:

- a direction-free route, whose guarantees are limited to faults distinguishable through the admitted measurements and residual operator;
- a structured route that introduces a fault basis, temporal basis, direction set, or low-dimensional fault family as an additional modeling assumption.

Do not import the structured assumption to make the first route feasible while continuing to describe the result as completely direction-free.

## Koopman lifting and output relations

In controlled Koopman preliminaries, show every control and measured exogenous argument consistently in the nonlinear transition, lifting, and operator action. Follow the infinite-dimensional definition with a genuinely distinct finite-dimensional controlled predictor and output or reconstruction equation; do not repeat the operator identity under another equation number or introduce the paper-specific history encoder in preliminaries.

A learned Koopman coordinate $\boldsymbol z$ is a latent lifted state and must not be identified with the physical state $\boldsymbol x$ without an explicit injectivity, reconstruction, or coordinate assumption. Do not assume that $\boldsymbol z$ contains the complete physical state.

If the physical measurement equation is

$$
\boldsymbol y_k=\boldsymbol C\boldsymbol x_k,
$$

do not reuse $\boldsymbol C\boldsymbol z_k$ as the latent output equation unless dimensions and the coordinate relation justify it. Use a learned output map, nonlinear decoder, or a separately defined latent output matrix.

When fault propagation is analyzed through a decoder Jacobian, state its evaluation point, first-order expansion, remainder, and local scope. Do not turn a decoder linearization into a global physical fault map.

## T–S premises, memberships, and attention

State membership nonnegativity and partition-of-unity conditions explicitly, and identify whether each membership is fixed, measured, data-driven, or learned.

T–S memberships must be calculated from premise variables or learned premise features, not from consequent matrices such as $\boldsymbol A_i$ or $\boldsymbol B_i$. Premise features may depend on permitted current or historical variables such as lifted states, inputs, measured scheduling variables, or causal histories.

Keep the meanings distinct:

- attention represents feature relevance, dependence, or weighting;
- fuzzy membership represents the degree to which a sample belongs to a rule or local regime.

Attention may construct or weight premise features, but it is not itself a membership function without an explicit normalized mapping and fuzzy interpretation.

For Gaussian, multivariate Gaussian, or attention-weighted Mahalanobis memberships, specify how centers, widths, temperatures, covariance or metric matrices, and stabilizers are obtained. If a matrix must be positive definite, give its parameterization or enforcement.

A lifted representation changes during encoder training. Centers, prototypes, memberships, and local consequents defined in that space may therefore require joint, alternating, or periodically updated training. Do not assume that clustering performed before representation learning remains valid without justification.

Changing independent scalar Gaussian memberships into a multivariate or attention-weighted form is not by itself a major contribution. The manuscript must identify the resulting modeling, theoretical, interpretability, or diagnostic capability.

## Measurement-decoupled normal reference route

Use the term *measurement-decoupled normal reference trajectory* or *measurement-decoupled normal reference branch* when the reference:

- starts from a delayed-confirmed normal state, safe anchor, or otherwise justified initial estimate;
- subsequently propagates without correction by new potentially faulty measurements;
- is designed to prevent current fault measurements from contaminating the normal reference.

Do not call this route a “protected reference” or “protected branch.”

For bilingual work, use “测量解耦正常参考轨迹” and “测量解耦正常参考分支” as the corresponding Chinese terms.

Distinguish the reference model from an unavailable counterfactual fault-free truth. A conditionally known scheduling sequence may be used only when its online availability is stated; a true fault-free scheduling trajectory is not automatically known.

Avoid using the same current measurement to update the normal reference, update attention or premise memberships, and then form the residual unless the resulting fault leakage is explicitly modeled. Analyze the first-order cancellation path through measurement-dependent weights when relevant, and explain how the decoupled structure removes or bounds it.

Measurements may remain in the data branch while the long-horizon normal reference is propagated from the anchor and admitted known inputs. State how the anchor is selected, how false confirmation is controlled, how model error accumulates, and how the reference is reset.

This is an optional research route, not a universal fault-diagnosis requirement. Record it as `confirmed` before using its terminology or guarantees.

## Joint latent and output residuals

A joint residual may combine a latent residual $\boldsymbol e_z$ and an output residual $\boldsymbol e_y$, but “more residual information” is not a proof of improved diagnosis.

Establish the benefit through an appropriate object, such as:

- an intersection of null spaces;
- a stacked observability or fault-transfer operator;
- enlargement of a detectable subspace;
- a rank or conditioning condition;
- a lower bound on minimum detectable gain.

For a scheduled, linear time-varying, or T–S interpolated error system, define the source and availability of every coefficient matrix and scheduling sequence. Separate propagated initial error, nominal modeling error, disturbances, and fault contributions. Show which nominal terms decay and which fault terms remain distinguishable.

Keep physical actuator and sensor fault channels separate until their one-step or multi-step lifted signatures are explicitly combined. State rank, conditioning, excitation, and reconstruction-error assumptions at the horizon used by the method.

## Dynamic thresholds

A theoretical dynamic threshold must be generated from fault-free residual or error dynamics. Propagate and separate, as applicable:

- initial-state or initial-estimation error;
- normal modeling error;
- Koopman one-step and multi-step prediction error;
- fuzzy-membership or scheduling variation;
- premise-variable error;
- output-decoding error;
- known-input-dependent uncertainty;
- disturbances and noise.

Empirical quantiles may calibrate a finite-sample noise term but must not replace the propagation bound. Keep nominal uncertainty bounds and fault-side detectable-gain conditions as different interfaces.

For a stacked or multi-step residual, propagate initial error, model uncertainty, known inputs, disturbances, and noise through the complete time window. Do not treat temporally coupled residual samples as independent blocks merely to simplify the threshold.

For a robust threshold, validate nominal coverage, false-alarm rate, operating-condition sensitivity, and membership or scheduling variation. If a probability level is claimed, identify the stochastic term and whether the guarantee is pointwise, window-wise, or sequential.

## Post-filtering and robust fault gain

Build post-filtering on established residual-generation, evaluation, and post-filter theory, including the Ding-style framework when it is the actual basis. Cite the verified source, state which assumptions transfer, and do not claim to reinvent the general idea.

Closed-form or LMI results derived for known LTI, LDTV, or LPV models do not automatically apply to matrices generated by deep historical encoders, Koopman lifting, predicted premises, or learned time-varying T–S interpolation. State the admissible model and scheduling trajectory set before transferring a result.

For an $H_-/H_\infty$ or minimum-fault-gain formulation, define separately:

- the admissible fault set;
- the nominal uncertainty and disturbance set;
- the membership or scheduling trajectory set;
- the norm and horizon;
- the claimed uniform lower or upper bound.

Match the post-filter output dimension to the design objective. A semidefinite relaxation without a rank or subspace constraint may return full-dimensional whitening rather than the intended low-dimensional diagnostic projection. If output dimension is fixed, address rank, subspace selection, or an equivalent low-dimensional parameterization.

Do not optimize a prescribed fault direction under a completely unknown-direction claim. A low-dimensional fault time basis, spatial basis, or structured family is an additional assumption and must be recorded as such.

## Structured disturbance bases in lifted space

One admissible route uses standard basis vectors or grouped standard bases in the lifted coordinates as candidate process-side disturbance channels. For example,

$$
\begin{bmatrix}
0 & 1 & 0 & \cdots & 0
\end{bmatrix}^{\mathsf T}
$$

may denote a candidate lifted-coordinate direction.

Such a basis represents a structured perturbation channel in the learned coordinates. It does not automatically identify a sensor, actuator, valve, physical component, or fault location. Any physical interpretation requires an additional mapping and evidence.

This route is incompatible with a claim of having no directional or structural prior at all. Mark the route as `confirmed` or retain it as an `alternative`; do not mix the two descriptions.

## Normal-term attenuation and fault preservation

A strong theoretical line for residual diagnosis may establish:

- decay of nominal and initialization contributions through products or powers of the normal error-transition matrices;
- a computable bound on remaining uncertainty;
- a uniform error-propagation bound along the admitted attention–T–S weight trajectories;
- an upper bound on residual displacement caused by scheduling or rule-weight errors;
- preservation or enhancement of admissible fault effects in the raw or post-filtered residual;
- a lower bound on minimum admissible fault gain;
- a joint-residual detectability improvement condition;
- a performance-loss bound caused by approximating a post-filter or hypernetwork gain;
- a detectability condition comparing a fault-side lower bound with the nominal threshold.

Each conclusion must identify the horizon, norm, scheduling assumptions, excitation assumptions, and computable quantities.

The target conclusions should not depend on an unknown true fault-free trajectory, an unavailable prediction-scheduling error, or online high-order differentiation. When a scheduling sequence is called known, state how it is obtained online. A network-generated sequence is a computable estimate, not the unknown fault-free scheduling sequence. Distinguish fixed estimated scheduling, conditionally known scheduling, a robust bound over unknown scheduling, and an empirical bound learned from finite data.

## Fault channels and control-input definitions

Distinguish the controller command $\boldsymbol u_k^c$ from the actuator-applied input $\boldsymbol u_k^a$. Select one as the model input, define its online availability, and write the actuator-fault channel that maps the command to the plant action. Do not use the two symbols interchangeably.

Treat sensor, actuator, and process faults as distinct physical channels. For every claimed type, show how it enters:

- the physical state equation;
- the measurement equation;
- the lifted Koopman or T–S dynamics;
- the latent and output residuals;
- the post-filter;
- the isolation score or structured fault basis.

If a structured lifted basis is used, do not claim that it identifies the physical fault type or component without an additional mapping.

## Sliding-mode route

When a discrete sliding-mode observer or residual route is selected, give both the disturbance-dominating lower gain bound and the upper bound that prevents repeated crossing or chattering of the stated boundary layer. Prove finite-step reachability and positive invariance under the admitted disturbance and sampling assumptions. Do not substitute an informal sign argument for these steps.

## Multi-step training

Use a multi-step Koopman prediction loss when it is needed to prevent a model from fitting only one-step error and to improve normal-trajectory propagation over the detection horizon. State the horizon, weighting, rollout convention, accumulated-error treatment, and training cost. Tie the loss to the deployed residual or detectability objective rather than adding an unrelated prediction task.

## Offline and online implementation

Place encoder and predictor training, T–S prototype or rule learning, local-matrix estimation, post-filter design, threshold-statistics estimation, structured fault-basis selection, and stability or robustness bound estimation offline.

Keep online processing close to:

- encoding current data or causal-history forward propagation;
- propagation or reset of a confirmed measurement-decoupled normal reference when that route is selected;
- one-step Koopman or T–S propagation;
- premise and membership calculation;
- local-model weighting;
- residual calculation;
- fixed or low-dimensional post-filtering;
- recursive dynamic-threshold update;
- fault decision and any admitted isolation score.

Avoid online reclustering, matrix inversion, SDP, SVD, full Hessians, second-order backpropagation, long-horizon parity-matrix redecomposition, dependence on a true fault-free trajectory, and long-window high-dimensional optimization. Prefer offline filter design, cached local models, short-window recursion, fixed bases, and low-rank structures.

Multi-step losses may improve training and long-horizon consistency. Online detection need not reevaluate the complete training loss, but the deployed recursion must match the model and bounds used in the proof.

## Experiments

Use only evidence permitted by the declared training regime. When normal-only training is claimed, keep fault labels out of training, hyperparameter tuning, threshold selection, and checkpoint selection.

In addition to general metrics, evaluate the claims actually made:

- false-alarm and missed-detection rates;
- detection delay across fault magnitudes;
- performance for the weakest admissible fault direction or minimum-gain case;
- the minimum singular value or an equivalent detectable-gain metric when the theory uses it;
- comparison of raw, joint, and post-filtered residuals when claimed;
- threshold coverage under normal operating changes;
- robustness to membership and premise errors;
- actuator and sensor faults separately and simultaneously when simultaneous diagnosis is claimed;
- ablations for attention, fuzzy memberships, reference decoupling, joint residuals, thresholds, and post-filters.

Do not report planned results as observed outcomes.

## Project-context requirements

Record the following in `MANUSCRIPT_CONTEXT.md`:

- whether training uses only normal data;
- exactly which fault properties are unknown;
- whether a structured basis or fault family is assumed;
- the physical-to-latent output relation;
- premise variables and how memberships are trained;
- whether measurement-decoupled reference propagation is selected;
- residual definitions and detectable-gain assumptions;
- threshold type and uncertainty decomposition;
- post-filter dimension and optimization route;
- offline and online computations;
- missing experimental evidence.
