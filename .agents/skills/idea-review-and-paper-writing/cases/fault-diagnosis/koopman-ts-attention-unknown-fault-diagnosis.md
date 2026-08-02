# Case: Koopman–T–S–attention diagnosis of unknown faults

## Case identity

- Case ID: `fault-diagnosis/koopman-ts-attention-unknown-fault-diagnosis`
- Task group: fault diagnosis
- Working title: to be confirmed
- Target journal: IEEE Transactions on Fuzzy Systems
- Manuscript language: Chinese and/or English, depending on the active task
- Project path or link: unresolved
- Active revision-folder `MANUSCRIPT_CONTEXT.md`: unresolved
- Related domain guidance: `references/domains/fault-diagnosis.md`
- Last explicit decision date: 2026-07-29

## Contents

- Positioning and central problem
- Information and data boundaries
- Decision history and selected route
- Incompatible alternatives
- Terminology, formal claims, and implementation
- Experiments and unresolved questions
- Reusable insights

## Positioning and central problem

This case explores fault detection using a normal-data-trained Koopman–T–S model with attention-assisted premise representation. The central tension is that a learned nonlinear normal model must remain diagnostically sensitive to unknown faults without using unavailable fault labels, directions, or counterfactual fault-free trajectories.

The exact paper route remains partly exploratory. Potential contribution themes include:

- a causal or measurement-decoupled normal reference mechanism that limits fault leakage;
- a joint latent/output residual with a provable detectability advantage;
- a dynamic threshold and optional post-filter whose uncertainty and fault interfaces are separated.

The presence of Koopman lifting, attention, or a T–S module is not by itself a contribution.

The complete explored framework also distinguishes sensor, actuator, and process faults; normal-only training; unknown fault classes; multi-step normal prediction; scheduling uncertainty; dynamic thresholds; and structured isolation. These elements are not automatically separate contributions and must be selected only when the final route proves and validates them.

## Information and data boundaries

- Training data are primarily normal-operation data.
- Online information may include known control commands, current measurements, and parameters learned offline.
- Fault class, physical direction, magnitude, onset time, temporal form, and physical action matrix are unknown unless an additional route explicitly assumes structure.
- True hidden states and counterfactual fault-free trajectories are unavailable online.
- Fault labels must not be used for model selection or threshold tuning under the normal-only route.

## Decision history

| Topic | Status | Current decision | Reason or evidence | Date or discussion source |
|---|---|---|---|---|
| Normal-data-only training | confirmed | Use normal data as the principal training source | Matches the intended unknown-fault setting | Prior exploratory discussions |
| Koopman latent state equals physical state | rejected | Treat the lifted state as a learned latent coordinate | Equality is not generally justified | Prior exploratory discussions |
| Attention equals fuzzy membership | rejected | Keep attention relevance and membership semantics distinct | They represent different mathematical objects | Prior exploratory discussions |
| Measurement-decoupled normal reference | exploratory | Propagate a normal reference from a delayed-confirmed safe anchor without current-measurement correction | May reduce fault leakage and residual cancellation | Prior exploratory discussions |
| Joint latent/output residual | exploratory | Combine latent and output residuals only with a detectable-subspace or minimum-gain proof | More signals alone do not establish an advantage | Prior exploratory discussions |
| Ding-style dynamic threshold | exploratory | Derive it from fault-free residual dynamics and use empirical calibration only as an auxiliary step | Required for a theoretical dynamic-threshold claim | Prior exploratory discussions |
| Low-dimensional post-filter | exploratory | Separate nominal attenuation from admissible fault gain and address output dimension or rank | Direct SDP formulations may degenerate | Prior exploratory discussions |
| Completely unknown fault direction | alternative | Make no directional prior | Incompatible with a preselected structured direction set | Prior exploratory discussions |
| Structured lifted disturbance basis | alternative | Use standard or grouped basis vectors as candidate latent disturbance channels | Supplies structure but not physical component identity | Prior exploratory discussions |
| Controller command versus applied actuator input | required distinction | Register $\boldsymbol u_k^c$ and $\boldsymbol u_k^a$ separately whenever actuator faults are modeled | The model command and plant input differ under an actuator fault | Web writing-preference inventory |
| Multi-step Koopman loss | exploratory | Tie its horizon and weights to long-horizon normal propagation and the deployed residual | A generic auxiliary forecast loss is not itself a diagnosis contribution | Web writing-preference inventory |

## Selected manuscript route

The final route is not yet fully confirmed. The active manuscript must select and record:

- the causal history encoder and Koopman state definition;
- the physical-to-latent output or decoder relation;
- premise variables and membership construction;
- whether the measurement-decoupled reference is used;
- residual components and horizon;
- the sensor, actuator, and process-fault entry equations;
- the controller-command and applied-input convention;
- dynamic-threshold uncertainty decomposition;
- whether post-filtering is part of the central contribution;
- whether the fault route is direction-free or uses a structured lifted basis.

## Alternatives that must remain separate

- A completely unknown-direction formulation must not be combined with optimization over a prescribed fault direction.
- A structured lifted standard basis must not be described as having no structural prior.
- A measurement-updated reference must not be described as measurement-decoupled.
- A rolling quantile or covariance threshold must not be called a Ding-style dynamic threshold.
- A decoder Jacobian first-order analysis must not be promoted to a global physical fault map.

## Terminology and notation

| Concept | Selected term or symbol | Rejected alternative | Scope |
|---|---|---|---|
| Normal reference independent of current measurements | measurement-decoupled normal reference trajectory/branch; 测量解耦正常参考轨迹/分支 | protected reference/branch; 受保护参考/分支 | This case if the route is confirmed |
| Estimator correction quantity in Chinese | 新息项 | 创新项 | Chinese manuscript |
| Lifted state | a distinct latent-state symbol | automatic reuse of the physical state symbol | This case |
| Latent output relation | learned output map, decoder, or separately defined matrix | unexamined reuse of the physical output matrix | This case |

## Formal claims

Potential results require separate proofs:

- nominal and initialization contributions decay under a uniform contraction condition;
- attention–T–S weight trajectories admit a uniform propagation bound;
- scheduling or rule-weight error produces a computable residual-shift upper bound;
- a joint residual reduces the relevant null space or improves a computable minimum detectable gain;
- the dynamic threshold bounds the fault-free residual under the admitted uncertainty set;
- a fault-side lower bound exceeds the nominal threshold under a stated sufficient condition;
- a post-filter improves a defined nominal-to-fault performance interface over an admitted model and scheduling set;
- approximating the post-filter or a hypernetwork gain has a bounded performance loss.

None of these is a reported theorem until assumptions, computable quantities, and proofs are complete.

## Trainability and implementation

The encoder, lifted coordinates, premise features, cluster centers, memberships, and local consequents may change together. Fixed preprocessing is valid only if the representation it depends on is genuinely fixed.

Avoid full Hessians, repeated spectral decompositions, online clustering, online SDP, large inverses, and long-window high-dimensional optimization. Prefer structured parameterizations, tractable norm bounds, offline filter design, cached local models, and short recursions.

Offline work may include encoder and predictor training, prototype or rule learning, local-matrix estimation, post-filter solution, threshold calibration, fault-basis selection, and stability or robustness bound estimation. Online work should be limited to encoding, optional measurement-decoupled reference propagation, membership and local-model weighting, joint-residual evaluation, fixed post-filtering, recursive thresholding, and the admitted decision or isolation score.

## Experiments

Required evidence depends on the final claims and may include:

- false-alarm and missed-detection rates;
- detection delay across fault magnitudes;
- normal-threshold coverage across operating regimes;
- weakest-direction or minimum-singular-value cases;
- comparisons among raw, joint, and post-filtered residuals;
- ablations for attention, memberships, reference decoupling, thresholds, and post-filtering;
- simultaneous actuator and sensor faults only if simultaneous diagnosis is claimed.

No numerical outcome is confirmed in this case.

## Unresolved questions

- Exact manuscript title and project path.
- Whether attention is central or only an implementation component.
- Whether measurement-decoupled reference propagation is selected.
- Whether the joint residual supports a nontrivial detectability theorem.
- Whether post-filtering remains a core contribution.
- Whether the paper assumes a structured lifted fault basis.
- Which TFS-specific fuzzy-system property forms the central journal fit.

## Reusable insights and promotion status

The reusable parts of this discussion have been generalized in:

- `references/domains/fault-diagnosis.md`;
- `references/technical-validity-and-implementation.md`;
- `references/living-user-rules.md`;
- `references/typical-errors.md`.

This case retains the paper-specific combination, decision history, and unresolved alternatives.
