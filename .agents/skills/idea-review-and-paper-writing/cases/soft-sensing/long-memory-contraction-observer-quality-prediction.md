# Case: long-memory contraction observer for quality prediction

## Case identity

- Case ID: `soft-sensing/long-memory-contraction-observer-quality-prediction`
- Task group: soft sensing
- Working title: to be confirmed
- Target journal: unresolved
- Manuscript language: Chinese and/or English, depending on the active task
- Project path or link: unresolved
- Active `MANUSCRIPT_CONTEXT.md`: unresolved
- Related domain guidance: `references/domains/soft-sensing-and-observers.md`
- Last explicit decision date: 2026-07-29

## Contents

- Positioning and central problem
- Information and data boundaries
- Decision history and selected route
- Incompatible alternatives
- Formal claims and implementation
- Experiments and unresolved questions
- Reusable insights

## Positioning and central problem

This case studies quality-variable prediction when important process dynamics or quality states are not directly measurable and the data exhibit long temporal dependence. The explored route combines a learned memory state with a structured nonlinear observer or coupling layer.

Potential contribution themes are:

- an output-preserving coordinate transformation with an explicit observer gain that separates the admitted measured and unmeasured coordinates;
- a column-wise contraction condition for the unmeasured-state error channel, together with an appropriately scoped nonlinear stability result;
- a realizable long-memory soft-sensing workflow whose ordinary training and online prediction do not require second-order backpropagation.

## Information and data boundaries

- Process measurements and known inputs may be available online.
- Quality variables may be delayed, sparse, laboratory measured, or available only as training labels.
- Learned GRU or LSTM hidden states are latent states, not automatically physical states.
- Unmeasured physical states remain unavailable unless an observer or coordinate assumption reconstructs them.

## Decision history

| Topic | Status | Current decision | Reason or evidence | Date or discussion source |
|---|---|---|---|---|
| GRU/LSTM hidden state equals physical state | rejected | Treat it as a learned latent dynamic state | Equality lacks a coordinate justification | Prior exploratory discussions |
| Linear output relation in latent coordinates | exploratory | Allow it only with an explicit output-map assumption | The coordinate relation must be established | Prior exploratory discussions |
| Measurable/unmeasurable state partition | exploratory | Define and justify the partition before using a block output matrix | It cannot be chosen only for proof convenience | Prior exploratory discussions |
| Column-wise contraction layer | exploratory | Constrain a specified block or error channel with a uniform margin | A local block constraint does not stabilize the entire observer | Prior exploratory discussions |
| Long-memory network proves observer stability | rejected | Separate memory representation performance from observer convergence | They are different claims | Prior exploratory discussions |
| Integral Jacobian in routine training or deployment | rejected | Retain it as a nonlinear proof device and use a first-order or two-stage realizable training route | Full Jacobians or higher-order derivatives create avoidable cost and are not needed online | Web writing-preference inventory |
| Exactly known unmeasured initial state | rejected | Permit zero, random, nominal, or learned initialization and analyze the initial error | Exact initialization is an unnecessarily strong assumption | Web writing-preference inventory |

## Selected manuscript route

The active manuscript must confirm:

- the meaning of the learned state;
- which quality variables are delayed or unavailable online;
- the physical or learned output relation;
- the measurable/unmeasurable state partition;
- the exact observer error recursion;
- the constrained matrix block, norm, weighting, and margin;
- the role of the memory network;
- whether an integral Jacobian is used only in the proof;
- a training route without full Hessians or second-order backpropagation;
- the initialization rule for the unmeasured coordinates;
- offline and online computations.

## Alternatives that must remain separate

- A latent-state predictor without an observer proof must not be described as a stable nonlinear observer.
- A local linearized error proof must not be described as global nonlinear stability.
- A soft contraction penalty must not be described as a hard stability certificate.
- A constraint on one coupling block must not be generalized to the complete network without a full error analysis.

## Formal claims

Potential results include local or nonlinear error convergence under explicitly stated assumptions, a uniform induced-norm contraction for a selected subsystem, and quality-prediction performance under delayed observations. Their assumptions and experiments must remain separate.

## Trainability and implementation

When normalized column budgets are used, specify the constrained columns, zero-column treatment, differentiability, uniform margin, and whether enforcement is structural or regularized. Quantities such as $\boldsymbol\Phi_{k+1}$ or $\mathcal I_k$ must have explicit members and justified dependencies.

Avoid unavailable physical states, repeated full Jacobians, costly spectral operations, and online high-dimensional optimization. State the deployed history update, forward pass, correction, and quality-prediction recursion.

## Experiments

Relevant tests may include:

- long-delay and long-horizon prediction;
- sparse or delayed quality-label robustness;
- initial-error decay;
- contraction-layer ablation and constraint verification;
- separation of predictive accuracy and observer stability;
- operating-condition generalization;
- online runtime and memory.

No numerical outcome is confirmed in this case.

## Unresolved questions

- Exact project identity and target journal.
- Whether the observer route proves full nonlinear or only local stability.
- The source and justification of the state partition.
- Whether the contraction layer is a hard parameterization or soft penalty.
- Which long-memory architecture is selected.

## Reusable insights and promotion status

The reusable rules are generalized in:

- `references/domains/soft-sensing-and-observers.md`;
- `references/technical-validity-and-implementation.md`;
- `references/living-user-rules.md`;
- `references/typical-errors.md`.

This file retains the paper-specific combination and unresolved research route.
