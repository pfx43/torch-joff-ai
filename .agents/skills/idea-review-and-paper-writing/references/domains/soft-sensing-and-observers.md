# Soft sensing and nonlinear observers

Read this document for quality-variable prediction, virtual sensing, learned latent-state models, nonlinear observers, partial measurements, long-memory networks, and contraction-constrained estimation. Confirm project-specific state partitions, output maps, and stability routes in the active versioned manuscript context.

## Contents

- Latent and physical states
- Information boundary
- Nonlinear observer stability
- Column-wise contraction and normalized coupling
- Long-memory networks
- Theory and implementation
- Experiments
- Project-context requirements

## Latent and physical states

Do not identify a GRU, LSTM, or other learned hidden state $\boldsymbol h_k$ with the physical state $\boldsymbol x_k$ without a stated coordinate, embedding, observability, or reconstruction assumption. A learned hidden state should normally be described as a latent dynamic state.

Likewise, do not assume that a lifted state $\boldsymbol z_k$ equals or contains the complete physical state merely because this simplifies an observer proof.

If the output is written as a linear map of a learned state, state the assumptions under which that coordinate and output structure are valid. If a partition such as

$$
\boldsymbol C=
\begin{bmatrix}
\boldsymbol C_0 & \boldsymbol 0
\end{bmatrix}
$$

is used, explain how measurable and unmeasurable substates are defined and why the partition is available.

## Information boundary

Separate:

- process measurements available at each sample;
- quality variables available only after laboratory delay or as training labels;
- latent states produced by the network;
- unmeasured physical states;
- known inputs and operating conditions;
- offline model parameters and online observer quantities.

Do not use delayed quality labels as if they were current online measurements. State how asynchronous, delayed, sparse, or missing quality observations enter training and validation.

## Nonlinear observer stability

A stability conclusion must cover the complete nonlinear estimation error or be explicitly restricted to a local or first-order result.

When the selected route uses an output-preserving coordinate transformation, derive its invertibility and output-preservation properties, show how it separates measured and unmeasured coordinates, and obtain the observer gain explicitly from the transformed structure. Do not introduce the partition only because it makes a contraction proof convenient.

If the proof uses a time-varying coordinate transformation, state:

- its invertibility conditions and uniform conditioning;
- how time variation enters the transformed error system;
- whether the transformation is known or computable online.

For block matrices such as $\boldsymbol A_{12,k}$ and $\boldsymbol A_{22,k}$, define every element or index entering a constrained set. State whether design objects depend on $\boldsymbol u_k$, $\boldsymbol y_k$, operating variables, or latent histories and justify that dependence.

When project-specific quantities such as $\boldsymbol\Phi_{k+1}$ or an index set $\mathcal I_k$ are introduced, list their members and dependencies explicitly. If their main role is only to certify contraction, do not add measurement or input dependence without a theoretical reason.

Do not introduce measurement dependence into a contraction design merely because the quantities are available. The dependence must follow from the observer structure or improve a stated bound.

The proof should show the error recursion, induced or weighted norm bound, uniform margin, and convergence conclusion. A contraction condition on one block, coupling layer, or unmeasured-state channel does not automatically establish stability of the entire nonlinear observer.

An integral Jacobian along the state segment may be used to derive the exact nonlinear error relation. In the preferred realizable route, this object remains a proof device: it is not evaluated in ordinary training or online prediction. If a proposed loss instead requires the full Jacobian, a second derivative, or a Hessian–vector product, state that cost and replace it with a two-stage, first-order, sampled, or tractable-bound implementation unless the higher-order operation is indispensable.

Do not assume the initial unmeasured state is exactly known. Permit and analyze zero, random, nominal, or learned initialization, and propagate the resulting initial error through the stated contraction bound.

## Column-wise contraction and normalized coupling

When a column-wise budget, weighted one-norm, or normalized coupling layer is used, specify:

- the exact block and error channel it constrains;
- column groups and their index sets;
- the budget and uniform contraction margin;
- treatment of zero columns and numerical stabilizers;
- whether the layer enforces a hard constraint or contributes a soft penalty;
- differentiability and train-time behavior.

Prefer a structural parameterization that maintains the constraint during learning. Relate the layer's mathematical bound to the complete error dynamics rather than claiming that it stabilizes every network module.

## Long-memory networks

Use LSTM, GRU, attention, or other memory mechanisms to represent long-range dependencies when the data support that need. Keep predictive representation capability separate from observer stability:

- memory-network accuracy does not prove estimation-error convergence;
- observer contraction does not prove that the latent state captures all long-term quality dynamics;
- each claim requires its own assumptions, metrics, and experiments.

The main narrative should remain focused on the actual problem, such as delayed or unavailable quality measurements, long temporal dependence, partially unmeasured dynamics, and a structured observer or estimator. Do not add unrelated modules merely to increase architectural complexity.

## Theory and implementation

Classify physical mappings, learned mappings, observer gains, coupling matrices, latent states, and quality outputs distinctly. State which quantities are trained, constrained, estimated online, or fixed offline.

Avoid proof conditions that require unavailable physical states or repeated high-dimensional Jacobians. When a Jacobian bound is necessary, state how it is estimated or enforced and whether a Jacobian–vector product or tractable norm bound suffices.

Report the online path: history update, network forward pass, state correction, quality prediction, and any recursive uncertainty or confidence calculation. Separate this from offline training, hyperparameter selection, and constraint certification.

For the output-preserving, long-memory contraction route, audit explicitly whether:

- delayed quality labels have been used as unavailable current measurements;
- the coordinate transformation and its inverse are valid over the claimed domain;
- a solvability condition involving $\boldsymbol A_{22,k}$ is unnecessarily conservative;
- the phrase “unstable unobservable mode” is stronger than the established evidence;
- the normalized coupling layer bounds a column norm, spectral norm, or another quantity;
- the contraction margin is uniformly positive;
- the LSTM or GRU prediction claim has been confused with observer convergence;
- dependencies of $\boldsymbol\Phi_{k+1}$ or $\mathcal I_k$ on $\boldsymbol u$, $\boldsymbol Y$, or a history object follow from the observer derivation;
- a regularizer is intended for stability, identifiability, or numerical training behavior;
- the matrix inequality has clearly separated designable and fixed quantities;
- the nonlinear stability proof closes every required step.

## Experiments

In addition to general prediction metrics, evaluate claims about:

- long-horizon or long-delay dependence;
- operating-condition generalization;
- robustness to sparse, delayed, or missing quality labels;
- initial-state sensitivity and error decay;
- the effect of the contraction or normalization layer;
- prediction performance versus observer stability;
- online runtime and memory when deployment is claimed.

Use RMSE, MAE, coefficient-of-determination, delay-aware metrics, or uncertainty coverage only after defining them and explaining their relevance. Do not fabricate measurements or turn an experiment protocol into a result.

## Project-context requirements

Record the following in the active versioned manuscript context:

- physical measurements, delayed quality labels, and unavailable states;
- interpretation of every learned hidden or lifted state;
- output-map assumptions;
- measurable and unmeasurable state partitions;
- selected observer and stability route;
- constrained blocks, norms, margins, and parameterizations;
- memory-network role;
- offline and online computations;
- experiments needed to separate prediction and stability claims.
