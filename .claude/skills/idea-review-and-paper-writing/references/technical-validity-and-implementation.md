# Technical validity and implementation

Read this document whenever a manuscript task evaluates or changes a model, theorem, observer, detector, controller, loss, constraint, threshold, training loop, or online algorithm. It is a cross-manuscript audit, not a source of project-specific assumptions.

## Contents

- Information availability
- Training-loop closure
- Offline and online separation
- Stability and conclusion scope
- Designable quantities and numerical realization
- Computationally practical constraints
- Contraction and normalization
- Formula and proof audit
- Experiment and evidence audit

## First-principles information audit

Before extending a method or adding formulas, identify:

- why each proposed structure is needed and which failure mode, uncertainty, or information limitation it addresses;
- what role every variable, matrix, loss term, regularizer, constraint, and module plays in the argument;
- which variables are physically measured online;
- which inputs or scheduling variables are known online;
- which quantities exist only as training labels;
- which states, faults, counterfactual trajectories, or disturbances are unknown;
- which matrices, gains, thresholds, bounds, and initial conditions can be computed from available data;
- which objects are design variables and which are fixed by the plant, data, or learned representation.

Do not silently use:

- a true hidden state when only outputs are measured;
- an unknown fault signal, direction, onset time, or fault-free counterpart;
- a counterfactual fault-free scheduling trajectory as an online known variable;
- a plant Jacobian, physical fault matrix, or true decoder Jacobian as a freely optimized parameter;
- fault labels for tuning when the stated training setting uses normal data only.

For each theorem or algorithm, state how every required quantity is obtained. If it is estimated, bounded, sampled, or learned, state the approximation and the information it uses.

Separate consequences forced by the model structure from conclusions that require an additional assumption. A derivation that only renames a quantity, expands a definition, or restates an imposed constraint is not a nontrivial analytical result.

## Training-loop closure

Audit dependencies that change during learning. In particular:

- a latent representation changes when its encoder changes;
- cluster centers, fuzzy prototypes, premise features, memberships, and local models defined in that representation may therefore need joint or alternating updates;
- a quantity called “pretrained” or “fixed” must genuinely remain valid after downstream representation changes;
- constraints in a theorem must be enforceable by the actual parameterization, loss, projection, or certified bound used in training;
- gradients must exist for every operation claimed to be trained end to end.

Do not present a mathematically attractive condition as a practical learning method unless the manuscript explains how it is evaluated and imposed.

## Offline and online separation

List offline and online computations separately. A realizable online method should normally rely on forward passes, fixed or recursively updated low-dimensional objects, matrix–vector products, and bounded-window operations.

Scrutinize any online use of:

- reclustering;
- semidefinite programming or repeated LMI solution;
- singular-value or eigendecomposition;
- large matrix inversion;
- long-window high-dimensional optimization;
- repeated parity-space reconstruction;
- high-order derivatives;
- full Jacobian storage.

When possible, move expensive design steps offline, precompute fixed filters or local models, cache reusable matrices, use short recursions, and exploit low-rank or structured parameterizations. Report both offline design cost and online per-sample cost when complexity is part of the contribution.

## Stability and conclusion scope

Never infer full nonlinear or global stability from a linearized error equation alone.

If the proof studies a local linearization, describe the result as appropriate:

- local stability;
- regional stability;
- local exponential stability along a specified trajectory;
- convergence of a first-order error approximation.

A nonlinear stability claim must account for the relevant nonlinear remainder, Jacobian variation, Lipschitz or incremental bound, invariant region or domain of attraction, and a uniform stability or contraction margin.

State explicitly whether a result is local or global, linearized or fully nonlinear, pointwise or uniform, one-step or multi-step, and tied to a fixed, known, estimated, or uncertain scheduling sequence. Do not transfer a result from one category to another without an additional argument.

Separate directly computable deterministic quantities from population-level statistical guarantees. Likewise, distinguish empirical behavior on the training or validation samples from a uniform guarantee over a stated domain or distribution.

Initial unmeasured states need not be known exactly. When zero, random, nominal, or learned initialization is used, show:

- how initial error enters the error dynamics;
- why the stated stability condition attenuates it;
- whether it affects finite-horizon detection or prediction performance.

When a method is intended to suppress nominal effects while retaining faults or informative inputs, identify the precise propagation operator. Prove decay of the nominal or initialization term and preservation, separation, or amplification of the target term. For combined residuals or measurements, use null-space, detectable-subspace, rank, conditioning, transfer-operator, or minimum-gain arguments rather than saying that “more information” is available.

## Designable and non-designable quantities

For every inequality, constraint, or optimization, classify each term as:

- fixed by the physical system or data;
- learned from data;
- explicitly designable;
- estimated or upper-bounded;
- unknown and unavailable.

State which conditions are hard constraints, which are structural parameterizations, which are soft regularizers, and which are post-training checks. Do not describe a soft penalty as a certified hard guarantee.

For an integral, expectation, infinite-horizon objective, or distributional bound, state its numerical realization, such as finite-window summation, sample averaging, mini-batch approximation, Monte Carlo estimation, quadrature, or offline precomputation.

For Jacobian, Hessian, matrix-inequality, or implicit-optimization expressions, state whether the implementation uses automatic differentiation, Jacobian–vector products, vector–Jacobian products, sampling, a tractable upper bound, a projection, or an offline numerical solver. A symbolic expression alone is not an implementation.

## Computationally practical constraints

For deep-network training and online inference, avoid by default:

- full Hessians and explicit second-order derivative matrices;
- optimization of a maximum eigenvalue;
- repeated singular-value decomposition;
- full spectral-radius calculation;
- explicit matrix inverses;
- complicated invertibility constraints;
- repeated large-scale LMIs during training;
- online clustering or high-dimensional semidefinite programs.

Use them only when they are indispensable, dimensionally controlled, differentiable where required, and accompanied by an implementation and complexity analysis.

Prefer practical substitutes when they support the required conclusion:

- Frobenius, weighted one-, or infinity-norm bounds;
- elementwise absolute-value constraints;
- row-sum, column-sum, or grouped budget bounds;
- structured and low-rank parameterizations;
- differentiable hard-constraint layers;
- matrix–vector products;
- Jacobian–vector or vector–Jacobian products;
- first-order automatic differentiation;
- offline projection or precomputation.

A Frobenius upper bound is not an exact spectral norm. State the implemented quantity and prove only the guarantee it actually supports.

## Contraction and normalization

When stability relies on row, column, block, or induced-norm contraction, state:

- the exact matrix or subsystem being constrained;
- whether normalization acts by row, column, group, block, or whole matrix;
- the norm and any weighting matrix;
- a uniform margin such as $\|A_k\|\leq 1-\varepsilon$ with $\varepsilon>0$;
- treatment of zero rows, zero columns, and numerical stabilizers;
- differentiability and behavior during training;
- whether the condition is enforced by parameterization, projection, or regularization.

Prefer a parameterization that satisfies a required column or group budget throughout training over an unsupported after-the-fact claim. A contraction proved for one block or channel does not automatically stabilize the full nonlinear network or observer.

Stability of each local matrix does not by itself establish stability of an arbitrary time-varying convex interpolation. When that stronger conclusion is needed, provide a common certificate, such as a common Lyapunov function or a uniform induced-norm contraction, and state its norm, weighting, constants, margin, and bounded inputs.

If a neural parameterization is claimed to certify a strict spectral or induced-norm bound during learning, establish differentiability at the admitted points and derive the differential or backpropagation expression actually used to train it. If only a tractable upper bound is optimized, name that upper bound and scope the certificate accordingly.

## Formula and proof audit

Before delivery:

- define every symbol and complete every index range;
- define sets, matrix blocks, dimensions, initial conditions, and data sources;
- for an index set such as $\mathcal I_k$, enumerate the admitted members or give an unambiguous construction and state whether entries of blocks such as $\boldsymbol A_{12,k}$ and $\boldsymbol A_{22,k}$ are included;
- state complete assumptions before each formal result;
- derive the error or propagation system, show the decisive inequalities and recursion, and carry the argument to the stated conclusion;
- name the norm, comparison result, contraction principle, or stability theorem used;
- distinguish local Jacobian analysis from a full nonlinear conclusion;
- state the evaluation point and remainder when using a decoder or model Jacobian;
- simplify stacked multi-step notation when auxiliary block matrices obscure the main argument;
- verify that every theorem-like environment follows the taxonomy in `living-user-rules.md`.

Do not use “obvious,” “straightforward,” “readily obtained,” or a one-line “by the cited theorem” statement to bypass a decisive derivation. If a cited result is used, verify its assumptions in the present model and show the substitution or recurrence that produces the claimed conclusion.

## Experiment and evidence audit

Never fabricate data, results, hardware, datasets, or citations. When results are unavailable, write only:

- the experimental protocol;
- data partition;
- baselines;
- ablations;
- metric definitions;
- validation steps;
- the property each experiment is intended to test.

Do not write “the experiments demonstrate” until results exist.

Experiments must test the paper's actual theoretical and methodological claims, not only average predictive accuracy. Depending on the claim, include:

- nominal coverage and false-alarm rate for thresholds;
- missed-detection rate and detection delay;
- operating-condition robustness;
- sensitivity to memberships, scheduling, noise, and uncertainty;
- weakest detectable direction or equivalent minimum gain;
- reconstruction RMSE or MAE;
- conditioning or residual-bound metrics;
- ablations for each claimed module or constraint;
- runtime, memory, or per-sample complexity when implementation is claimed.

Define every metric mathematically. If training is stated to use normal data only, do not use fault labels to tune hyperparameters, select thresholds, or choose checkpoints.

Report the experimental scenario, data split, baselines, ablations, software and hardware conditions relevant to reproducibility, random seeds or repetition policy, hyperparameters, and enough implementation detail to reproduce the claimed comparison.
