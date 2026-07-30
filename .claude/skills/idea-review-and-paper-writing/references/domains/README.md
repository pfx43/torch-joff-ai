# Domain guidance

Domain files preserve reusable technical guidance without turning a particular paper route into a global writing rule.

## When to read a domain file

Read a domain file when the manuscript's central model, claim, theorem, residual, controller, estimator, loss, or experiment belongs to that research direction. Do not load unrelated domains merely because they use a common neural-network or fuzzy-system component.

Domain guidance contains three kinds of material:

- invariants and cautions that normally apply throughout the domain;
- admissible modeling routes that require explicit assumptions;
- incompatible alternatives that must not be combined without a new derivation.

Before applying a route-specific rule, check the active `MANUSCRIPT_CONTEXT.md`. If the route is not `confirmed`, treat it as exploratory guidance only.

Paper-specific discussions do not belong in this directory. Store each distinct manuscript as one file under `cases/<task-group>/`, and promote only the generalized domain lesson back into a domain document.

## Available domains

- [`fault-diagnosis.md`](fault-diagnosis.md): model-based and data-driven fault detection, isolation, residual generation, thresholds, post-filtering, Koopman lifting, T–S memberships, and unknown-fault boundaries.
- [`soft-sensing-and-observers.md`](soft-sensing-and-observers.md): quality-variable prediction, latent dynamic states, nonlinear observers, partial measurements, contraction constraints, and long-memory networks.

## Reserved future domains

The following directions have been identified but do not yet have enough confirmed reusable rules:

- model predictive control;
- data completion and missing-data reconstruction.

Add a file only after concrete rules are supplied or derived from an active project. Do not create generic content merely to populate the directory.

## Adding a domain

Use a short descriptive filename. Organize the document in this order:

1. scope and trigger conditions;

2. domain-wide information and modeling boundaries;

3. compatible modeling routes;

4. mutually exclusive alternatives;

5. theory and implementation checks;

6. experiment and evidence requirements;

7. items that must be recorded in `MANUSCRIPT_CONTEXT.md`.

Promote a rule to `living-user-rules.md` only when it transfers across domains. Put one-paper symbols, equations, architectures, datasets, and temporary hypotheses in the project context instead.
