# Published style exemplar: AM-DAE time-series imputation

- Corpus ID: P04
- Paper: *Imputation of Missing Values in Time Series Using an Adaptive-Learned Median-Filled Deep Autoencoder*
- Case role: published-style exemplar
- Task group: missing-data completion and time-series imputation
- Positive-use tier: primary mechanism, training-stage, and experiment source

## Architecture worth retaining

The Introduction starts from causes and consequences of missing industrial data,
classifies imputation methods, distinguishes model choice from update strategy,
and identifies supervision and overwhelming-missingness limitations. The body
places the supervised DAE baseline before AM-DAE, then separates method and case
studies.

## Curated micro-example

Source, PDF p. 1:

> “..., which allows the imputation information to be transmitted with the
> training process.”

Use: the relative clause states why a recursive update matters. It should follow
the exact update operation and name the information being propagated.

Source, PDF p. 2, adapted:

> Adaptation: “During early iterations, the loss emphasizes nonmissing-value
> reconstruction; it then shifts toward a reconstruction–imputation tradeoff.”

Use: stage-scoped prepositional opening plus a semicolon expresses temporal
change without a list of short sentences.

## Narrative pattern

`missingness cause -> incomplete-data consequence -> method taxonomy ->
supervision/update limitations -> recursive median-filled mechanism -> adaptive
training emphasis -> time-series stacking -> pattern-matched evidence`

## Do not imitate

Do not generalize simulated missingness patterns to all industrial settings or
claim unsupervised operation when complete targets influence training/model
selection. Keep reconstruction and imputation metrics distinct.
