# Published style exemplar: TDN decoupled residuals

- Corpus ID: P07
- Paper: *Generation of Uncorrelated Residual Variables for Chemical Process Fault Diagnosis via Transfer Learning-Based Input–Output Decoupled Networks*
- Case role: published-style exemplar
- Task group: residual decoupling, fault detection, and estimation
- Positive-use tier: primary macro-to-micro and notation source

## Architecture worth retaining

The Introduction moves from model-based decoupling to the data-driven gap,
separates FD, FI, and FE, identifies two fundamental issues, gives the overall
TDN response, and only then decomposes IDN and VAE roles. A Notations paragraph
precedes the preliminaries/problem section. The method is one coherent TL-based
IDN section rather than one main section per neural module.

## Curated micro-examples

Source, PDF p. 2:

> “This is realized through diagonalization and parallel computation, yielding
> uncorrelated residuals.”

Use: passive voice foregrounds the operation; the participial clause gives its
immediate, defined consequence.

Source, PDF p. 2:

> “Fig. 1 displays the basic structure of VAEs. The forward propagation ... can
> be expressed by ...”

Use: orient once with the figure, then replace visual summary with the governing
mapping and immediate symbol definitions.

## Narrative pattern

`decoupling value -> nonlinear/big-data limitation -> missing data-driven
decoupler/migration -> overall TDN -> IDN constraint -> VAE-guided transfer ->
faulty-to-normal output -> covariance and task evidence`

## Do not imitate

Do not treat uncorrelated residuals as independent without proof, or transfer
the paper's diagonalization claim to a different architecture. Verify every
notation family against the current context registry.
