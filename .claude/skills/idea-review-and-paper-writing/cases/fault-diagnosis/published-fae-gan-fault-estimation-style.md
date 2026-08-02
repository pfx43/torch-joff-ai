# Published style exemplar: FAE-GAN fault estimation

- Corpus ID: P06
- Paper: *A New Generative Adversarial Networks-Based Fault Diagnosis Framework: Learning a Mapping to Estimate Fault*
- Case role: published-style exemplar
- Task group: fault detection and estimation
- Positive-use tier: primary purpose, mapping, and workflow source

## Architecture worth retaining

The paper defines FD and FE separately, reviews deterministic/probabilistic and
shallow/deep routes, identifies classifier limitations for unseen faults, and
motivates a faulty-to-normal mapping. The method separates detector construction
from the genuinely independent FE extension and gives distinct offline and
online algorithms.

## Curated micro-examples

Source, PDF p. 1:

> “...are fed into the FAE-GAN for mapping training, aiming to learn the
> conditional distribution from faulty to normal.”

Use: a passive procedure foregrounds data flow; the nonfinite phrase states the
training purpose without another rigid sentence.

Source, PDF p. 1:

> “In this way, ... can be eliminated adaptively, thereby achieving the fault
> estimation purpose.”

Use: `In this way` refers to the already explained mapping and `thereby` marks
its task consequence. Both fail when the mechanism is absent.

## Narrative pattern

`classifier limitation -> need for detector/estimator -> faulty-to-normal
mapping -> augmented training pairs -> learned elimination -> FD/FE outputs ->
offline and online algorithms -> separate metrics`

## Do not imitate

Do not copy generated-fault assumptions into a paper that lacks admissible
fault data. Distinguish `fault elimination mapping` from a physical fault model,
and verify whether the term is standard in the target literature.
