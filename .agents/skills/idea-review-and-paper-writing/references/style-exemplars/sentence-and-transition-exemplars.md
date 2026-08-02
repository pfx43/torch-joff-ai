# Sentence and transition exemplars

This file synthesizes sentence forms and transitions across the published-paper
cases. Exact excerpts and paper-specific cautions remain in the task-grouped
case files; reusable frames here must still be rewritten for the current paper.

## Contents

- Route to concrete one-paper examples
- Sentence-form palette
- Transition and prepositional-phrase bank
- Modifier and terminology guidance
- Adaptation procedure and prohibitions

## Route to concrete one-paper examples

| Writing need | Task-matched case |
|---|---|
| Problem questions and contribution ordering | [P03 LCP](../../cases/fault-diagnosis/published-lcp-fault-isolation-style.md) or [P05 VAE-ILVM](../../cases/process-monitoring/published-vae-ilvm-monitoring-style.md) |
| Purpose clauses and faulty-to-normal mapping | [P06 FAE-GAN](../../cases/fault-diagnosis/published-fae-gan-fault-estimation-style.md) |
| Macro-to-micro model and equation introduction | [P07 TDN](../../cases/fault-diagnosis/published-tdn-decoupled-residual-style.md) |
| Recursive training mechanism | [P04 AM-DAE](../../cases/data-completion/published-am-dae-imputation-style.md) |
| Chinese problem–method–validation flow | [P09 ELM-AAE](../../cases/fault-diagnosis/published-elm-aae-chinese-style.md) |
| Older fault-classification structure | [P01 EDBN](../../cases/fault-diagnosis/published-edbn-fault-classification-style.md) or [P02 CG-SAE](../../cases/fault-diagnosis/published-cg-sae-fault-classification-style.md) |
| Nonlinear dynamics–circuit–application architecture | [P08 memristive network](../../cases/nonlinear-dynamics/published-memristive-multi-butterfly-style.md) |

Select one matching case before using the cross-case palette below. Do not reuse
a pattern when its connector lacks a concrete antecedent: `In this way` is valid
only after the mechanism has been explained, and `thereby` requires a defensible
causal consequence.

## Sentence-form palette

| Form | Best use | Reusable pattern | Guard |
|---|---|---|---|
| Active voice | Authorial design choice, analysis, or verified observation | `We construct [object] to [purpose].` | Do not begin every sentence with `We` |
| Passive voice | Foreground the object, operation, or reproducible procedure | `[Object] is optimized under [condition].` | Name the agent when responsibility or data provenance matters |
| Relative clause | Attach a mechanism, qualification, or consequence to a defined noun | `[Module], which enforces [property], receives ...` | Avoid chains of nested `which` clauses |
| Adverbial or subordinate clause | State condition, contrast, time, or reason before the main consequence | `Because/When/Although [condition], [response].` | The dependent clause must change the interpretation of the main clause |
| Nonfinite construction | Express purpose, simultaneous action, or immediate result compactly | `To [purpose], ...`; `..., yielding ...`; `..., enabling ...` | Do not use a dangling participle or claim causality without a mechanism |
| Prepositional opener | Locate the claim in a stage, scope, or condition | `During offline training, ...`; `Under Assumption 2, ...` | Avoid a sequence of vague `In terms of` openers |
| Coordination | Join equally ranked consequences or tightly coupled operations | `[Method] reduces X and preserves Y.` | Do not coordinate ideas with different logical levels |
| Apposition | Define an abbreviation or exact role without a separate sentence | `[Object], a [precise class], ...` | Keep the apposition short and non-promotional |

Natural variation comes from choosing among these forms according to meaning.
It is not necessary to use every form in a paragraph.

## Transition and prepositional-phrase bank

Use a phrase only after verifying the relation in the right column.

| Relation | Calibrated options | Required semantic check |
|---|---|---|
| Contrast or limitation | `However,`; `Nevertheless,`; `In contrast,`; `Although [condition], ...` | The new proposition must conflict with or limit the previous one |
| Cause or premise | `Because ...`; `Given [condition], ...`; `Owing to [verified cause], ...`; `Under [assumption], ...` | The cause or assumption must be sufficient for the stated dependency |
| Purpose or response | `To address this limitation, ...`; `For this purpose, ...`; `Motivated by these issues, ...`; `To this end, ...` | `this` or `these` must name an immediately recoverable problem |
| Refinement | `Specifically,`; `More precisely,`; `At the component level, ...`; `In mathematical terms, ...` | The next sentence must narrow, formalize, or instantiate the previous claim |
| Consequence | `Consequently,`; `Thus,`; `As a result,`; `thereby ...`; `yielding ...` | A mechanism or derivation must support the consequence |
| Continuation | `Moreover,`; `In addition,`; `Beyond [first task], ...` | The added point must be distinct but parallel, not a disguised causal claim |
| Stage or scope | `During offline training, ...`; `At test time, ...`; `For the jth rule, ...`; `Within the admissible region, ...` | The scope must be defined and maintained throughout the sentence |
| Evidence | `To test this property, ...`; `Consistent with [prediction], ...`; `Compared with [baseline], ...` | The test, prediction, and baseline must be specified before interpretation |
| Limitation | `This conclusion is restricted to ...`; `The result does not imply ...`; `Outside this region, ...` | The boundary must match the theorem, data, or experimental protocol |

## Precise modifiers

Prefer modifiers that can be traced to a definition, computation, test, or
structural property:

- structural: `decoupled`, `causal`, `input-conditioned`, `dimensionally
  compatible`, `nonvanishing`;
- computational: `offline-computed`, `recursively updated`, `closed-form`,
  `computationally tractable`;
- statistical: `distribution-free`, `weakly dependent`, `confidence-calibrated`,
  `statistically significant` only after an appropriate test;
- evidential: `empirically observed`, `theoretically guaranteed`, `numerically
  verified`, with the evidence type kept explicit;
- temporal or scope: `one-step`, `multi-step`, `local`, `uniform`, `asymptotic`,
  `finite-sample`.

Useful adverbs include `explicitly`, `jointly`, `recursively`, `separately`,
`asymptotically`, and `empirically` when they change the scientific meaning.
Remove `remarkably`, `obviously`, `clearly`, `significantly`, `efficiently`, or
`effectively` when no defined comparison or evidence supports them.

## Corpus-aligned terminology bank

The following expressions recur in the stronger corpus papers and may be useful
when they name the current object exactly:

- `structural fault detectability`;
- `monitoring metric` or the target field's established alternative;
- `faulty-to-normal mapping`;
- `uncorrelated residual variables`;
- `input–output decoupled network`;
- `multivariate time-series data`;
- `reconstruction error` or `reconstruction residual`, kept distinct;
- `offline modeling`, `model selection`, `final testing`, and `online
  monitoring` as separate stages;
- `maximum mean discrepancy` when the actual loss is used;
- `normal-operation`, `fault-free`, or `nominal` only after identifying the
  precise object and regime.

Verify every field-specific term in current peer-reviewed literature before
introducing it. Do not use this bank to coin an acronym or replace a more
standard term in the target community.

## Synthetic before-and-after calibration

The following is an adaptation, not a source quotation.

Rigid version:

> The network contains a decoupling layer. The layer generates residuals. The
> residuals are used for fault estimation. The loss trains the network.

Purpose-led revision:

> Because coupled residuals confound the fault channels, an input–output
> decoupling layer is introduced before the estimator. By constraining each
> residual to depend on its corresponding input, the layer yields a structured
> residual vector that can be used for fault estimation. The resulting mapping
> is learned offline through the loss defined in (12), while online inference
> requires only one forward pass.

The revision is better because the limitation precedes the response, the
relative and nonfinite constructions express actual relations, and each
sentence advances one paragraph purpose. It is not better merely because it is
longer.
