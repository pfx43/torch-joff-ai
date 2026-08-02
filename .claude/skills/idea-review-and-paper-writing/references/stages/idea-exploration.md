# Stage 1: Idea review

Use this stage whenever a new paper idea, component, theoretical route,
contribution, or major technical revision is being explored. Record each pass
in `idea-assessment.md`; do not turn an unresolved idea into manuscript prose.

## Contents

- Loop overview
- I1: prior-art overlap
- I2: task and information boundary
- I3: assumption and realizability closure
- I4: theoretical nontriviality
- I5: contribution main line
- I6: maturity decision
- Repetition points and output

## Loop overview

| ID | 概括词语 | 简述 |
|---|---|---|
| I1 | 现有工作重合 | 查清最接近工作是否已实质完成相同构造、机制或结果 |
| I2 | 任务边界清楚 | 明确解决什么、不解决什么，以及哪些信息真实可得 |
| I3 | 假设能够落地 | 检查训练、计算、在线使用和验证是否能在已声明假设下实现 |
| I4 | 理论并非平凡 | 分析必须实质利用模型结构并产生可计算、可证明或可检验结果 |
| I5 | 贡献主线集中 | 将创新收敛为两到三个相互衔接的问题—贡献主题 |
| I6 | 成熟度不夸大 | 区分 confirmed、exploratory、alternative、rejected 和 unresolved |

## I1: prior-art overlap — 现有工作重合

Define the candidate novelty at the correct technical granularity:

- problem and task;
- model class and information setting;
- assumptions and data regime;
- mathematical construction or mechanism;
- theoretical result or guarantee;
- implementation route;
- observable capability.

Search current primary literature and record the search date, databases or
sources, terms, closest papers, exact overlap, and exact difference. Distinguish
the same method/result under equivalent assumptions from a related method,
shared components with a new mechanism, an adjacent application, and a merely
notational or packaging difference.

Do not infer novelty from an architecture combination or from the absence of an
identical complete network. Identify what is newly constructed, derived,
guaranteed, made computable, or made possible and why that change matters.
Until the comparison is sufficient, keep novelty `unresolved` or `exploratory`.
Avoid absolute priority language; use exact verbs such as constructs,
establishes, derives, provides, or applies when supported.

## I2: task and information boundary — 任务边界清楚

State the exact scientific or engineering task before selecting modules. Record:

- the target object, input, output, time setting, and operating condition;
- what the paper will establish or compute;
- adjacent tasks explicitly excluded;
- quantities that are measured, known, estimated, learned, bounded, or
  unavailable during design and use;
- labels, future information, counterfactual trajectories, fault variables, or
  physical parameters that cannot be assumed available.

Do not silently add a fault direction, clean reference, true latent state, test
label, future sample, or other unavailable quantity to rescue the route. When a
domain guide offers mutually exclusive information settings, record them as
separate alternatives and explain how each changes the claim.

## I3: assumption and realizability closure — 假设能够落地

Apply `references/technical-validity-and-implementation.md`. For every
assumption ask whether it is physically plausible, observable or estimable,
proof-only or method-essential, stable under the intended operating condition,
and robust to approximate satisfaction.

Trace a credible path from admitted information to numerical computation,
optimization or algorithm execution, and online use. Identify design variables,
data-determined quantities, offline computations, online computations,
differentiability, complexity, initialization, and any higher-order derivative
or repeated optimization cost.

This is a feasibility audit, not experimental-section writing. Stage 1 may
record what kind of evidence would be needed, but it does not draft protocols,
results, or comparisons. Preserve an interesting but unrealizable route as
`exploratory` rather than disguising it as a completed method.

## I4: theoretical nontriviality — 理论并非平凡

A model-specific derivation may be a contribution, but model uniqueness is not
sufficient. Require the analysis to:

- derive a consequence that is not true by definition;
- use the distinctive model structure essentially;
- state complete assumptions and exact conclusion scope;
- provide a checkable proof or derivation;
- produce a computable condition, design rule, bound, stability or
  detectability result, or another testable consequence;
- identify the new step relative to the closest analyses.

Use the formal-claim taxonomy in `references/living-user-rules.md`. A narrowly
scoped result may be a property, proposition, derivation, or theoretical
analysis rather than a theorem. Renaming an unsupported guarantee does not make
it valid.

## I5: contribution main line — 贡献主线集中

Organize candidate contributions around two or three central problem-facing
themes. For each theme record:

`problem -> insufficiency -> construction -> nontrivial consequence -> required evidence`

Merge module-level claims that answer the same problem. Do not promote a loss,
network layer, implementation detail, ordinary data split, background fact, or
unverified adjective to an independent contribution. Three to five concrete
construction/analysis/validation statements may appear later under the two or
three principal themes, but they must not become unrelated main lines.

## I6: maturity decision — 成熟度不夸大

Classify every route:

- `confirmed`: sufficiently supported to enter the conception baseline;
- `exploratory`: promising but missing proof, evidence, or feasibility;
- `alternative`: a coherent route under different assumptions;
- `rejected`: contradicted, redundant, infeasible, or off the main line;
- `unresolved`: awaiting literature, derivation, data, or a user decision.

Only `confirmed` decisions may appear as completed Stage 2 baselines. Keep
unresolved secondary items visible, but do not present them in the title,
abstract, contributions, theorem statements, or Conclusion as established.

## Repetition points and output

Run I1–I6 when the idea is first proposed; the model or information setting
changes; a route is promoted; a theorem or central claim is drafted; closest
literature changes; or Stage 2/3/4 reveals an untestable or unavailable premise.

Stage 1 outputs `idea-assessment.md` and updates the matching paper case. Move
to Stage 2 only with a defined task and information boundary, current
closest-work comparison, two or three candidate themes, selected/rejected
routes, scoped analytical claims, credible realizability, and explicit
unresolved items. Do not create or modify a versioned manuscript context before that
central route is sufficiently confirmed.
