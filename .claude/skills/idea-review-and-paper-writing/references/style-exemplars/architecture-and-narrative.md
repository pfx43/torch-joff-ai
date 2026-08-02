# Architecture and narrative exemplars

Use this file to plan information order. It summarizes the strongest recurring
structures in P03–P07 and P09; it does not replace the authoritative chapter
rules or the frozen context blueprint.

## Contents

- Compact paper architecture
- Paragraph-level causal architecture
- Method-description progression
- Experiment narration
- Cross-section synchronization
- Failure patterns and use boundary

## Compact paper architecture

### Introduction

Use the following dependency chain rather than a list of topics:

1. Establish the application-level need and the consequence of failure.

2. Define the exact technical task and distinguish it from adjacent tasks.

3. Narrow the method taxonomy toward the family relevant to the paper.

4. Review representative mechanisms, not author-by-author summaries alone.

5. Identify the shared limitation and explain why it matters under the current
   information or deployment setting.

6. Convert that limitation into two or three focused questions or subproblems.

7. Give one macro-level description of the proposed mapping and its intended
   consequence before naming all internal modules.

8. State contributions in the same order as the questions.

9. End with the roadmap and, when needed, a compact Notation paragraph.

P05 provides a strong question-led transition: a structural detectability
requirement is stated, explicit questions are raised, and those questions are
then converted into contribution themes on PDF page 2. P07 similarly narrows
from missing data-driven decoupling and state-migration capabilities to two
fundamental issues before presenting the overall TDN construction on PDF pages
1–2.

### Preliminaries and problem formulation

Introduce only the standard concepts and information boundary required by the
method. Then define the system or data objects, permitted information, and
central subproblems mathematically. P03, P05, P06, and P07 all use a dedicated
preliminaries/problem-formulation stage before the main construction.

### Proposed method

Use a macro-to-micro ladder:

1. **Overall question and mapping:** what input is transformed into what output,
   under which information restrictions, and for what purpose?

2. **Governing relation:** give the overall mapping, state equation, decoder,
   residual generator, or optimization problem before local neural blocks.

3. **Interface definition:** define dimensions, known and learned quantities,
   parameters, and the output passed to the next component.

4. **Component decomposition:** introduce each module only after stating why the
   overall mapping cannot meet the objective without it.

5. **Mechanism:** explain which quantity the component changes and how that
   change supports the next result.

6. **Training or algorithm:** state the loss, constraint, update order, data
   availability, and offline/online boundary.

7. **Consequence and scope:** close with the property, diagnostic use, design
   implication, or limitation that the next subsection consumes.

P07 gives the overall TDN role before describing IDN and VAE internals, then
introduces the VAE forward map before defining its symbols. P06 states the
faulty-to-normal mapping objective before separating the AE-GAN detector, data
enhancement, FAE-GAN training, and online estimator.

### Genuinely independent second task

Create a separate main section only when the task has its own question, inputs,
output, method, and evidence. P05 separates threshold learning because it is not
merely an internal layer of the VAE indicator. P06 separates fault estimation
because it extends the detector with a distinct learned mapping and workflow.
Do not use this precedent to promote every module to a main section.

### Experiments

For each claim, use this order:

1. name the claim or property under test;

2. define data, split, operating condition, fault or missingness construction,
   baseline, metric, and repetition rule;

3. report the result without evaluative inflation;

4. compare it with the relevant baseline or theoretical expectation;

5. interpret the result through the proposed mechanism;

6. state the remaining limitation or non-tested scope.

P04 explicitly distinguishes different missing patterns and repeated
experiments. P07 adds covariance visualization to test decoupling rather than
relying only on aggregate detection metrics.

## Paragraph blueprints

### Background-to-need paragraph

`context -> failure or constraint -> practical consequence -> technical need`

The paragraph must end with a need that the next paragraph can narrow. Do not
continue adding generic importance claims after the need is established.

### Literature-synthesis paragraph

`method family and strength -> representative mechanisms -> shared limitation -> requirement for this paper`

Organize by mechanism or assumption. Author chronology is supporting evidence,
not the paragraph's sole structure.

### Method-opening paragraph

`focused problem -> overall mapping/architecture -> intended property -> route into components`

This paragraph prevents local module names from appearing before the reader
knows the model's system-level purpose.

### Component paragraph

`why the component is needed -> input and operation -> equation/structure -> changed quantity -> output/interface`

If the first sentence only says that the paper “uses” a module, the motivation
is probably missing.

### Theory paragraph

`question -> assumptions -> formal result -> decisive reasoning -> design or implementation meaning -> scope`

The theorem environment may hold the formal statement, but the surrounding
paragraphs must explain why it matters and what it does not prove.

### Experiment paragraph

`claim -> test design -> observation -> mechanism-facing interpretation -> limitation`

Avoid result narration such as “Fig. X shows...” when the paragraph has not
identified which claim the figure tests.

### Transition paragraph

`output now established -> unresolved requirement -> why the next section is needed`

A transition exists to change dependency level, not to repeat the roadmap.

## Macro-to-micro progression audit

Before accepting a method subsection, verify this order:

- paper-level question;
- model-level input–output mapping;
- subsystem or module role;
- variable and equation details;
- training or numerical realization;
- evidence or downstream use.

Skipping a level is allowed only when the missing level has already been stated
unambiguously and the subsection explicitly links back to it.
