# Stage 1: Idea exploration

This is the authoritative rule document for the idea-exploration stage. Read it whenever a new paper idea, model component, theoretical route, contribution statement, or major technical revision is being explored. These checks are iterative gates, not a one-time review performed after the paper is written.

The purpose of this stage is to decide what is genuinely new, under which assumptions it is valid, whether it can be trained and implemented, what theoretical conclusion can be supported, and what evidence would be needed. It does not yet turn unresolved ideas into manuscript claims.

## Contents

- Gate A: prior-art overlap
- Gate B: assumptions and realizability
- Gate C: model-specific theoretical contribution
- Repetition points
- Stage output

Use `assets/templates/idea-assessment.md` to record the result of applying these universal checks. Do not copy the checks themselves into a paper case.

## Gate A: has the idea or result already been done?

Before calling an idea a contribution, define the candidate novelty at the correct technical granularity:

- problem and task;
- model class and information setting;
- assumptions and data regime;
- mathematical construction or mechanism;
- theoretical result or guarantee;
- implementation route;
- experimental capability.

Search and compare current primary literature. Record the search date, databases or sources, search terms, closest papers, and the exact overlap and difference. Distinguish:

- the same method and substantially the same result;
- a related method under different assumptions;
- the same model components but a different mechanism or guarantee;
- an adjacent application without the proposed analysis;
- a claimed difference that is only notation, packaging, or module combination.

If prior work already contains the same substantive construction or theoretical result under equivalent assumptions, that item cannot be claimed as an original contribution. It may remain background, an implementation choice, a baseline, or the starting point for a genuinely new extension.

Conversely, sharing high-level components with prior work does not automatically eliminate novelty. The paper must identify a defensible novelty delta: what is newly constructed, derived, guaranteed, made computable, or made possible, and why the difference is technically consequential.

Do not infer novelty merely because no paper has exactly the same complete architecture. Almost every implementation differs in some detail. The relevant question is whether the paper produces a new nontrivial mechanism, analysis, condition, or capability beyond the closest prior work.

Until the literature comparison is sufficiently complete, mark novelty as `unresolved` or `exploratory`. Avoid absolute priority claims such as “the first,” “for the first time,” “pioneering,” and “fills the gap”; state the exact novelty delta with verbs such as “constructs,” “establishes,” “derives,” “provides,” or “applies ... to ...”.

## Gate B: what assumptions are required, and can the idea be implemented?

For each candidate idea or result, complete the authoritative audit in `references/technical-validity-and-implementation.md`. Record its outcome in the idea assessment under four headings: information and assumptions, trainability, offline/online computation, and validation.

For every assumption, ask:

- Is it physically plausible?
- Can it be checked or estimated from the available data?
- Is it required only for proof convenience, or is it inherent to the method?
- Does the online algorithm need information that the problem statement declares unavailable?
- Does the assumption become unrealistic under the target operating conditions?
- What happens when the assumption is violated or only approximately satisfied?

When the matching domain guide defines mutually exclusive routes, surface them explicitly in the feasibility result. Do not silently add structure to rescue an idea while continuing to describe the original information setting; classify the added assumption as a separate alternative and state how it changes the claim.

An idea is not ready to become a manuscript contribution until there is a credible path from the stated information and assumptions to training, numerical computation, online use, and validation. If a route is theoretically interesting but not yet realizable, preserve it as `exploratory` rather than presenting it as a completed method.

Use `technical-validity-and-implementation.md` for the complete technical audit.

## Gate C: can model-specific analysis count as a contribution?

Yes. A theoretical derivation tailored to the paper's particular model structure can be a legitimate contribution even when it is not a broad, general theorem. Model specificity does not disqualify an analytical contribution.

However, the fact that no other paper has exactly the same model is not sufficient by itself. The model-specific analysis should:

- derive a nontrivial consequence that is not true by definition;
- use the distinctive structure of the proposed model in an essential way;
- state all assumptions and the exact scope of the conclusion;
- contain a complete and checkable derivation or proof;
- produce a computable condition, interpretable property, implementable design rule, error bound, detectability result, stability result, or other testable consequence;
- be compared with the closest existing analyses so that the new step is identifiable;
- be validated numerically or experimentally when the conclusion is testable.

Classify formal claims using the sole taxonomy in `references/living-user-rules.md`. A narrow model-specific contribution may instead be presented as a property, analytical result, derivation, or theoretical analysis when a theorem-like environment would overstate its role.

In Chinese discussion, “推理” may describe the research reasoning process, but formal manuscript labels should normally use “理论分析,” “推导结果,” “性质,” “命题,” or another standard term matching the rigor. Renaming an unsupported statement does not make it valid: any claimed guarantee still requires assumptions and a complete derivation.

## Repetition points

Run all three gates:

- when the idea is first proposed;
- whenever the model structure or information setting changes;
- before selecting the paper's core contributions;
- before promoting an exploratory route to `confirmed`;
- before drafting a theorem, proposition, or central analytical claim;
- after the closest literature has been updated;
- when experiment design reveals that a claim is not testable;
- before finalizing the title, abstract, Introduction contributions, and Conclusion.

Record each pass in the matching case file and summarize the current decision in `MANUSCRIPT_CONTEXT.md`. A failed gate does not require deleting the idea; downgrade it, revise its assumptions, separate it as an alternative, or redefine the contribution precisely.

## Stage output

Before moving to journal-paper writing, the project should have:

- a defined technical problem and information boundary;
- a current closest-literature comparison;
- two or three candidate contribution themes with explicit novelty deltas;
- an assumption and realizability ledger;
- selected and rejected technical routes;
- appropriately scoped candidate analytical results;
- a validation plan;
- unresolved items clearly marked rather than hidden.

The idea may move to Stage 2 when its central route is sufficiently `confirmed`. Unresolved secondary items may remain, but the title, abstract, contributions, and formal claims must not present them as completed results.
