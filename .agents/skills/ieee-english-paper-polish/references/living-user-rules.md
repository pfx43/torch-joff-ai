# Living user rules

Update this file when the user gives a durable manuscript-writing or notation rule that should apply to later papers.

## Current rules

- Use formal English suitable for IEEE Transactions, with connected technical paragraphs and explicit causal transitions.
- Place no displayed equation before Section II; keep the title, abstract, keywords, and Introduction formula-free except for indispensable inline symbols.
- Use the compact main-section sequence Introduction; Preliminaries and Problem Formulation; Proposed Method; an optional section for a genuinely second task; Experiments; Conclusion. Within the method, present theoretical modeling/analysis before data-driven forward modeling and loss design, and finish with the end-to-end workflow; put long proofs in appendices.
- Structure the Introduction as background and problem, modeling-method categories, problem-focused literature and limitations, the core question, contributions, and finally section organization plus notation conventions.
- Keep Section II as background/problem formulation; place paper-specific innovations in Section III or the designated method section. End Section II with a mathematically stated `Problem Formulation` containing two central objectives and no more than three; do not substitute a long prose checklist such as `Monitoring Objectives`.
- Define every symbol at first use, or define a large coherent symbol family in the Notation paragraph at the end of the Introduction.
- Keep one semantic family per base character across lowercase/uppercase and font variants, and keep one object type per visual format. A change from italic to bold, calligraphic, script, or blackboard-bold does not by itself justify an unrelated meaning. Reserve `f`/`F` for fault or faulty quantities, `n` for normal/nominal quantities rather than dimensions, `k` for discrete sample indices, and `u` for control inputs. Use a dedicated dimension prefix such as `m`, keep disturbance/noise symbols disjoint from weight/whitening symbols such as `w`/`W`/`\omega`, and use boldface for vectors/matrices, calligraphic capitals for mappings, blackboard bold for sets/spaces, and one dedicated operator font. Descriptive roman-text subscripts and universally standard operators may be retained when they cannot be mistaken for independent variables.
- Avoid multiple visually similar Greek letters for the same concept; use one observable/lifting symbol family and distinguish scalar/vector members by indices and formatting.
- Use calligraphic capitals for nonlinear mappings and blackboard-bold notation only for number fields, domains, sets, and spaces. Use ordinary bold vectors or matrices—not blackboard-bold letters—for stacked histories, information variables, states, and data arrays.
- Avoid code-style stacking notation such as `\operatorname{col}`. Display vectors and block matrices explicitly with brackets and transposes so their orientation is mathematically unambiguous.
- Within the reserved fault family, use one visually consistent form such as `f`/`\mathfrak f`; keep actuator and sensor fault channels distinct and combine them only in a clearly defined generalized fault vector.
- Reserve `H` for the neural/history-network side; do not reuse it for a physical output mapping.
- Denote the infinite-dimensional Hilbert space explicitly, e.g. `\mathbb H^{\infty}`.
- Avoid symbol collisions between Koopman operators/matrices and attention keys; use disjoint characters or fonts.
- In controlled Koopman preliminaries, show control and measured exogenous arguments consistently in the operator action. Follow the infinite-dimensional definition with a distinct finite-dimensional controlled linear predictor and output/reconstruction equation; do not repeat the operator identity under a new equation number.
- Combine related multi-line equations into one numbered formula using `equation` plus `aligned`/`split`.
- Strengthen stability and sliding-mode proofs rather than relying on informal claims; state all assumptions and bounds.
- Distinguish a residual-dynamics-based theoretical dynamic threshold generator from rolling statistical, covariance-based, adaptive-quantile, or periodically recalibrated thresholds. Use the intended term and derive the corresponding guarantee explicitly.
- Do not fabricate results, data, or references. Preserve technical content unless the user explicitly authorizes deletion.
- Use standard field-specific academic terminology rather than literal or improvised phrases such as `healthy transition`; choose `nominal`, `fault-free`, or `normal-operation` according to the precise object.
- Remove drafting commentary, author-process notes, self-explanatory asides, and previews of where paper-specific material will appear. State the necessary definition or argument directly; retain only conventional roadmaps and technically necessary cross-references.
- Record only cross-manuscript writing and reasoning rules in this file; do not turn one paper's temporary notation, equation numbering, or technical design choice into a global rule.

## Formal-claim taxonomy (authoritative)

Use this as the only taxonomy for theorem-like environments. Do not restate a parallel taxonomy elsewhere in the skill or manuscript instructions.

- Reserve **Theorem** for a central, genuinely original, broadly applicable result of the paper. State complete assumptions and an exact conclusion, and provide a rigorous proof; move a long proof to an appendix. A routine consequence, narrowly scoped result, or informally justified claim is not a theorem.
- Present an imported supporting result from prior work as a **Lemma**, with its assumptions and an exact citation, and make its non-original status explicit. Do not relabel a cited result as the paper's own theorem. If the manuscript materially adapts or extends it, identify and prove the new part separately.
- Use **Corollary (推论)** for a result derived directly from a cited lemma or a theorem established in the manuscript. Name the parent result and provide an explicit proof, even when the proof is brief.
- Use **Proposition (命题)** for a verifiable but narrower, less general, or more context-dependent mathematical result. State its assumptions and provide a proof. Limited generality permits the proposition label; it does not permit an unsupported claim.
- Use **Remark (备注)** only for interpretation, scope, limitations, intuition, or implementation guidance. A remark must not hide a new mathematical guarantee; relabel such a guarantee as a proposition or corollary and prove it.
- Minimize theorem count. Normally retain only one or two central theorems; use more only when every theorem is independently central, genuinely original, and rigorously proved. Merge closely related statements and downgrade routine or local claims according to this taxonomy.
- Before delivery, inventory every theorem-like environment, verify its label, originality, assumptions, proof or citation, and dependency on prior results, and remove duplicate or contradictory classification rules.

## Update protocol

When a later user message adds or changes a rule, append or revise the smallest relevant authoritative passage only if it generalizes beyond the current manuscript. For each rule family, retain one noncontradictory source of truth: merge overlaps and delete or rewrite conflicts instead of appending duplicates. Keep paper-specific conclusions in the active task notes or manuscript, not here.
