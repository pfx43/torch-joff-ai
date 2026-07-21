# Living user rules

Update this file when the user gives a durable manuscript-writing or notation rule that should apply to later papers.

## Current rules

- Use formal English suitable for IEEE Transactions, with connected technical paragraphs and explicit causal transitions.
- Keep Section II as background/problem formulation; place paper-specific innovations in Section III or the designated method section.
- Use calligraphic capitals for nonlinear mappings and blackboard-bold notation for domains/spaces.
- Reserve `f`/`\mathfrak f` for faults; keep actuator and sensor fault channels distinct and combine them only in a clearly defined generalized fault vector.
- Reserve `H` for the neural/history-network side; do not reuse it for a physical output mapping.
- Denote the infinite-dimensional Hilbert space explicitly, e.g. `\mathbb H^{\infty}`.
- Avoid symbol collisions between Koopman operators/matrices and attention keys; use disjoint characters or fonts.
- Combine related multi-line equations into one numbered formula using `equation` plus `aligned`/`split`.
- Strengthen stability and sliding-mode proofs rather than relying on informal claims; state all assumptions and bounds.
- Distinguish a residual-dynamics-based theoretical dynamic threshold generator from rolling statistical, covariance-based, adaptive-quantile, or periodically recalibrated thresholds. Use the intended term and derive the corresponding guarantee explicitly.
- Do not fabricate results, data, or references. Preserve technical content unless the user explicitly authorizes deletion.
- Record only cross-manuscript writing and reasoning rules in this file; do not turn one paper's temporary notation, equation numbering, or technical design choice into a global rule.

## Update protocol

When a later user message adds or changes a rule, append or revise the smallest relevant bullet only if it generalizes beyond the current manuscript. Keep paper-specific conclusions in the active task notes or manuscript, not here.
