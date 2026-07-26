---
name: ieee-english-paper-polish
description: Revise, extend, structure, and quality-check English academic manuscripts in IEEE Transactions style, especially IEEE TFS papers involving nonlinear systems, T-S fuzzy models, Koopman operators, observers, fault diagnosis, neural networks, stability proofs, and LaTeX. Use when the user asks for manuscript rewriting, notation unification, theorem/proof strengthening, IEEE chapter organization, reference-aware polishing, LaTeX generation, PDF compilation, or incorporation of durable writing rules from later feedback.
---

# IEEE English Paper Polish

Use this skill as a manuscript-revision workflow, not as surface-level copyediting. Preserve the author's technical claims and evidence, make the mathematical narrative auditable, and keep all notation, section boundaries, references, and compiled artifacts consistent.

## Workflow

1. Read the manuscript source, bibliography, project instructions, and any named reference papers or extracted literature summaries before editing. At the project root, look for `MANUSCRIPT_CONTEXT.md`; read it in full when present. If it is absent and the paper's scope is sufficiently established, create it before substantive revision with the paper idea, modeling and data boundaries, section outline, notation ledger, confirmed decisions, and unresolved issues. Treat explicit user corrections as higher priority than defaults in this skill.

2. Build or update the project notation ledger before rewriting. Record each symbol's meaning, font, dimensions, and first definition. Search the entire source for collisions before introducing a new symbol.

3. Separate background from contributions. Put generic system definitions, standard T-S/Koopman facts, assumptions, and the abstract problem statement in preliminaries. Put causal lifting, attention membership generation, learned rule construction, and other paper-specific ideas in the method section.

4. Rewrite in formal English with connected paragraphs. Use a compact IEEE Transactions hierarchy: Introduction; Preliminaries and Problem Formulation; Proposed Method; an optional separate section only for a genuinely second task; Experiments; Conclusion. End Section II with a subsection titled `Problem Formulation` that states, primarily through formulas, the two central tasks or guarantees pursued by the paper; use at most three objectives. Do not replace this subsection with a long prose checklist such as `Monitoring Objectives`. Do not promote every module or proof stage to a main section. Within the proposed-method section, normally order the material as theoretical modeling and analysis, data-driven forward modeling and loss design, and the end-to-end task workflow. Move long proofs to appendices; backpropagation-gradient formulas are optional unless the optimization derivation itself is a contribution.

5. Close every important mathematical argument: motivation sentence, numbered equation, immediate symbol/dimension explanation, and consequence. Classify formal claims using the single authoritative taxonomy in `references/living-user-rules.md`, and audit both their proofs and their total count before delivery.

6. Compile with `latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error`, inspect undefined references and overfull boxes, then render representative/all PDF pages for visual checking.

7. When the user supplies a new durable rule, update `references/living-user-rules.md` before finishing. Record only rules that transfer across manuscripts; do not record one-paper symbol choices, equation numbers, temporary hypotheses, or task-specific technical decisions. Put those decisions in the active project's `MANUSCRIPT_CONTEXT.md` instead, and update only the affected entries after an explicit decision changes. Keep each retained rule concise, generalized, and actionable. Treat `D:\AI\cc-switch\skills\ieee-english-paper-polish` as the sole canonical copy. Installed locations must reference the cc-switch-managed copy through a junction or symbolic link; never create or synchronize an independent duplicate.

8. Read `references/typical-errors.md` during each manuscript revision and use it as a pre-delivery negative checklist. When the user supplies a concrete recurring error, generalize it there as an incorrect pattern, explain why it fails, and record an acceptable replacement. Do not preserve paper-specific equation numbers in the reusable catalog.

## Markdown artifact rules

Apply these rules to every paper-related Markdown file created or edited under this skill, including project contexts, outlines, notation ledgers, literature notes, experiment plans, review notes, and response drafts.

- Use `$...$` for inline mathematics and standalone `$$...$$` blocks for display mathematics. Do not use `\(...\)` or `\[...\]` in Markdown.
- Leave a blank line before every opening display-math delimiter and after every closing display-math delimiter so the formula remains a separate block.
- Leave a blank line before every ordered-list item, including consecutive numbered items.
- Use a Unicode en dash (`–`) in Markdown prose instead of the LaTeX-style double hyphen (`--`). Preserve Markdown table separator rows such as `|---|---|`.
- Before delivery, audit every affected Markdown file for legacy math delimiters, exposed renderer placeholders such as `@@TOLARIA_MATH...`, unmatched math delimiters, prose double hyphens, missing required blank lines, control characters, and Unicode replacement characters.

## Notation rules

- Do not place displayed equations before Section II. The title, abstract, keywords, and Introduction may use indispensable inline symbols, but mathematical models and derivations begin in the preliminaries.
- Define every symbol at its first occurrence. When many related symbols would interrupt the narrative, define their shared convention in a Notation paragraph at the end of the Introduction and give object-specific meanings when the objects first appear.
- Assign one semantic family to each base character and one visual format to each object type. As defaults, use `f` for faults/faulty quantities, `n` for normal/nominal quantities, `k` for discrete-time sample indices, and `u` for control inputs. Use boldface consistently for vectors and matrices, calligraphic capitals for mappings, blackboard bold for sets/spaces, and a single dedicated font family for operators or functionals.
- Minimize visually similar variants such as `\phi`, `\varphi`, `\psi`, and `\Psi`. Select one base-symbol family for a concept such as observables/liftings, and use subscripts, boldface, or dimensions to distinguish its scalar and vector forms.
- Use a consistent, visually distinct fault font throughout a manuscript; reserve the same base letter for physical faults, generalized fault vectors, estimates, and variations.
- Use calligraphic capitals for nonlinear mappings and blackboard-bold notation for sets and spaces. Never use one symbol as both a mapping and its domain.
- Reserve blackboard-bold fonts for mathematical number fields, domains, sets, and spaces. Do not use blackboard-bold letters for information stacks, data vectors, model states, or other ordinary variables.
- Reserve symbols by semantic family when the paper contains neural networks, physical mappings, Koopman operators, attention keys, residuals, and stability constants. Record paper-specific choices in `MANUSCRIPT_CONTEXT.md` rather than hard-coding them in this skill.
- Avoid reusing one symbol for different objects even when their fonts or subscripts differ only slightly. Audit the whole manuscript for collisions before accepting new notation.
- Keep Koopman operator/matrix notation disjoint from attention key/prototype notation. Distinguish infinite-dimensional operators from finite-dimensional approximations by font or symbol family.
- Avoid programming-style aggregation operators such as `\operatorname{col}` for vectors and matrices. Write explicit bracketed row or column arrays with a transpose when needed, so the mathematical object and its orientation are visible.
- Give related multi-line relations one equation number. Prefer `equation` with `aligned`/`split` over an `align` environment that silently creates multiple numbers. Keep labels outside the inner alignment when possible.

## Theory and learning rules

- For a controlled Koopman preliminaries section, define the nonlinear transition mapping, lifting mapping, and controlled operator action with every control or exogenous argument shown consistently on both sides. Then give a genuinely distinct finite-dimensional controlled linear approximation, such as a lifted state equation with separate control/exogenous input matrices and an output equation, supported by appropriate citations. Do not relabel the operator identity as the finite-dimensional approximation, and do not introduce the paper's causal history lift or attention architecture there.
- For T-S models, state nonnegativity and partition-of-unity membership conditions explicitly. Explain whether memberships are fixed, measured, data-driven, or learned.
- For stability, distinguish local-rule stability from stability of arbitrary time-varying convex interpolations. A common induced-norm contraction is a valid route; state the norm, weighting, contraction constants, and all bounded inputs used in the proof.
- If a neural parameterization certifies a spectral-norm bound, prove differentiability, show the strict bound, and derive the differential/backpropagation gradient. Do not claim exact spectral normalization when the proof uses the Frobenius upper bound.
- For discrete sliding-mode analysis, give both the lower disturbance-dominating gain bound and the upper no-repeated-crossing bound. Prove finite-step reachability and positive invariance of the boundary layer.
- For simultaneous actuator/sensor diagnosis, keep the physical channels distinct until their lifted one-step signatures are combined. State single-step and multi-step rank/conditioning assumptions and reconstruction-error bounds.
- Use the term *dynamic threshold* precisely. A Ding-style dynamic threshold is an online threshold generator derived from the fault-free residual dynamics; it propagates and separates the effects of admissible initial error, known/measured inputs and model uncertainty, and disturbances/noise. Do not substitute a rolling empirical quantile, moving covariance, or periodically refitted constant threshold and call it the same method. For multi-step residuals, preserve cross-time propagation in the stacked bound. If a confidence level is claimed, identify which noise term is probabilistic and keep deterministic input/uncertainty bounds separate.
- Do not invent experiments, numerical results, hardware, datasets, or citations. If evidence is missing, retain a clearly marked protocol or placeholder.

## IEEE/TFS writing patterns

- Organize the Introduction in a stable sequence: one background paragraph that states the topic and monitored problem; one paragraph classifying relevant modeling approaches; problem-focused literature and limitation paragraphs; the core unresolved question; concrete contributions; and a closing paragraph for section organization and notation conventions. Do not include displayed equations in this sequence.
- Drive the Introduction by technical tension: nonlinear operating regimes and uncertainty -> limits of linear/model-free diagnosis -> value of T-S/IT2 or structured fuzzy representations -> specific unresolved issue -> precise contributions.
- Write contributions as three to five concrete claims, each tied to a construction, analysis, or appropriately classified formal result. Do not place a paper-specific innovation in the preliminaries section.
- Use established control, systems, and signal-processing terminology. Avoid literal or ad hoc phrases such as `healthy transition`; select the context-appropriate standard term, for example `nominal state-transition mapping`, `fault-free state equation`, or `normal-operation trajectory`.
- Remove drafting commentary, author-process notes, and self-explanatory previews from the manuscript body. Sentences such as “the paper-specific construction is introduced later,” “this object is not part of the preliminaries,” or “importantly, the learned state need not ...” should be replaced by the required definition or omitted. Keep only conventional section roadmaps and technically necessary cross-references.
- Use the recurring TFS proof chain: model/observer definition -> augmented error dynamics -> Lyapunov or induced-norm condition -> appropriately classified formal result -> residual, threshold, isolation, or reconstruction logic.
- For experiments, specify scenario, data split, fault types (including simultaneous actuator and sensor faults when claimed), baselines, metrics, and reproducibility information before discussing results. Never turn a planned experiment into a reported result.
- Prefer paragraphs over code-like notation or unexplained abbreviations. Define abbreviations at first use and keep terminology stable.

## Reference use

Read `references/tfs-reference-patterns.md` when the task concerns IEEE TFS fault diagnosis, T-S/IT2 fuzzy observer design, residual thresholds, event-triggered diagnosis, knowledge-distilled fuzzy diagnosis, or reference-aware chapter organization. Use the reference library to learn structure and proof patterns, not to copy text or fabricate bibliographic entries.

Read `references/living-user-rules.md` at the start of each manuscript task. Update it only when the user makes a durable, cross-manuscript notation, organization, proof, formatting, or evidence rule explicit; keep manuscript-specific conclusions in the current task artifacts instead.

Read `references/typical-errors.md` at the start of each manuscript task. Treat its examples as prohibited patterns to search for, not as text templates to copy into a paper.

Read the active project's `MANUSCRIPT_CONTEXT.md` before modifying its manuscript, bibliography, equations, proofs, or section organization. Update that file when—and only when—an explicit project-level decision changes the paper's scope, model, training data, section plan, notation, proof route, residual, or threshold design.

## Delivery checklist

- Return links to the modified `.tex`, compiled `.pdf`, and any bibliography file when requested.
- Report the substantive notation/structure/theory changes briefly, without dumping the whole diff.
- Confirm that `MANUSCRIPT_CONTEXT.md` remains consistent with the delivered manuscript whenever project-level decisions were changed.
- Before delivery, search for forbidden collisions (`\\mathfrak f` vs. `f`, mapping letters, reused `H`, reused `K`, duplicate labels), compile twice if needed, and visually inspect the rendered PDF.
