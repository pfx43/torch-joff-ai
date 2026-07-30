# Rule scope map

This file maps rule families to their authoritative locations. It is a maintenance index, not a second copy of the rules.

| Rule family | Authoritative location | Scope |
|---|---|---|
| Interaction language and answer structure | `living-user-rules.md` | Cross-manuscript |
| IEEE/Chinese academic voice, standard terminology, section organization, notation, and contribution focus | `living-user-rules.md` | Cross-manuscript |
| Complete web-conversation writing inventory, project-only preferences, non-manuscript formats, and explicit unknown boundaries | `user-writing-requirements-and-preferences.md` | Load matching section only |
| Recoverable first-version and web-source coverage, conflict reconciliation, and migration provenance | `source-rule-coverage.md` | Skill maintenance only |
| Prior-art comparison, novelty wording, and novelty maturity | `stages/idea-exploration.md` | Stage 1 |
| Repeated assumption, realizability, and model-specific contribution gates | `stages/idea-exploration.md` | Stage 1 |
| Journal manuscript construction and scientific figures | `stages/journal-paper-writing-and-figures.md` | Stage 2 |
| Mandatory chapter-conformance, sentence logic, causal narrative, notation, formula, model-completeness, train/test-flow, and abstract–problem–contribution loop | `manuscript-quality-gates.md` | Every Stage 2 manuscript |
| Frozen scientific-figure palette strips and color-role presets | `scientific-figure-palettes.md` | Stage 2 figures |
| Formal-claim taxonomy and theorem count | `living-user-rules.md` | Cross-manuscript |
| Information availability and hidden-variable audit | `technical-validity-and-implementation.md` | Every technical manuscript |
| Training-loop closure and differentiability | `technical-validity-and-implementation.md` | Learned models |
| Nonlinear stability scope and initial-error treatment | `technical-validity-and-implementation.md` | Every relevant theorem |
| Designable quantities and numerical realization | `technical-validity-and-implementation.md` | Every optimization or bound |
| Computational constraints and practical alternatives | `technical-validity-and-implementation.md` | Training and deployment |
| General row, column, block, and induced-norm contraction audit | `technical-validity-and-implementation.md` | Cross-manuscript |
| Evidence, metrics, no-fabrication, and technical review | `technical-validity-and-implementation.md` | Cross-manuscript |
| Markdown, figures, LaTeX, compilation, rendering, and PDF delivery | `stages/journal-paper-writing-and-figures.md` | Stage 2 artifacts |
| Official IEEE and 《控制理论与应用》 template selection, download, editing boundary, and integrity | `latex-template-workflow.md` | Stage 2 LaTeX projects for those journals |
| TFS model-to-proof and citation patterns | `tfs-reference-patterns.md` | Relevant IEEE TFS tasks |
| One-paper exploration history, alternatives, and unresolved routes | Matching file under `cases/<task-group>/` | One identified paper only |
| Unknown-fault information boundary | `domains/fault-diagnosis.md` | Fault diagnosis |
| Koopman latent/output relations | `domains/fault-diagnosis.md` | Lifted fault diagnosis |
| Attention, premise variables, and fuzzy memberships | `domains/fault-diagnosis.md` | Koopman/T–S/attention route |
| Measurement-decoupled normal references | `domains/fault-diagnosis.md` | Optional fault-diagnosis route |
| Joint latent/output residuals and detectability | `domains/fault-diagnosis.md` | Optional fault-diagnosis route |
| Dynamic thresholds and nominal/fault interfaces | `domains/fault-diagnosis.md` | Fault diagnosis |
| Post-filtering, robust fault gain, rank, and output dimension | `domains/fault-diagnosis.md` | Optional fault-diagnosis route |
| Structured lifted disturbance bases | `domains/fault-diagnosis.md` | Alternative fault-basis route |
| Soft-sensing latent-state interpretation | `domains/soft-sensing-and-observers.md` | Soft sensing |
| Nonlinear observer stability and time-varying coordinates | `domains/soft-sensing-and-observers.md` | Observer route |
| Observer-specific column contraction and long-memory networks | `domains/soft-sensing-and-observers.md` | Learned observer route |
| Active one-paper assumptions, evidence, and route maturity | Active `MANUSCRIPT_CONTEXT.md` | One manuscript only |
| Active one-paper symbol registry, naming basis, type, dimensions, typography, first definition, and scope | Active `NOTATION_LEDGER.md` created from `assets/templates/notation-ledger.md` | One manuscript only |
| Repeated concrete failure examples | `typical-errors.md` | Non-authoritative negative examples for pre-delivery audit |
| Case, context, idea-assessment, section-role, notation-ledger, and figure-plan document structures | Matching file under `assets/templates/` | Artifact scaffolding only |
| Skill version | `VERSION` | Repository release |
| Structural, link, formatting, and exact-text duplication checks | `scripts/validate_skill.py` | Skill maintenance |
| Writing-loop, manuscript, figure, and compilation checks | Matching file under `scripts/` | Stage 2 validation |
| Official template download, initialization, and integrity checks | `sources.json` under `assets/latex-templates/` and matching scripts | Registered LaTeX templates |
| Forward-test scenarios and expected behavior | `tests/forward-tests.json` | Skill regression evaluation |

When a rule seems to belong in two places, keep the complete rule in the narrowest authoritative document and use a short routing sentence elsewhere. Domain documents may specialize a general audit without restating it, and `typical-errors.md` may show a concrete failure signature without becoming a second authority. Do not maintain two independently editable full copies.
