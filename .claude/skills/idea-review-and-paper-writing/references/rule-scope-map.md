# Rule scope map

This file maps rule families to their authoritative locations. It is a maintenance index, not a second copy of the rules.

| Rule family | Authoritative location | Scope |
|---|---|---|
| Interaction language and answer structure | `living-user-rules.md` | Cross-manuscript |
| IEEE/Chinese academic voice, standard terminology, section organization, notation, and contribution focus | `living-user-rules.md` | Cross-manuscript |
| Complete web-conversation writing inventory, project-only preferences, non-manuscript formats, and explicit unknown boundaries | `user-writing-requirements-and-preferences.md` | Load matching section only |
| Recoverable first-version and web-source coverage, conflict reconciliation, and migration provenance | `source-rule-coverage.md` | Skill maintenance only |
| No-detail-loss refactoring, exact-semantic deduplication, and preservation evidence | `detail-preservation-and-refactoring.md` | Every skill maintenance pass |
| Prior-art comparison, novelty wording, and novelty maturity | `stages/idea-exploration.md` | Stage 1 |
| Repeated assumption, realizability, and model-specific contribution gates | `stages/idea-exploration.md` | Stage 1 |
| Paper background/task, contribution map, technical story, model-definition order, notation/terminology baseline, chapter blueprint, narrative progression, and context freeze | `stages/paper-conception.md`; active revision folder's `MANUSCRIPT_CONTEXT.md` | Stage 2 only |
| Non-experimental manuscript drafting order, construction, LaTeX schematics, compilation, and PDF QA | `stages/manuscript-writing.md` | Stage 3 only |
| Mandatory baseline consistency, chapter precedence, narrative logic, non-rigid/nonempty prose, symbol consistency, specialized terminology, formula rigor, schematic completeness, and manuscript alignment | `manuscript-writing-loop.md` | Every Stage 3 manuscript |
| Concrete published-paper architecture, narrative, sentence, transition, and terminology exemplars | One-paper published-style case under `cases/<task-group>/`; cross-case routing in `style-exemplars/README.md` | Stage 2 calibration and Stage 3 prose only; never technical, novelty, or citation authority |
| Scientific-figure and experimental-result integration loops | `stages/figures-and-experiments.md` | Stage 4 only |
| Model/principle/workflow composition, arrows, frames, icons, editability, copyright, and verification | `figure-composition-rules.md` | Stage 4 conceptual figures |
| Multi-reference precedent inventory and structural synthesis for exactly three figure classes | `cases/figure-exemplars/<figure-category>/` | Stage 4 matching principle/model/workflow category only |
| Frozen scientific-figure palette strips and color-role presets | `scientific-figure-palettes.md`; `assets/palettes/` | Stage 4 only |
| Reusable icon source, license, asset naming, and registry | `assets/icons/README.md`; `assets/icons/icon-registry.md` | Stage 4 icon acquisition/use only |
| Short Baseline ID, revision-folder identity, and internal role filenames | `artifact-naming.md`; `scripts/audit_artifact_names.py` | Stages 2–4 and delivery |
| Formal-claim taxonomy and theorem count | `living-user-rules.md` | Cross-manuscript |
| Information availability and hidden-variable audit | `technical-validity-and-implementation.md` | Every technical manuscript |
| Training-loop closure and differentiability | `technical-validity-and-implementation.md` | Learned models |
| Nonlinear stability scope and initial-error treatment | `technical-validity-and-implementation.md` | Every relevant theorem |
| Designable quantities and numerical realization | `technical-validity-and-implementation.md` | Every optimization or bound |
| Computational constraints and practical alternatives | `technical-validity-and-implementation.md` | Training and deployment |
| General row, column, block, and induced-norm contraction audit | `technical-validity-and-implementation.md` | Cross-manuscript |
| Evidence, metrics, no-fabrication, and technical review | `technical-validity-and-implementation.md` | Cross-manuscript |
| Markdown, LaTeX, compilation, rendering, and non-experimental PDF QA | `stages/manuscript-writing.md` | Stage 3 artifacts |
| Result-bearing final PDF delivery and figure/caption/text closure | `stages/figures-and-experiments.md` | Stage 4 artifacts |
| Official IEEE and 《控制理论与应用》 template selection, download, editing boundary, and integrity | `latex-template-workflow.md` | Stage 3 LaTeX projects for those journals |
| TFS model-to-proof and citation patterns | `tfs-reference-patterns.md` | Relevant IEEE TFS tasks |
| One-paper exploration history, alternatives, and unresolved routes | Active manuscript case under `cases/<task-group>/` | One identified paper only |
| One-paper published style observations and micro-examples | Published-style exemplar case under `cases/<task-group>/` | Matching task calibration only |
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
| Active one-paper conception baseline, assumptions, evidence needs, problem/contribution mapping, model order, chapter/narrative plan, notation, and terminology | Frozen `<revision-folder>/MANUSCRIPT_CONTEXT.md` created from `assets/templates/manuscript-context.md` | One manuscript revision only |
| Stage 3 paragraph/subsection/schematic/alignment evidence | `<revision-folder>/WRITING_LOOP_LOG.md` created from `assets/templates/writing-loop-log.md` | One manuscript revision only |
| Stage 4 figure, plot, result-narration, and final-delivery evidence | `<revision-folder>/STAGE4_FIGURE_EXPERIMENT_LOG.md` created from `assets/templates/stage4-figure-experiment-log.md` | One manuscript revision only |
| Repeated concrete failure examples | `typical-errors.md` | Non-authoritative negative examples for pre-delivery audit |
| Case, context, idea-assessment, Stage 3/4 logs, figure-plan, and figure-category-case structures | Matching file under `assets/templates/` | Artifact scaffolding only |
| Skill version | `VERSION` | Repository release |
| Structural, link, formatting, and exact-text duplication checks | `scripts/validate_skill.py` | Skill maintenance |
| Writing-loop, manuscript, figure, and compilation checks | Matching file under `scripts/` | Stage 3 or Stage 4 validation |
| Official template download, initialization, and integrity checks | `sources.json` under `assets/latex-templates/` and matching scripts | Registered LaTeX templates |
| Forward-test scenarios and expected behavior | `tests/forward-tests.json` | Skill regression evaluation |

## Folder boundary decision

Use this decision before adding or moving a file:

| Question | Destination | Must not contain |
|---|---|---|
| Is it a reusable rule or workflow instruction? | `references/` | Active-paper decisions, downloaded assets, copied manuscript prose |
| Is it the history or style evidence of one paper? | `cases/<task-group>/` | Cross-manuscript mandatory rules |
| Is it precedent for one of the three scientific-figure classes? | `cases/figure-exemplars/<principle-diagrams|model-structure-diagrams|task-workflow-diagrams>/` | A directory per reference figure, technical authority, unlicensed images |
| Is it a reusable template, palette, icon, or other output resource? | `assets/` | Workflow policy duplicated from `references/`, unregistered downloads |
| Is it an active manuscript artifact? | The `<short-baseline-id>-r<revision>/` folder, using short internal role filenames | Reusable skill rules or shared reference cases |
| Is it deterministic maintenance/QA logic? | `scripts/` | Paper-specific reasoning or editable truth sources |
| Is it a regression scenario or synthetic fixture? | `tests/` | Live manuscript evidence or user-owned research data |

## Stage ownership test

- Stage 1 may change the idea assessment and paper case, not the conception,
  manuscript, final figure, or experiment-result artifacts.
- Stage 2 may change only the active versioned context and its paper case
  history. It must not write final prose or final figures.
- Stage 3 may change the versioned TeX/bibliography, writing log, and LaTeX
  schematics. A conception change returns to Stage 2.
- Stage 4 may change figure plans/sources/exports, icon use records, result
  prose, Stage 4 log, and final versioned PDF/archive. A method/topology change
  returns upstream.

If two files can independently change the same paper decision, symbol meaning,
chapter responsibility, figure topology, license fact, or experimental result,
the boundary is invalid. Keep the full fact in the narrowest owner and link to
it elsewhere.

When a rule seems to belong in two places, keep the complete rule in the narrowest authoritative document and use a short routing sentence elsewhere. Domain documents may specialize a general audit without restating it, and `typical-errors.md` may show a concrete failure signature without becoming a second authority. Do not maintain two independently editable full copies.
