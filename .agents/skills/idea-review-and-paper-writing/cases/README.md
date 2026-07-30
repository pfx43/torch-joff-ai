# Manuscript case library

This directory stores paper-specific research discussions. Each manuscript or clearly distinct paper task receives one Markdown file under the matching task-domain folder.

Cases preserve exploratory reasoning, rejected alternatives, local terminology, model boundaries, unresolved questions, and paper-specific decisions without turning them into global skill rules.

Cases are not workflow checklists. The universal idea-exploration rules belong to `references/stages/idea-exploration.md`; the universal journal-writing and figure rules belong to `references/stages/journal-paper-writing-and-figures.md`.

## When to read a case

Read a case file only when:

- the active manuscript is the same paper or project;
- the user explicitly names or links the case;
- the task is to review the history of that specific paper;
- the active `MANUSCRIPT_CONTEXT.md` identifies the case as its archive.

Cases in the same task group may be consulted as examples to surface candidate routes, recurring failure modes, terminology choices, or experiment ideas. Treat every borrowed item as a hypothesis for the current paper: independently recheck prior art, assumptions, information availability, derivation, and evidence before adopting it.

Do not apply a case to another paper merely because both papers use Koopman operators, fuzzy systems, attention, observers, or the same dataset. A related case is neither a novelty search nor proof that a route transfers.

## Authority and synchronization

For an active manuscript:

- the user's current instruction has highest priority;
- the active project's `MANUSCRIPT_CONTEXT.md` is the operational source of truth;
- the matching case file preserves the longer research history;
- domain guidance supplies reusable options and cautions;
- living user rules supply cross-manuscript preferences.

If the case and active project context disagree, do not silently choose one. Use the newer explicit decision, update the active context, and then update the case history so the disagreement is visible and resolved.

## Directory structure

Current task groups:

- [`fault-diagnosis/`](fault-diagnosis/): individual fault-diagnosis manuscripts;
- [`soft-sensing/`](soft-sensing/): individual soft-sensing and quality-prediction manuscripts.

Create additional groups such as `model-predictive-control/` or `data-completion/` when the first concrete manuscript case exists. Do not create speculative case content in advance.

## File naming

Use one stable, descriptive, lowercase filename per paper. Prefer:

```text
primary-method-central-task.md
```

Do not name files `paper1.md`, `new-idea.md`, or by temporary equation numbers.

## Creating a case

Copy [`../assets/templates/paper-case.md`](../assets/templates/paper-case.md) and record:

- paper identity and task group;
- target journal and positioning;
- information and data boundaries;
- confirmed, exploratory, alternative, rejected, and unresolved routes;
- important terminology and notation decisions;
- theoretical claims and implementation questions;
- experiment status;
- links to the active project and domain guidance;
- reusable insights that may later be promoted.

Apply the two stage documents while discussing the case, but record only this paper's actual findings and decisions in the case file.

One file represents one paper. If a discussion becomes a genuinely different paper with a different central problem or contribution structure, create another case instead of mixing both papers.

## Promoting a case insight

Promote an insight only after deciding its scope:

- one-paper decision: keep it in the case and `MANUSCRIPT_CONTEXT.md`;
- reusable domain rule: generalize it under `references/domains/`;
- cross-manuscript rule: generalize it in `references/living-user-rules.md` or `references/technical-validity-and-implementation.md`;
- recurring error: generalize it in `references/typical-errors.md`.

When promoting a rule, keep the detailed history in the case but link to the new authoritative rule. Do not maintain two independently editable full copies.
