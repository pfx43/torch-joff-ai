# Section Role Matrix

## Manuscript

- Working title: Writing-loop fixture
- Target journal: Generic journal fixture
- Main source: `main.tex`
- Last synchronized: 2026-07-29

## Chapter arrangement conformance

- Skill default sequence: Introduction; Preliminaries and Problem Formulation;
  Proposed Method; optional genuinely independent second task; Experiments;
  Conclusion.
- Actual top-level sequence: Introduction; Preliminaries and Problem
  Formulation; Proposed Method; Experiments; Conclusion.
- Target-journal or scientific justification for each deviation: None; the
  fixture follows the default sequence.
- Status: PASS

## Problem–contribution alignment

| Problem ID | One central subproblem | Contribution ID | Problem-facing construction | Explicit result or output | Body location | Validation or evidence | Status |
|---|---|---|---|---|---|---|---|
| P1 | Maintain scientific chapter order | C1 | Chapter-control procedure | Aligned section roles | Sections I–V | Final table of contents | PASS |
| P2 | Audit every revised subsection | C2 | Seven-part subsection loop | Recorded audit evidence | Proposed Method | Loop record | PASS |

## Section responsibilities

| Section / subsection | Single primary question | Required input | Reader-facing output | Problem ID | Contribution ID | Depends on | Narrative dependency / causal transition | Skill-sequence conformance or justified deviation | Status |
|---|---|---|---|---|---|---|---|---|---|
| Abstract | What was checked and obtained? | Final body | Problem–method–result summary | P1–P2 | C1–C2 | All body sections | Summarizes the final causal chain | Conforms as front matter | PASS |
| Introduction | Why is the loop needed? | Writing-quality limitation | Gap and contributions | P1–P2 | C1–C2 | Abstract revised last | Limitation motivates the control procedure | Conforms | PASS |
| Preliminaries and Problem Formulation | Which problems are solved? | Information boundary | Two numbered subproblems | P1–P2 | C1–C2 | Introduction gap | Converts the gap into auditable tasks | Conforms | PASS |
| Proposed Method | How are the problems solved? | P1–P2 and notation | Model and seven-part loop | P1–P2 | C1–C2 | Problem formulation | The control artifacts resolve the tasks | Conforms | PASS |
| Experiments | What evidence is admissible? | Selected method | Split-role audit | P2 | C2 | Proposed Method | Tests the workflow without invented results | Conforms | PASS |
| Conclusion | What is established? | Final body | Scoped conclusion | P1–P2 | C1–C2 | All prior sections | Closes the established result | Conforms | PASS |

## Subsection writing-loop record

| Subsection | Chapter arrangement | Sentence-to-sentence logic | Narrative causality | Symbol consistency | Formula rigor | Model completeness | Training / validation / testing / deployment clarity | Evidence inspected and revision action | Last checked | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Problem Formulation | Located at the end of Section II | Each item states one task | Introduction gap leads to P1–P2 | No new symbols | Numbered tasks checked | Information boundary precedes method | Not applicable before method definition | Compared source headings and contribution list; retained two items | 2026-07-29 | PASS |
| Proposed Method | Located after Section II | Definition precedes equation and workflow | Audit need leads to model and split controls | $x$ and $y$ match ledger | Eq. 1 is scalar and dimensionally valid | Input state, output, map, and update relation stated | Training, validation, and test roles separated | Checked source, ledger, equation, and split sentences; clarified each split role | 2026-07-29 | PASS |

## Gate record

| Gate | Last run | Evidence inspected | Conflict | Revision action | Affected sections | Result |
|---|---|---|---|---|---|---|
| Gate 1: section responsibilities | 2026-07-29 | Table of contents and role table | None | Retained default order | All | PASS |
| Gate 3: seven mandatory subsection audits | 2026-07-29 | Subsection records and source | None | Recorded evidence | Sections II–III | PASS |
| Gate 4: abstract–problem–contribution alignment | 2026-07-29 | Abstract, P1–P2, C1–C2 | None | Preserved order | Abstract and Sections I–III | PASS |
| Gate 5: manuscript-wide audit | 2026-07-29 | Source, ledger, and role matrix | None | Final synchronization | All | PASS |
