# Manuscript quality gates

This file defines the mandatory execution loop for Stage 2. It does not replace
the writing and notation rules in `living-user-rules.md`; it specifies when and
how those rules must be checked.

## Contents

- Gate 0: initialize the control artifacts
- Gate 1: establish section responsibilities
- Gate 2: register notation before drafting
- Gate 3: audit every drafted subsection
- Seven mandatory subsection audits
- Gate 4: align the abstract, problem, and contributions
- Gate 5: complete the manuscript-wide audit
- Mandatory repetition questions
- Failure and recovery behavior
- Definition of done

## Gate 0: initialize the control artifacts

Before substantive drafting or restructuring:

1. Read the complete manuscript source and the active
   `MANUSCRIPT_CONTEXT.md`.

2. Create or update `SECTION_ROLE_MATRIX.md` from
   `assets/templates/section-role-matrix.md`.

3. Create or update `NOTATION_LEDGER.md` from
   `assets/templates/notation-ledger.md`.

4. Mark every unresolved section role, symbol collision, missing dimension,
   and problem–contribution mismatch as `BLOCKED`.

Do not begin prose revision while either control artifact is absent or known to
be stale.

## Gate 1: establish section responsibilities

Assign one primary scientific question and one reader-facing output to every
section and subsection. The output may be a definition, model, analytical
result, design rule, algorithm, experiment, or limitation, but it must be
specific.

Before accepting the table of contents, compare it with the default compact
sequence in `living-user-rules.md`: Introduction; Preliminaries and Problem
Formulation; Proposed Method; an optional section only for a genuinely
independent second task; Experiments; Conclusion. Record the result under
`Chapter arrangement conformance` in `SECTION_ROLE_MATRIX.md`. A deviation
passes only when the target journal, a necessary scientific dependency, or an
independent second task requires it. Drafting history, module count, and a
desire for more headings do not justify a deviation.

Use the following loop before moving or drafting content:

1. Ask whether the section answers exactly one primary question.

2. Search the matrix for another section that answers the same question.

3. Merge, move, or delete duplicated responsibilities.

4. Verify that prerequisites appear before dependent results.

5. Record the accepted input and output of the section in the matrix.

End the preliminaries/problem section with `Problem Formulation` or `Problem
Description`. State one numbered item per central subproblem. Normally retain
two or three central subproblems. Do not use a long implementation checklist
and do not title this subsection `Monitoring Objectives`.

If more than three central subproblems are claimed, record in
`SECTION_ROLE_MATRIX.md` why each additional item is theoretically independent
and cannot be merged with another item. Without that explicit justification,
Gate 1 fails.

Map every central subproblem to exactly one principal contribution theme and
at least one result or construction in the body. A contribution may combine
closely coupled analytical and algorithmic consequences of the same
subproblem, but it must not collect unrelated modules.

Gate 1 passes only when the matrix has no duplicate responsibility and the
problem–contribution mapping is one-to-one in count and order.

## Gate 2: register notation before drafting

Treat notation as a registry, not as local author convenience. Before inserting
any new mathematical symbol:

1. Check the target journal, the research field, and the manuscript's established
   notation for a conventional symbol.

2. Select the symbol in this order: field or journal convention; standard
   mathematical convention; semantically meaningful English initial or
   mnemonic; another unclaimed symbol with an explicit project-specific
   rationale. A mnemonic must not override a stronger convention or create a
   misleading association.

3. Search the complete ledger for the exact glyph.

4. Search for the same base character across case, bold, calligraphic, script,
   fraktur, and blackboard-bold variants.

5. Check whether the intended meaning belongs to the already registered
   semantic family.

6. Check whether its object type uses the same typography as comparable
   objects elsewhere in the paper.

7. Record the symbol, semantic family, meaning, naming basis or convention,
   object type, dimension, typography, first-definition route, and scope before
   using it. The first-definition route must identify either the Introduction-end
   Notation paragraph or the exact first-use location.

8. If any meaning, object-type, semantic-family, or typography check fails,
   choose a new symbol or refactor the existing family before drafting the
   equation. Propagate the resolution through the source, ledger, equations,
   algorithms, appendices, captions, and figure labels.

Use one ledger row per mathematical object. Do not combine several symbols in
one row even when their dimensions are identical.

The exact font conventions and reserved families are governed by
`living-user-rules.md`. Gate 2 enforces those conventions but does not create a
second notation standard.

## Gate 3: audit every drafted subsection

After drafting or materially revising one subsection, stop and run this loop:

1. Re-read the subsection without using the intended outline as context.

2. State its single primary question in one sentence.

3. State its output in one sentence and compare it with the section-role
   matrix.

4. List every symbol introduced or assigned a new meaning in the subsection.

5. Confirm that each item already appears in the notation ledger with the same
   meaning, naming basis, type, dimension, typography, first-definition route,
   and scope.

6. Search the complete source for every newly introduced base character and
   inspect all visual variants.

7. Check equation dependencies, first definitions, dimensions, labels, and
   citations.

8. Remove repeated explanations, drafting commentary, module inventories, and
   claims that are not used by a later result.

9. Execute all seven mandatory subsection audits below and record their
   evidence and revision actions in `SECTION_ROLE_MATRIX.md`.

10. Update both control artifacts and mark the subsection `PASS` only after all
   conflicts are resolved.

Do not postpone symbol reconciliation until the full manuscript is complete.
Gate 3 repeats after every substantive subsection edit.

## Seven mandatory subsection audits

Gate 3 must check chapter conformance plus the six highest-priority writing
qualities. These are semantic audits performed by rereading the manuscript;
automated scripts verify that the checks were recorded but must not pretend to
judge causal or mathematical quality from keywords alone.

### 1. Chapter and subsection arrangement

- Compare the current table of contents and local subsection order with the
  skill's default sequence and the accepted section-role matrix.
- Confirm that each heading has one primary question and one reader-facing
  output, prerequisites precede dependent results, and no formula or module
  has been promoted to a heading without a scientific reason.
- Record the exact target-journal or scientific justification for every
  accepted deviation.
- Verify that Section II (Preliminaries and Problem Formulation) contains
  only standard background definitions and the mathematically stated problem.
  All paper-specific innovations, constructions, and methods must appear in
  Section III or the designated method section, not in preliminaries.
- Confirm that the exposition follows top-down order within each section:
  overall system architecture and objective before module structures, modules
  before parameters. Variables must not appear before their defining context.

### 2. Sentence-to-sentence logic and language variety

- Assign each sentence a role such as premise, limitation, motivation,
  definition, derivation, consequence, evidence, or transition.
- Confirm that every referent is unambiguous and every sentence follows from,
  qualifies, contrasts with, or prepares the next part of the argument.
- Remove unexplained topic jumps, unsupported conclusions, repeated claims,
  vague pronouns, and connective words that do not express the actual logical
  relation.
- Avoid stacking declarative statements without sentence variety. Use diverse
  sentence patterns: active and passive voice, subordinate clauses,
  participial phrases, prepositional phrases, adverbial modifiers, and
  appropriate adjectives and adverbs to create connected technical paragraphs
  rather than command-like prose.
- Verify that each paragraph serves one clear purpose: introduce or clarify a
  module, support the paper's argument and contributions, or provide logical
  transition. Remove paragraphs that pad word count without advancing
  understanding or supporting a claim.
- Verify that each paragraph's first sentence explicitly states the paragraph's
  single theme or function. The first sentence should answer "What is this 
  paragraph about?" without requiring readers to infer from later sentences.
  Common patterns for first sentences:
  * Definition: "[Term] is [definition]" or "[Term] consists of [components]"
  * Background: "In [domain], [established fact] is [state]"
  * Procedure: "[Method] requires [steps]" or "Training [model] involves [process]"
  * Transition: "In order to [goal], [construction] is [action]"
- Reject first sentences that are vague setup phrases like "There are several
  considerations" or "It is important to note that" without immediately naming
  the actual topic.

### 3. Narrative causality and exposition order

- Trace the chain from problem or limitation, through the need for the method,
  to the action of the proposed construction, the derived consequence, and the
  evidence that tests it.
- State why each module is needed, what quantity it changes, and how that
  change supports the next result.
- Do not substitute chronology, correlation, architecture order, or a module
  inventory for a causal explanation.
- Within each paragraph, follow reason-before-result order: state the
  motivation, problem, or limitation first, then present the construction or
  outcome as its consequence.
- Between paragraphs and subsections, follow macro-to-micro progression:
  present the overall system mapping and objective before introducing
  module-level structures, and introduce modules before their parameter-level
  details.
- Verify that variables are defined before use: do not introduce H, S, ρ, ω,
  or other parameters before explaining what overall system or mapping they
  serve.

### 4. Symbol consistency

- Confirm that every symbol is registered before use and retains one meaning,
  naming basis, object type, typography, dimension, first-definition route, and
  scope.
- Confirm that its name follows the priority of field or journal convention,
  mathematical convention, meaningful English initial or mnemonic, and only
  then an explicitly justified project-specific choice.
- Search the complete manuscript for the exact symbol and every visual variant
  of its base family.
- Reconcile equations, algorithms, captions, appendices, and figures rather
  than checking prose equations alone.

### 5. Formula rigor

- Confirm that definitions and assumptions precede use and that every index,
  dimension, operator, norm, evaluation point, initial condition, and data
  source is explicit.
- Verify dimensional compatibility, algebraic transitions, equality and
  inequality conditions, intermediate recursions, boundary cases, and the
  exact scope of the conclusion.
- Ensure that prose claims neither skip a decisive derivation nor enlarge the
  formula's local, conditional, one-step, or empirical result.

### 6. Model-description completeness

- Identify the model's inputs, outputs, physical and latent states, measured
  and unavailable quantities, disturbances or faults, fixed and learned
  mappings, parameters, assumptions, and initial conditions.
- State the update order, time indices, dimensions, interfaces between
  components, and how each required quantity is obtained.
- A diagram, architecture name, or list of neural modules does not replace the
  governing equations and information boundary.

### 7. Training, validation, testing, and deployment clarity

- Separate training, validation or model selection, final testing, and online
  deployment.
- State data provenance and splitting, permitted inputs and labels,
  preprocessing fitted on each split, objectives, constraints, optimization,
  stopping and checkpoint rules, initialization, test-only operations,
  baselines, metrics, repetitions, and reported uncertainty.
- Identify offline and online computations and check explicitly for label,
  future-time, test-set, fault, threshold, normalization, or preprocessing
  leakage.

Record one row for every audited subsection in the `Subsection writing-loop
record`. A `PASS` requires concrete evidence inspected and any revision action;
seven unsubstantiated status words are not a completed audit.

## Gate 4: align the abstract, problem, and contributions

Perform a structure-level version of this gate before substantive body
drafting. Repeat the content-level version after the main technical body is
stable and again after any change to the paper's central route. The early pass
checks count, order, and mapping; the later pass also checks the exact technical
claims and wording.

1. Count the central subproblems in `Problem Formulation` or `Problem
   Description`.

2. Count the principal contribution items in the Introduction.

3. Verify equal counts, identical order, and one-to-one mapping.

4. For each pair, identify the corresponding body section and explicit output.

5. Merge or delete any contribution that only names a model component,
   training module, standard tool, or implementation detail.

6. Rewrite the abstract in the same order: problem, method, central analytical
   result, diagnostic/design result, and validation or scope.

7. Remove background survey text, equation-level detail, adjective-heavy
   novelty language, and repeated module names from the abstract.

Unless the target journal imposes a different limit, use 150–220 English words
as the working range. Keep two or three principal contribution items. Each item
should state one problem-facing construction and its nontrivial result in no
more than two sentences.

Gate 4 fails if the abstract, problem statement, contribution list, and body
present different task counts, different ordering, or different technical
claims.

## Gate 5: complete the manuscript-wide audit

Before compilation and delivery:

1. Run
   `python -X utf8 scripts/audit_writing_loops.py <project-root>`.

2. Run
   `python -X utf8 scripts/audit_manuscript.py <project-root>`.

3. Compile and perform the LaTeX, PDF, and visual checks required by the Stage
   2 guide.

4. Re-read the complete notation ledger against the compiled paper, including
   captions, algorithms, appendices, and figure labels. Confirm that every
   displayed and inline mathematical symbol is explained either in the
   Introduction-end Notation paragraph or at its exact first use.

5. Re-read the section-role matrix against the final table of contents.

6. Verify that every final subsection has one completed seven-audit record
   after its last substantive change.

7. Repeat Gate 4 on the final abstract and Introduction.

Warnings require an explicit disposition. Errors block delivery.

For every repeated gate, record the check time, evidence inspected, conflict,
revision action, affected sections, and resulting status in the control
artifacts. A bare `PASS` without this trace is not a completed gate.

## Mandatory repetition questions

Ask these questions internally at Gate 1, after every subsection at Gate 3, and
before delivery at Gate 5:

- Does this section answer exactly one primary scientific question?
- Is the same question answered elsewhere?
- Does the chapter and subsection order follow the skill, or is every deviation
  explicitly justified?
- Is each problem item one central subproblem rather than a checklist entry?
- Does every problem item map to one contribution theme and one body result?
- Does every sentence have a clear role, referent, and logical dependency?
- Is the problem–method–mechanism–result–evidence causal chain explicit?
- Has every symbol been registered before use?
- Does every symbol have an appropriate field, mathematical, mnemonic, or
  explicitly justified project-specific naming basis?
- Is every symbol explained either in the Introduction-end Notation paragraph
  or at its exact first use, and is that route recorded?
- Is the same base character already registered with another meaning?
- Does the exact symbol conflict with an existing object's type?
- Does the typography match the mathematical object type and the rest of its
  semantic family?
- Are dimensions and scopes explicit?
- Are the formulas dimensionally compatible, fully conditioned, and derived
  far enough to support the prose conclusion?
- Are the model inputs, outputs, states, known and unknown quantities,
  mappings, parameters, update order, and interfaces complete?
- Are training, validation, testing, and online deployment distinct,
  reproducible, and free of information leakage?
- Can the abstract or a contribution item be shortened without losing a
  technical result?
- Are there more than three principal contribution themes that should be
  merged?

These are self-audit questions. Ask the user only when resolving a conflict
requires a substantive scientific choice that cannot be inferred safely from
the manuscript.

## Failure and recovery behavior

Use the following state transition:

`DRAFT -> CHECK -> PASS -> NEXT`

If a gate fails, use:

`DRAFT -> CHECK -> FAIL -> REVISE -> CHECK`

Do not advance to the next subsection, compile a delivery PDF, or describe the
manuscript as submission-ready while a mandatory gate remains in `FAIL` or
`BLOCKED`.

When late editing changes the chapter order, a paragraph's causal chain, a
symbol family, a formula, a model interface, the training or testing flow, a
section responsibility, a central subproblem, or a contribution claim,
invalidate every dependent later gate and rerun it. A successful earlier audit
is not permanent after dependent content changes.

## Definition of done

Stage 2 writing is complete only when:

- the section-role matrix matches the final manuscript;
- the notation ledger covers every paper-specific symbol without unresolved
  family collisions;
- every subsection has passed Gate 3 after its last substantive edit;
- every subsection has a completed record for chapter arrangement,
  sentence-to-sentence logic, narrative causality, symbol consistency, formula
  rigor, model completeness, and training/testing clarity;
- the abstract, problem description, contributions, and body pass Gate 4;
- all automated and visual checks have passed or have an explicitly reported
  nonblocking limitation;
- the delivered status does not overstate the manuscript's readiness.
