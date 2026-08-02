# Detail-preserving skill maintenance

Apply this contract whenever `skill-creator` or another maintenance pass is
used to reorganize, deduplicate, shorten, or validate this skill. Its purpose is
to improve retrieval without losing any user-supplied detail.

## Nondeletion rule

Do not delete a user rule, exception, example, project-specific constraint,
workflow preference, negative instruction, or explicitly unknown boundary
merely because it is detailed, appears specialized, lengthens the repository,
or is not needed by the current task. Detail is retained by classification and
selective loading, not by semantic compression.

`skill-creator` guidance may change:

- where a detail lives;
- whether SKILL.md contains the rule or only routes to it;
- whether exact semantic duplicates share one authoritative owner;
- how long files are split and indexed;
- which script or forward test protects the rule;
- which stage, domain, case, or asset branch loads the detail.

It must not erase the detail or weaken its conditions, exceptions, examples,
scope, priority, failure behavior, or provenance.

## Required refactoring procedure

Before modifying rule structure:

1. inventory each independent requirement, exception, example, and unknown
   boundary in the affected files;

2. assign each item a narrow authoritative destination using
   `rule-scope-map.md`;

3. distinguish exact semantic duplication from similar statements that carry
   different scope, strength, timing, examples, or exceptions;

4. move the complete semantic content before removing an old container;

5. preserve historical wording or superseded decisions in
   `source-rule-coverage.md` when it explains compatibility or provenance;

6. add or update routing sentences so the owning file is discoverable from
   SKILL.md or its directly selected stage/domain/case;

7. add validation fragments or forward tests when silent loss could recur;

8. run the structural validator, Skill Creator quick validation, relevant
   artifact audits, and a diff review before delivery.

## Deduplication rule

Deduplicate only exact operational truth. Keep the full rule in the narrowest
authoritative file and replace secondary copies with precise routes. Do not
collapse two rules merely because they share keywords. In particular, retain:

- general rules and narrower project-specific specializations;
- a positive requirement and its concrete negative examples;
- a workflow rule and its execution-evidence template;
- a current operational rule and the provenance needed to understand a later
  correction;
- a concise route and the complete detail it points to.

When two instructions conflict, do not silently choose one and delete the
other. Record the later user priority, preserve both original intentions, and
write the compatibility rule in `source-rule-coverage.md`.

## Evidence of preservation

For every structural refactor, report:

| Item | Required evidence |
|---|---|
| Files retired or split | destination mapping for every rule family |
| Exact duplicates removed | surviving authoritative location |
| Conflicts reconciled | source order and compatibility decision |
| Project-only details moved | matching domain or case destination |
| Examples reorganized | new case/index location and loading route |
| Validation changed | test or contract that detects future loss |

A refactor is incomplete when a retired file contains an independent detail
that has no mapped current owner, even if the skill still passes syntax or link
validation.
