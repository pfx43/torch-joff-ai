# Scientific-figure category cases

This library has exactly three category-level cases. A category—not an
individual published figure—is the case boundary:

1. `principle-diagrams/`;

2. `model-structure-diagrams/`;

3. `task-workflow-diagrams/`.

Read only the category matching the current Stage 4 figure. Local zoom-ins,
repeated-module stacks, and parallel branches are composition patterns governed
by `references/figure-composition-rules.md`; they are not a fourth case type.

## Category directory contract

Each of the three directories contains:

```text
<figure-category>/
  README.md       # category purpose, selection criteria, and synthesis rules
  references.md   # one inventory row per top-tier reference figure
  images/         # only authorized local source images/crops/thumbnails
```

Do not create one directory per reference figure. Add each selected figure as
one row in the category's `references.md`, assign a stable reference ID, and use
that ID in any authorized image filename or independent annotation.

## Rights and evidence boundary

Store a local source image only when it is user-provided, openly licensed,
publisher-authorized, or otherwise lawfully reusable. Otherwise store source
metadata, DOI/URL, figure number, rights status, and structural analysis without
the image.

Reference figures calibrate hierarchy, reading direction, frame nesting,
zoom-ins, stacks, icon restraint, and information density. They do not supply
technical truth or authorize copying a distinctive complete composition,
scientific claim, symbols, caption, experimental data, or color mapping.

## Selection standard

Prefer current primary papers from the target or a neighboring top-tier venue
when a figure clearly externalizes a scientific relation. Reject decorative,
poster-like, ambiguous, text-heavy, scientifically incomplete, or average
figures. Record why each selected reference is above the field average, which
pattern transfers, which content cannot transfer, and how the pattern will be
adapted independently.
