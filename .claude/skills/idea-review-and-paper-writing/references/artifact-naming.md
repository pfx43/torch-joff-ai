# Revision-folder and artifact naming

Use this contract whenever Stage 2 freezes a conception, Stage 3 creates the
manuscript, or Stage 4 creates figures, result text, and delivery artifacts.
The version identity belongs to the revision folder; files inside that folder
use short, stable role names.

## Revision-folder identity

Define

```text
revision folder = <baseline-id>-r<context-revision>
```

Choose a short Baseline ID: normally one to three recognizable lowercase terms
or accepted abbreviations, preferably no more than 16 characters and never
more than 24 characters before `-rN`. It must
match `[a-z0-9]+(?:-[a-z0-9]+)*`, remain meaningful to the paper team, and stay
unchanged for the same paper lineage. Prefer `koop-fd`, `qss-observer`, or
`data-impute` over a title-length phrase.

Example:

```text
koop-fd-r3/
```

The revision is the positive integer recorded in the frozen context. Do not
zero-pad it or maintain a second file revision. During Stage 2 drafting, use
`<baseline-id>-r0/`; the first frozen snapshot becomes `r1`. Refreezing creates
a new sibling folder and preserves the previous frozen folder.

## Files inside one revision folder

```text
<baseline-id>-r<revision>/
  MANUSCRIPT_CONTEXT.md
  manuscript.tex
  references.bib
  WRITING_LOOP_LOG.md
  manuscript.pdf
  STAGE4_FIGURE_EXPERIMENT_LOG.md
  submission.zip
  rendered/
  figures/
  plots/
```

These role names are intentionally simple because the parent folder already
binds every artifact to the exact Baseline ID and revision. Do not repeat the
folder identity in every filename. Do not use `final`, `new`, dates, or author
initials as a second versioning system.

If a publisher requires a different main filename, keep that name only in a
temporary upload copy or documented compatibility wrapper. The maintained
project source and compiled PDF remain `manuscript.tex` and `manuscript.pdf`
inside the revision folder.

## Figure and plot names

Use short manuscript-local names inside their dedicated folders:

```text
figures/fig-<NN>-principle-<slug>.<ext>
figures/fig-<NN>-model-<slug>.<ext>
figures/fig-<NN>-workflow-<slug>.<ext>
plots/plot-<NN>-<slug>.<ext>
```

Stage 3 LaTeX schematics add `-schematic` before `.tex`. Stage 4 editable
sources and exports share a basename and differ only by extension, for example
`.drawio`/`.svg`, `.pptx`/`.pdf`, or `.tex`/`.pdf`.

## Binding and invalidation

- `MANUSCRIPT_CONTEXT.md`, `WRITING_LOOP_LOG.md`, and the Stage 4 log repeat the
  Baseline ID, revision, and exact revision-folder name in metadata.
- `manuscript.tex` records `% Frozen snapshot: <baseline-id, revision>` near
  its first line.
- The revision-folder name, stored metadata, and TeX snapshot comment must
  agree. A mismatch is blocking.
- Refreezing invalidates dependent manuscript, PDF, log, figure, and plot
  artifacts until they are deliberately copied into the new sibling folder,
  rechecked, and recorded in its logs.
- Reusable skill assets and category reference libraries do not use a paper's
  revision-folder name; they keep source/provenance-oriented names.

Run `scripts/audit_artifact_names.py <revision-folder>` before compilation and
delivery. Add `--require-pdf` for final PDF delivery and `--require-stage4` when
the Stage 4 log is required.
