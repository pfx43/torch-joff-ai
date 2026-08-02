# Scientific-figure composition rules

Use this Stage 4 reference for principle schematics, model structure diagrams,
and task workflow diagrams. It governs scientific topology and editable visual
construction; quantitative data plotting remains in the separate Python
plotting workflow.

## Contents

- Universal composition contract
- Model structure diagrams
- Task workflow diagrams
- Principle schematics
- Special composition patterns
- Arrow and connector contract
- Shapes, icons, text, and formulas
- Tool and source routing
- Captions and integration
- Integrity, copyright, and verification

## Universal composition contract

- Do not place the complete figure title inside the graphic. The LaTeX caption
  owns the figure title and explanation.
- Use titled major frames to define meaningful stages or modules; use contained
  shapes for internal operations and details.
- Every visual element must correspond to an equation, algorithm step, data
  object, task state, or necessary explanation.
- Preserve one clear reading direction and a hierarchy visible before the text
  is read.
- Keep scientific topology independent from the visual style of a reference
  paper. Borrow a composition principle, not a distinctive complete figure.
- Keep symbols, terminology, abbreviations, and semantic color roles identical
  to the frozen context and manuscript.

## Model structure diagrams

### Major frames

Use major frames for system-level units, for example:

- overall forward model;
- encoder, observer, predictor, decoder, controller, or residual generator;
- loss/constraint/optimization module;
- task-output module when it is genuinely separate.

Every major frame has a concise title. Arrange major frames in computational or
scientific dependency order, not by aesthetic symmetry alone.

### Contained details

Within each frame, show only details needed to understand the contribution:

- complete input and output ports;
- state, latent, estimated, or predicted objects;
- essential transformations and interfaces;
- distinctive neural layers or repeated blocks;
- loss terms and the quantities they constrain;
- parameters shared, fixed, learned, or transferred when visually necessary.

Represent ordinary layers with simple shapes or restrained motifs. Use repeated
stacking for repeated layers instead of listing every instance. Avoid showing
generic activations, normalization, or fully connected layers unless their
placement or parameter sharing matters to the paper.

Follow:

`input -> overall forward structure -> local modules -> task output`

Place loss/optimization frames after or adjacent to the forward quantities they
consume, while maintaining unambiguous directed connectors.

## Task workflow diagrams

Use major frames for ordered steps, typically selected from:

- data acquisition;
- cleaning, synchronization, and preprocessing;
- offline dataset construction;
- offline model learning/design;
- model selection or threshold preparation when part of the actual workflow;
- online measurement and inference;
- monitoring, diagnosis, estimation, prediction, or control output;
- decision, alarm, or intervention when scientifically defined.

Within each step frame, use a short one-way chain of shapes. Distinguish offline
and online regions explicitly. Do not place an online label, test outcome, or
future quantity inside an offline training frame, or vice versa.

The task flow must explain operational order and information availability, not
repeat the model structure at a smaller scale. A model frame may be referenced
as one component without redrawing all of its layers.

## Principle schematics

Principle figures have no universal template. Before drawing one:

1. define the exact mechanism, relation, or contrast to explain;

2. search current primary papers from the same technical community, prioritizing
   top-tier journals/conferences appropriate to the field;

3. classify how the references externalize the principle—geometric relation,
   information separation, state evolution, transformation, boundary, energy,
   probability, or another scientific form;

4. record a figure-exemplar case with source and rights metadata;

5. select an independent composition that expresses the current paper's own
   equations and objects.

Do not select a principle layout merely because it is visually attractive. If
the mechanism has no defensible visual representation, keep it in equations and
prose rather than inventing one.

## Special composition patterns

### Local zoom-in

Use a zoomed inset when a major frame contains one contribution-relevant local
mechanism that would otherwise be unreadable. Connect the source region to the
inset with non-data guide lines distinct from computational arrows. The inset
must clarify, not duplicate, the main frame.

### Repeated-module stacking

Use offset or stacked repeated shapes to denote repeated layers, time steps,
rules, channels, or ensemble members. State the multiplicity through a compact
index or ellipsis and define it in the caption. Do not use pseudo-3D depth that
suggests an unclaimed physical dimension.

### Parallel branches

Use aligned branches only for scientifically parallel objects. Merge them at a
defined operator/shape; do not let arrows converge into empty space.

## Arrow and connector contract

- Every arrow starts at a visible shape or named port and ends at another.
- Use one arrowhead and one direction per connector.
- Arrows may cross major frames when the scientific dependency crosses them.
- Keep the visible topology acyclic. Eliminate feedback-looking ornamental
  loops and ambiguous bidirectional arrows.
- For recurrent/iterative methods, unroll time or iteration, use $k\to k+1$ or
  a repeated-block notation, or display the update equation next to the chain.
  Preserve recurrence without a closed visual loop.
- Define distinct styles for data flow, parameter/loss dependency, control, and
  non-computational guide lines only when the distinction is necessary.
- Avoid line crossings; when unavoidable, use spacing or an explicit bridge and
  ensure the direction remains obvious.
- No dangling connector, arrow into a frame boundary without a port, or arrow
  whose label is an unexplained verb.

## Shapes, icons, text, and formulas

Do not make every small component an identical basic rectangle. Select shapes
or restrained editable icons by scientific function:

- stacked sheets/table motif for datasets;
- matrix/tensor motif for arrays;
- compact layer stack for neural blocks;
- sensor or actuator outline only when the physical object matters;
- summation, distance, loss, or decision shapes for exact operators;
- standard geometric shapes for ordinary transformations.

Icons must be academic, flat, restrained, licensed, and semantically stable.
Reject clip-art, stock-business people, glossy gradients, decorative AI/circuit
symbols, dashboards, poster headings, and marketing-style illustrations.

Candidate icon sources include Alibaba Iconfont and EmojiAll. Read
`assets/icons/README.md`, verify the exact asset/author/license rather than the
platform name alone, and register every downloaded file in
`assets/icons/icon-registry.md`. Prefer editable SVG. Emoji/vendor artwork with
unclear or noncommercial-only terms is blocked for a publication whose reuse
rights are incompatible; replace it with an independently drawn symbol or a
verified compatible open-source asset.

Use short module names and established symbols. Avoid sentences, paragraphs,
long formulas, redundant abbreviations, and claims such as “novel” or “high
performance” inside the image. Include at most the compact equation needed to
distinguish a principle; place definitions and derivations in the caption/body.

## Tool and source routing

- Architecture/workflow/mechanism: editable `.drawio`, structured SVG, native
  PowerPoint `.pptx`, or TikZ/LaTeX.
- Quantitative plots: reviewed outputs from the dedicated external Python
  plotting library plus retained data/code provenance; do not reproduce that
  library inside this skill.
- Raster: only inherently raster evidence or a journal-mandated export.
- Interactive icon search/download: use the Codex Chrome browser plugin when a
  dynamic page, signed-in session, or interactive asset selection requires it;
  record the exact URL, access date, author/library, and license. Do not bulk
  scrape a site or inspect browser credentials/session storage.

For an AI concept draft, first fix scientific content and topology, generate at
most a labeled concept, audit every block/label/arrow, and reconstruct the
accepted design with native editable primitives. A bitmap, screenshot, or
automatic raster trace is not an editable scientific source.

When Draw.io Scientific Illustrator or CCF-Figure is callable, treat it as a
draft/construction aid. Validate the graph model, semantic IDs, groups,
connectors, notices, and export; do not claim availability when absent.

For SVG, retain a valid `viewBox`, meaningful groups/IDs, editable text when
permitted, and explicit connectors. For PowerPoint, use native shapes, text,
connectors, and groups rather than an embedded flattened image.

## Captions and integration

The caption defines panels, abbreviations, symbols, styles, line meanings,
shading, bounds, conditions, repeated-module indices, and required provenance.
The body states the figure's scientific purpose and the conclusion or
explanation it supports; “Fig. X shows...” alone is insufficient.

Introduce each figure near its placement, cite every panel, and synchronize
figure number, short canonical filename inside the revision folder, LaTeX label, caption, body, and result scope. Do not
repeat the same evidence in a table unless the two representations answer
different reader questions.

## Integrity, copyright, and verification

Do not fabricate apparatus, industrial scenes, sensors, microscopy, or data as
observations. Label conceptual AI-assisted schematics honestly. Do not alter
quantitative or raster content in a way that changes interpretation.

Tool access does not grant reuse rights. Store a source image in the figure-case
library only when authorized; otherwise retain source metadata/link and a
structural analysis. Independently organize any derived composition and follow
current journal disclosure and copyright requirements.

Retain editable source, plotting provenance, fonts/assets/licenses, figure
mapping, export settings, and neutral layout-edit notes. Run
`scripts/audit_figures.py`, inspect final-size exports, and check clipping,
fonts, equations, labels, arrow topology, grayscale/color meaning, captions,
panels, whitespace, searchable text, and embedded fonts.
