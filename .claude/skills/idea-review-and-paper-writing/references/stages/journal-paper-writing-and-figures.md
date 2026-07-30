# Stage 2: Journal-paper writing and figures

This is the authoritative stage guide for turning sufficiently confirmed research ideas, derivations, evidence, and project decisions into a submission-shaped journal paper and its scientific figures.

## Contents

- Entry conditions
- Manuscript construction
- Mandatory writing gates
- Markdown artifacts
- Figure planning and tool routing
- Architecture and workflow diagrams
- Quantitative plots
- Visual language
- Captions and manuscript integration
- Scientific integrity and copyright
- Figure source and reproducibility
- Official LaTeX template preservation
- LaTeX, PDF, and visual verification
- Pre-delivery review
- Stage output

## Entry conditions

Before drafting strong novelty or guarantee claims, confirm that Stage 1 has established:

- the central problem and information boundary;
- the closest prior work and novelty delta;
- the selected model route and main assumptions;
- the scope and status of analytical results;
- the training and implementation path;
- the validation plan and available evidence.

If writing reveals a new technical idea, changed assumption, unavailable variable, unproved guarantee, or conflicting prior work, return that issue to Stage 1. Do not solve an idea-stage defect through rhetorical polishing.

## Manuscript construction

Read the complete manuscript source, bibliography, project instructions, active `MANUSCRIPT_CONTEXT.md`, exact matching case, and any reference paper or extracted literature resource named for the task before substantive editing. Read only the relevant domain guidance, and verify a bibliographic source before relying on it as evidence.

Apply the writing, organization, notation, terminology, evidence-preservation, and formal-claim rules in `references/living-user-rules.md`. Build or update the notation ledger and search the complete source for collisions.

When the user requests a complete paper, preserve a submission-shaped set of elements: title, abstract, keywords, Introduction, related background, system description and problem definition, method, central analytical results and proofs, training or algorithm realization, complexity and online implementation, experiments or an explicitly labeled experiment-design section, Conclusion, and references. Results that do not yet exist may remain as a protocol, but their absence does not justify reducing the rest of the manuscript to research notes.

Keep standard background in preliminaries and paper-specific constructions in the method section. Do not promote every module or proof stage to a main section. Move long proofs to appendices; include backpropagation-gradient formulas only when the optimization derivation is itself a contribution.

Avoid fragmented third- and fourth-level headings. Merge material that belongs to one dependency chain, keep problem before method and method before consequence, and never create one subsection merely to hold one equation. Introduce a theorem by stating its role and follow it with its design, implementation, or scope implication.

Write the Introduction around the technical tension, closest literature limitations, unresolved question, and precise contributions. Map each contribution to a construction, analytical result, algorithm, experiment, or combination of these in the body.

Close each important mathematical argument with motivation, a numbered relation, immediate symbol and dimension definitions, and a stated consequence. A model-specific derivation need not be forced into a theorem-like environment.

Use `references/technical-validity-and-implementation.md` whenever writing or revision changes the technical content. Writing must not enlarge the conclusion beyond the assumptions, derivation, information setting, or evidence established in Stage 1. Apply its experiment and evidence audit instead of inventing results or turning protocols into findings.

Structure the abstract as problem, method, theoretical or analytical result, and validation. Keep Introduction contributions synchronized with the constructions, results, algorithms, and experiments in the body.

For a repeatedly confused theoretical issue, such as scheduling uncertainty, a uniform margin, or computability, a conclusion-first explanation is permitted: state the exact conclusion, list its conditions, derive or prove it, and then state what the method cannot guarantee. Use this device to remove ambiguity, not as a reason to add explanatory bulk everywhere.

## Mandatory writing gates

Execute `references/manuscript-quality-gates.md` throughout manuscript
construction. Create or refresh the section-role matrix and notation ledger
before drafting, repeat the subsection audit after every substantive edit, and
rerun the abstract–problem–contribution alignment after any change to the
central route. Do not defer these checks to pre-delivery review.

The subsection audit is not limited to headings and symbols. It must recheck
chapter placement, sentence-to-sentence logic, narrative causality, formula
rigor, model-description completeness, and the separation and clarity of
training, validation, testing, and deployment.

## Markdown artifacts

Apply these rules to every paper-related Markdown artifact, including contexts, idea assessments, outlines, notation ledgers, literature notes, experiment plans, review notes, and response drafts:

- use `$...$` for inline mathematics and standalone `$$...$$` blocks for display mathematics;
- do not use `\(...\)` or `\[...\]`;
- leave a blank line before every opening display delimiter and after every closing display delimiter;
- leave a blank line before each ordered-list item;
- use a Unicode en dash in prose instead of a LaTeX-style double hyphen, while preserving Markdown table separators;
- before delivery, audit legacy delimiters, exposed renderer placeholders, unmatched mathematics, prose double hyphens, missing required blank lines, control characters, and Unicode replacement characters.

## Figure planning and tool routing

Create a figure only when it materially clarifies architecture, information flow, mathematical relationships, experimental evidence, comparisons, or a multi-step procedure.

Create the plan from `assets/templates/figure-plan.md`.

Before drawing, record:

- the figure's single primary purpose;
- the manuscript claim or explanation it supports;
- the intended figure type;
- required inputs, outputs, variables, and groups;
- whether it represents a conceptual structure, actual data, or a derived quantity;
- the target one-column, two-column, or page-width placement.

Do not use a decorative figure to imply a mechanism, physical interpretation, or experimental result that has not been established.

Choose the source format by scientific content:

- for architecture, workflow, and mechanism diagrams, prefer editable `.drawio`, structured SVG, or editable PowerPoint shapes in `.pptx`;
- for mathematical block diagrams tightly coupled to LaTeX, use TikZ or another editable vector source when practical;
- for quantitative plots, retain the plotting script and reviewed data, and export vector SVG or PDF whenever the marks remain tractable;
- use raster formats only for inherently raster evidence such as photographs, microscopy, dense heatmaps, or journal-mandated exports.

Select the diagram structure from the scientific claim and information topology rather than applying a generic left-input–center-model–right-output layout. Paper-type classification and layout prompts from [CCF-Figure](https://github.com/Deepshare-Official/CCF-Figure) may guide an initial composition when that skill or an equivalent image-generation workflow is callable, but treat the result as a non-authoritative concept draft. A title and abstract may identify the figure family; require the method, equations, algorithm, or an explicit module-and-edge specification before accepting topology.

Use an AI-draft-to-editable-reconstruction route only when a concept image will materially help resolve a complex or visually underdetermined layout:

1. Build the scientific content brief and select the figure family before generating an image prompt.

2. Generate at most a concept draft, record its prompt and provenance, and label any unresolved structure rather than asking the image model to invent it.

3. Audit every block, label, arrow, feedback path, and claimed relationship against the manuscript.

4. Reconstruct the accepted composition from native editable primitives in the chosen source format; do not deliver automatic raster tracing as the editable source.

5. Compare the export with the audited content brief, not merely with the appearance of the concept image.

Skip the concept-draft pass when the structure is already specified, the figure is a quantitative plot, or direct editable construction is simpler. If the user requires PowerPoint editing, build native PowerPoint shapes and connectors; do not describe a draw.io file or embedded bitmap as an editable `.pptx`.

When Draw.io Scientific Illustrator is installed and its tools are callable, use its live draw.io workflow for suitable scientific diagrams: inspect the reference, decompose it into editable primitives, launch and confirm the graph, add shapes and connectors visibly, inspect after each logical section, refine the live graph, save the completed `.drawio`, validate it, and export the requested SVG, PDF, or PNG. Do not claim this integration is available when the plugin or draw.io desktop is absent.

For draw.io work, control the graph model rather than the operating-system mouse or keyboard, and do not prebuild XML as the drawing method. Preserve stable semantic element identifiers so that later revisions can target exact shapes and edges. These practices are adapted from the MIT-licensed [Draw.io Scientific Illustrator](https://github.com/icebird1998/drawio-scientific-illustrator); retain required notices if its code or substantial documentation is copied.

For native SVG, retain a valid `viewBox`, meaningful groups or IDs, editable text when publication constraints permit, and explicit connector geometry. For PowerPoint, build the figure from editable shapes, text, groups, and connectors rather than a flattened screenshot; retain the `.pptx` source and export a publication format separately.

## Architecture and workflow diagrams

Keep diagrams synchronized with the manuscript:

- use the same symbols, terminology, subscripts, and module names;
- distinguish measured, known, latent, estimated, predicted, and unavailable quantities visually;
- distinguish offline training from online inference;
- distinguish data flow, control flow, optimization dependency, and feedback through consistent arrow styles;
- show time direction and causal history explicitly when relevant;
- keep actuator, sensor, process, and residual channels distinct until the model combines them;
- avoid arrows whose source, destination, or meaning is ambiguous;
- avoid decorative blocks that do not correspond to an equation, algorithm step, or necessary explanation.

If a figure depicts an exploratory route, label it as conceptual or proposed and do not present it as the final implemented architecture.

## Quantitative plots

Generate plots reproducibly from the actual data or reviewed numerical results. Do not hand-edit points, hide inconvenient samples, invent curves, or use smoothing that changes the scientific conclusion.

Every quantitative figure should, as applicable, provide:

- clearly named axes and physical units;
- legible ticks and scales;
- defined line, marker, color, and shading meanings;
- visible group names through a legend or direct labels;
- sample size, aggregation rule, and uncertainty representation;
- event, fault, intervention, or operating-condition markers;
- a caption that states what is plotted without claiming more than the data support.

Use consistent scales when visual comparison depends on magnitude. Avoid truncated axes, dual axes, three-dimensional decoration, excessive interpolation, and color mappings that exaggerate small differences unless the choice is explicitly justified.

When reporting uncertainty, distinguish standard deviation, standard error, confidence interval, quantile range, and deterministic bounds. Do not render one as another.

## Visual language

Use a restrained, consistent visual system across the paper:

- one font family compatible with the manuscript;
- readable text at final publication size;
- consistent line weights, markers, arrowheads, corner radii, and spacing;
- a small color palette with stable semantic meanings;
- color combinations that remain distinguishable for common color-vision deficiencies and in grayscale when practical;
- mathematical labels typeset consistently with the manuscript.

For architecture and workflow diagrams, default to a flat two-dimensional visual language. Avoid 3-D perspective, neon gradients, decorative textures, or illustration effects unless the depicted scientific quantity itself requires them.

Read `references/scientific-figure-palettes.md` and select one palette strip for each figure. Use cool gray, blue-gray, or desaturated blue-green tones for the background hierarchy and most structural elements, then reserve one brighter color from the same strip for the contribution, anomaly, intervention, warning, or comparison that truly needs attention. Do not mix independent palette strips by default, and do not spread the accent across ordinary modules until it loses its semantic role.

Do not rely on color alone when line style, marker shape, direct labeling, or grouping can preserve meaning.

Verify the current target journal's author instructions for permitted formats, resolution, dimensions, fonts, color mode, and file packaging. Do not assume one journal's figure specifications apply to another.

Prefer editable vector sources for diagrams and plots when supported, such as PDF, SVG, EPS, TikZ, or the source plotting project. Use raster images only when the content is inherently raster or the journal requires it, and retain sufficient resolution at final size.

## Captions and manuscript integration

A caption should be understandable with minimal return to the main text. Define panel labels, abbreviations, line styles, shading, bounds, and experimental conditions that are necessary to interpret the figure.

The main text should explain why the figure matters and what supported conclusion the reader should take from it. Do not merely write “Fig. X shows the results.”

Introduce each figure before or near its placement, cite every panel correctly, and keep figure numbering and cross-references synchronized after reorganization.

Do not duplicate the same evidence in a figure and a large table unless each representation serves a distinct analytical purpose.

## Scientific integrity and copyright

Do not fabricate microscopy, sensor imagery, experimental apparatus, industrial scenes, or data visualizations and present them as observed evidence.

AI-assisted or illustrative graphics may be used for a clearly conceptual schematic when appropriate, but they must not masquerade as experimental data or a real apparatus. Preserve provenance and follow the target journal's current disclosure policy.

Do not remove, add, or selectively enhance image content in a way that changes scientific interpretation. Apply any global contrast, crop, denoising, or normalization consistently and document it when scientifically material.

An open-source drawing tool does not remove copyright obligations attached to a reference figure. Prefer original composition, self-owned material, public-domain or appropriately licensed assets, or references for which reuse permission is available. When learning from a published figure without permission to reproduce it, transfer the scientific idea and create an independently organized diagram rather than tracing its distinctive expressive layout.

Match the color, spacing, and layout of a self-owned or licensed reference only to the extent authorized. For an unlicensed third-party figure, preserve the scientific relationships and clarity targets while using an independently organized composition; “high visual similarity” is not a delivery criterion.

## Figure source and reproducibility

Retain:

- editable figure source;
- scripts and reviewed input data for quantitative plots;
- font and asset dependencies;
- a mapping from source files to manuscript figure numbers;
- export settings;
- notes for any manual but scientifically neutral layout edits.

Do not make the final PDF or PNG the only surviving source when an editable or reproducible source can be retained.

## Official LaTeX template preservation

For IEEE or 《控制理论与应用》 manuscripts, first apply `references/latex-template-workflow.md`. Initialize the paper from a hash-verified official archive, write in the project copy, and preserve the document class, header files, package configuration, and layout directives. Treat a missing dependency as an environment issue to report, not permission to alter the template.

## LaTeX, PDF, and visual verification

Compile with `latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error`, resolve undefined references, and inspect overfull boxes. Compile again when cross-references require it.

Render representative or all PDF pages and inspect:

- figure readability at final size;
- clipped labels, legends, and equations;
- incorrect font substitution;
- inconsistent symbol rendering;
- line and marker visibility;
- grayscale and color interpretation when relevant;
- caption placement and panel references;
- excessive whitespace or crowded layouts;
- searchable text and embedded fonts.

For Chinese or bilingual papers, verify Chinese font embedding and consistent terminology across languages. Do not rasterize equations or ordinary figure text merely to avoid font problems.

Run `python -X utf8 scripts/audit_writing_loops.py <project-root>`,
`python -X utf8 scripts/audit_manuscript.py <project-root>`, and
`python -X utf8 scripts/audit_figures.py <figure-root>` when the relevant
artifacts exist.

## Pre-delivery review

Evaluate:

- novelty and contribution focus against the confirmed Stage 1 assessment;
- theoretical rigor and conclusion scope;
- information availability, trainability, numerical realization, and online complexity;
- experimental verifiability and evidence boundaries;
- terminology, language, notation, and organization;
- fit to the target journal;
- figure meaning, editability, reproducibility, provenance, and final-size readability.

Search for notation-ledger violations, reused reserved symbol families, duplicate LaTeX labels, unresolved citations or references, missing figure files, and stale cross-references. Compile until references stabilize; `latexmk` may perform the required repeated passes automatically.

Return links to the modified manuscript, bibliography, compiled PDF, editable figure sources, and requested exports. Report substantive changes and unresolved limits briefly. Confirm that `MANUSCRIPT_CONTEXT.md` and the matching case remain consistent with the delivered paper.

## Stage output

Stage 2 should deliver, as requested:

- a coherent manuscript with claims aligned to Stage 1;
- consistent notation and formal-result classification;
- verified references and evidence boundaries;
- publication-ready figures with editable or reproducible sources;
- correct captions and cross-references;
- compiled and visually checked PDF artifacts;
- synchronized `MANUSCRIPT_CONTEXT.md` and matching case decisions.
