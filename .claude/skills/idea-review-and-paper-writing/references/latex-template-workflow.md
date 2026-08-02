# Official LaTeX template workflow

This is the authoritative rule set for starting and editing IEEE journal or 《控制理论与应用》 LaTeX manuscripts from official templates.

## Contents

- Template selection
- Obtain and initialize
- Editing boundary
- Journal-specific constraints
- Integrity and delivery

## Template selection

Use the registered 《控制理论与应用》 template as the default for every Chinese-language paper. When no Chinese journal is specified, select this template and proceed without asking whether to use it. Never pause the task or describe the next step as waiting for confirmation of this default. Keep using it unless the user explicitly names another target journal or instructs you to replace the Chinese default; do not infer a replacement merely from topic, institution, or a cited paper.

For English-language journal papers, use the registered IEEE journal template by default when no more specific target template has been named. In either language, an exact template explicitly selected by the user or required by the target journal overrides the corresponding default. Do not manually imitate one journal's layout in another template.

The registered sources and verified archive hashes are in `assets/latex-templates/sources.json`. Publisher pages remain authoritative if a template or instruction changes.

## Obtain and initialize

From the skill root, run:

```powershell
python -X utf8 scripts/fetch_latex_templates.py
python -X utf8 scripts/create_latex_project.py --template ieee-journal --destination <parent>/<short-baseline-id>-r<revision> --baseline-id <short-baseline-id> --revision <revision>
python -X utf8 scripts/create_latex_project.py --template control-theory-and-applications --destination <parent>/<short-baseline-id>-r<revision> --baseline-id <short-baseline-id> --revision <revision>
```

Select only the command for the active journal. The fetcher verifies the exact byte length and SHA-256 digest before accepting an archive. Archives live under the gitignored `assets/latex-templates/cache/`; never edit that cache.

The initializer copies the official files without decoding or normalizing them and writes `TEMPLATE_LOCK.json`. Work only in the initialized project copy.
The destination directory must be named `<short-baseline-id>-r<revision>`. The
initializer renames the editable template entry source to `manuscript.tex` and
records both the revision folder and original publisher filename in the lock.
The compiled file is therefore `manuscript.pdf` in the same revision folder.
Read `artifact-naming.md`; do not rename protected class/header/support files.

## Editing boundary

Edit article content: title and author data, abstract, keywords, body sections, equations, algorithms, figures, tables, captions, acknowledgments, and bibliography data.

Do not refactor, modernize, translate, re-encode, or restyle the template. Keep the document class, class options, header files, package set and order, bibliography style, margins, page and column geometry, fonts, heading definitions, caption rules, headers, footers, and template control commands unchanged.

Do not add `geometry`, font-replacement, spacing, title-format, caption-format, or margin-adjustment commands to make the paper look different. Add a package only when the manuscript technically requires it, the target journal permits it, and the change is recorded for review; never remove or replace a publisher package merely to make a local installation compile.

If compilation exposes a missing dependency or obsolete environment, report the dependency and use the publisher-prescribed environment or a lawfully obtained compatible dependency. Do not patch the template around the problem.

## Journal-specific constraints

For IEEE work, initialize from `bare_jrnl.tex` with `IEEEtran.cls` unless the exact IEEE publication supplies a more specific template, then edit `manuscript.tex` recorded in `TEMPLATE_LOCK.json`. The registered journal class defaults to a two-column article body. Respect the source warning against changing margins, column widths, line spacing, or fonts. Retain the IEEEtran notices and identify any permitted modification when redistribution requires it.

For 《控制理论与应用》, preserve the official archive's GBK encoding and all seven supplied files. Its title and front matter are assembled in a one-column class context, while the manuscript body is explicitly enclosed by `\begin{multicols}{2}` and `\end{multicols}`; therefore its final manuscript body is two-column. Do not remove or relocate that structure. The journal instruction explicitly tells authors not to change the header file or packages and currently names the legacy CTeX 2.4.5 environment. Recheck that publisher page before quoting a toolchain version; do not invent or silently substitute a different version. The current official template requests `picins.sty`; CTAN states that its license prevents distribution, so the skill does not bundle it. A missing `picins.sty` is an environment requirement, not authorization to change `kzllyyhead.tex`.

## Integrity and delivery

Before and after substantive writing, run:

```powershell
python -X utf8 scripts/audit_latex_template.py <project-directory>
```

The audit must pass before delivery. It checks protected file hashes and the main file's formatting-directive signature against `TEMPLATE_LOCK.json`. A changed protected file or signature is a template modification: restore it from a fresh initialized copy and reapply only manuscript content.

Then run the manuscript audit and the appropriate compiler. Report unresolved journal-specific toolchain or dependency limits explicitly; never claim successful compilation when only template integrity was verified.
Before compilation and delivery, run `scripts/audit_artifact_names.py` so the
context, main source, compiled PDF, logs, and figure/plot files remain bound to
the same frozen revision.
