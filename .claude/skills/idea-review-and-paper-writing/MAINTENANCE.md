# Skill Maintenance Guide

## Directory Structure

This skill exists in two locations:

- **Primary**: `.claude/skills/idea-review-and-paper-writing/`
- **Mirror**: `.agents/skills/idea-review-and-paper-writing/`

## Maintenance Policy

**Always edit the PRIMARY location only** (`.claude/skills/`). After making changes to the primary location, synchronize to the mirror location using:

```powershell
# From project root
robocopy ".claude\skills\idea-review-and-paper-writing" ".agents\skills\idea-review-and-paper-writing" /MIR /XD .git
```

Or manually copy changed files:

```powershell
cp ".claude\skills\idea-review-and-paper-writing\<changed-file>" ".agents\skills\idea-review-and-paper-writing\<changed-file>"
```

## Why Two Locations?

- `.claude/skills/` is used by Claude Desktop and CLI
- `.agents/skills/` may be used by custom agent configurations

Keeping them synchronized ensures consistent behavior across all environments.

## Recent Updates (2026-07-30)

Integrated meeting feedback on AI-generated manuscript quality issues:

1. **Added language expression rules** (`typical-errors.md`):
   - Sentence variety and natural academic flow
   - Causality and exposition order (reason before result, macro before micro)
   - Paragraph thematic focus and purpose

2. **Enhanced narrative checks** (`manuscript-quality-gates.md`):
   - Strengthened sentence-to-sentence logic audit with language variety requirements
   - Expanded narrative causality audit with macro-to-micro progression checks
   - Added chapter arrangement verification for Section II content boundaries

3. **Updated core writing principles** (`living-user-rules.md`):
   - Explicit sentence variety requirements
   - Clear causality progression (within and between paragraphs)
   - Thematic focus and purposeful writing
   - Section II must contain only standard background, not paper-specific innovations
