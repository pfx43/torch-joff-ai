# Scientific figure palettes

Use this reference for Stage 2 scientific figures that need a restrained, reusable color system.

## Contents

- Selection contract
- Curated palette strips
- Selection by scientific intent
- Semantic role presets
- Verification

## Selection contract

- Select exactly one named palette strip for a figure. Use a subset of that strip when fewer colors are sufficient; do not combine independent strips unless the user explicitly requests a cross-palette design.
- Treat pure white, black, transparency, and journal-mandated grayscale as substrate or ink rather than a second decorative palette.
- Default to cool gray, blue-gray, or desaturated blue-green structural tones. Reserve one brighter swatch from the same strip for the most important innovation, anomaly, intervention, warning, or comparison.
- Keep architecture and workflow diagrams to no more than three semantic color groups. A quantitative plot may use more swatches from the selected strip when the data require them, but must also use line style, marker shape, direct labeling, or grouping.
- Keep each color's meaning stable within a figure and, when practical, across the paper. Do not reuse the accent for unrelated semantics.
- Prefer a near-white background and dark text. Check text, thin lines, and small markers at final publication size rather than judging large screen previews.

The frozen palette codes below were curated from [Coolors Trending Palettes](https://coolors.co/palettes/trending) on 2026-07-29. The live page changes over time; use these recorded codes for reproducibility instead of assuming that the current ranking is unchanged.

Use the editable [SVG palette-strip overview](../assets/palettes/scientific-figure-palettes.svg) for visual selection; keep this reference file as the authoritative source for the codes and role mappings.

## Curated palette strips

Treat the four palettes in the user's first selection screenshot and the red-ticked palettes in the later screenshots as the preferred set. The final two rows remain useful secondary alternatives.

| ID | Coolors strip and original hex sequence | Recommended use |
|---|---|---|
| `summer-ocean-breeze` | Summer Ocean Breeze — `#E63946`, `#F1FAEE`, `#A8DADC`, `#457B9D`, `#1D3557` | Default rigorous blue-gray figure with one clear red highlight |
| `dark-sunset` | Dark Sunset — `#335C67`, `#FFF3B0`, `#E09F3E`, `#9E2A2B`, `#540B0E` | Dark teal structure with warm comparison, risk, or transition states |
| `earthy-green` | Earthy Green — `#CAD2C5`, `#84A98C`, `#52796F`, `#354F52`, `#2F3E46` | Restrained green-gray process, observer, or industrial-system diagram |
| `fresh-greens` | Fresh Greens — `#386641`, `#6A994E`, `#A7C957`, `#F2E8CF`, `#BC4749` | Environmental, biological, energy, or condition-monitoring figure with red contrast |
| `ocean-breeze` | Ocean Breeze — `#03045E`, `#0077B6`, `#00B4D8`, `#90E0EF`, `#CAF0F8` | Fully cool architecture, layered hierarchy, or multi-series plot |
| `sunny-beach-day` | Sunny Beach Day — `#264653`, `#2A9D8F`, `#E9C46A`, `#F4A261`, `#E76F51` | Cross-domain framework or balanced method comparison |
| `warm-autumn-glow` | Warm Autumn Glow — `#003049`, `#D62828`, `#F77F00`, `#FCBF49`, `#EAE2B7` | Dark-blue foundation with red-to-yellow severity or stage progression |
| `whimsical-dreams` | Whimsical Dreams — `#9F9AA4`, `#E7CFCD`, `#CFD8D7`, `#B5C9C3`, `#CAB1BD` | Muted grouping regions, background panels, or a soft conceptual overview |
| `vibrant-spring` | Vibrant Spring — `#3C1642`, `#086375`, `#1DD3B0`, `#AFFC41`, `#B2FF9E` | High-contrast conceptual mechanism; keep lime accents sparse |
| `cool-waters` | Cool Waters — `#22577A`, `#38A3A5`, `#57CC99`, `#80ED99`, `#C7F9CC` | Cool blue-green feedback, estimation, sensing, or closed-loop diagram |
| `watermelon-sorbet` | Watermelon Sorbet — `#EF476F`, `#FFD166`, `#06D6A0`, `#118AB2`, `#073B4C` | Multi-category comparison or cross-domain framework with a dark teal anchor |
| `sunny-beach-day-deep-orange` | Sunny Beach Day (Deep Orange) — `#001524`, `#15616D`, `#FFECD1`, `#FF7D00`, `#78290F` | Strong nominal-versus-anomalous or cold-versus-hot comparison |
| `fiery-ocean` | Fiery Ocean — `#780000`, `#C1121F`, `#FDF0D5`, `#003049`, `#669BBC` | Fault, alarm, unsafe-state, or before–after comparison |
| `refreshing-summer-fun` | Refreshing Summer Fun — `#8ECAE6`, `#219EBC`, `#023047`, `#FFB703`, `#FB8500` | Architecture or workflow figure needing yellow or orange emphasis |
| `deep-sea` | Deep Sea — `#0D1B2A`, `#1B263B`, `#415A77`, `#778DA9`, `#E0E1DD` | Conservative theorem, observer, or control-flow diagram without a vivid warning color |
| `bold-berry` | Bold Berry — `#F9DBBD`, `#FFA5AB`, `#DA627D`, `#A53860`, `#450920` | Ordered severity, diagnostic classes, or ablation emphasis with a dark berry anchor |
| `pastel-dreamland-adventure` | Pastel Dreamland Adventure — `#CDB4DB`, `#FFC8DD`, `#FFAFCC`, `#BDE0FE`, `#A2D2FF` | Soft conceptual grouping or multi-panel background; use dark neutral text and outlines |
| `bold-hues` | Bold Hues — `#F72585`, `#7209B7`, `#3A0CA3`, `#4361EE`, `#4CC9F0` | Vivid categorical comparison or benchmark overview; avoid using every swatch as an equal accent |
| `earthy-tones-muted` | Earthy Tones (Muted) — `#EDAFB8`, `#F7E1D7`, `#DEDBD2`, `#B0C4B1`, `#4A5759` | Muted lifecycle, materials, health, or cross-domain grouping with a charcoal anchor |
| `ocean-sunset` | Ocean Sunset — `#001219`, `#005F73`, `#0A9396`, `#94D2BD`, `#E9D8A6`, `#EE9B00`, `#CA6702`, `#BB3E03`, `#AE2012`, `#9B2226` | Multi-stage system or multi-series plot; normally use only four to six swatches |
| `black-gold-elegance` | Black & Gold Elegance — `#000000`, `#14213D`, `#FCA311`, `#E5E5E5`, `#FFFFFF` | Minimal high-contrast diagram with a single gold innovation marker |

## Selection by scientific intent

| Scientific intent | Prefer |
|---|---|
| General method, model architecture, or paper overview | `summer-ocean-breeze`, `ocean-breeze`, or `deep-sea` |
| Observer, estimation, sensing, control loop, or industrial process | `cool-waters`, `earthy-green`, or `deep-sea` |
| Fault, anomaly, alarm, intervention, or severity progression | `fiery-ocean`, `dark-sunset`, `warm-autumn-glow`, or `sunny-beach-day-deep-orange` |
| Environmental, biological, energy, or condition-monitoring application | `fresh-greens`, `cool-waters`, or `sunny-beach-day` |
| Cross-domain framework or several categorical groups | `sunny-beach-day`, `watermelon-sorbet`, or `bold-berry` |
| Soft background grouping without a strong warning color | `whimsical-dreams`, `pastel-dreamland-adventure`, or `earthy-tones-muted` |
| Deliberately vivid conceptual mechanism or benchmark overview | `vibrant-spring` or `bold-hues`; keep the brightest swatches to small emphasis regions |

## Semantic role presets

Use `summer-ocean-breeze` by default:

| Role | Hex |
|---|---|
| Page or panel background | `#F1FAEE` |
| Primary text, main outline, or principal flow | `#1D3557` |
| Secondary structure or comparison flow | `#457B9D` |
| Low-emphasis fill, grouping region, or uncertainty band | `#A8DADC` |
| Sparse accent for contribution, anomaly, or intervention | `#E63946` |

Use `refreshing-summer-fun` when warm emphasis is preferable:

| Role | Hex |
|---|---|
| Primary text or outline | `#023047` |
| Secondary structure | `#219EBC` |
| Low-emphasis fill | `#8ECAE6` |
| Primary accent | `#FFB703` |
| Warning or secondary accent, only when a second emphasis is necessary | `#FB8500` |

Use `deep-sea` when the figure should remain entirely cool and conservative:

| Role | Hex |
|---|---|
| Primary text or outline | `#0D1B2A` |
| Principal structure | `#1B263B` |
| Secondary structure | `#415A77` |
| Low-emphasis fill | `#778DA9` |
| Background or grouping region | `#E0E1DD` |

## Verification

- Record the selected palette ID, role mapping, and any unused swatches in the figure plan.
- Verify grayscale separation and common color-vision deficiencies; revise lightness, line style, markers, or labels before introducing a second palette.
- Verify the accent occupies a small visual area and still identifies the intended scientific meaning.
- Verify the export preserves the selected hex values or document why the target journal's color conversion changed them.
- After changing the curated palette table, run `python -X utf8 scripts/render_palette_strips.py` and then `python -X utf8 scripts/render_palette_strips.py --check`.
