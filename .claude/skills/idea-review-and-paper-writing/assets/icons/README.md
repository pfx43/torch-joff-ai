# Scientific icon assets

This directory contains only locally reusable icon files whose provenance and
license have been checked. It is an asset store, not a visual-precedent case
library. Register every file in `icon-registry.md` before using it in a figure.

## Candidate sources

- Alibaba Iconfont: <https://www.iconfont.cn/>
- EmojiAll: <https://www.emojiall.com/zh-hans>

Use these official operational references when checking a candidate:

- Iconfont download/help page: <https://www.iconfont.cn/help/detail?helptype=code>
- EmojiAll copyright notice: <https://www.emojiall.com/zh-hans/notices>

Iconfont supports downloadable SVG/AI/PNG assets, but availability on the
platform is not blanket permission to reuse every contributor's icon. Inspect
the exact icon/library/author terms and record them.

EmojiAll offers downloadable emoji artwork, but its own copyright notice and
the licenses of manufacturer-specific emoji styles may restrict commercial
reuse. Do not treat a Unicode character, EmojiAll's original artwork, and a
vendor-rendered emoji image as the same asset. Verify the exact artwork license;
when publication rights are unclear or incompatible, use an independently
drawn academic symbol or an explicitly compatible open-source emoji set.

## Web and download route

- For ordinary source discovery, search the current official page and record
  the exact asset URL, author/library, access date, and license page.
- When the page is dynamic, requires a signed-in session, or needs interactive
  selection/download, use the Codex Chrome browser plugin. Do not inspect
  cookies, passwords, profiles, or hidden session data.
- Download only the exact selected asset after resolving its license. Do not
  bulk scrape a library merely because the page is accessible.
- Prefer SVG for editable scientific figures. Keep an untouched source copy and
  create a separately named normalized/recolored derivative when needed.
- Never embed remote scripts, icon-font JavaScript, tracking code, or an entire
  web font in a manuscript figure when one local vector asset is sufficient.

## Academic-use filter

Use an icon only when it gives a small component a clearer scientific meaning,
such as data acquisition, sensing, storage, matrix/tensor input, neural layers,
loss aggregation, monitoring, or decision output. Reject expressive faces,
mascots, glossy 3-D illustrations, decorative technology marks, brand logos,
business-process clip art, and poster-like artwork.

Normalize approved assets to the manuscript's line weight and palette while
preserving license notices and attribution. A visual redesign does not erase
the source license.

## Storage names

Use source-oriented names for reusable icons:

```text
icon-<source>-<asset-id>-<short-slug>-source.<ext>
icon-<source>-<asset-id>-<short-slug>-normalized.<ext>
```

Paper-specific derivatives use the short `fig-<NN>-<type>-<slug>` basename
inside the active revision folder and must be recorded in the Stage 4 figure
plan.
