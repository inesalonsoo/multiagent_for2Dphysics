# presentation/

The rehearsal slide deck for presenting this project (e.g. to the PI / ICFO
QNO group), as a single self-contained HTML file published via Claude
Code's Artifact tool.

**Live link:** https://claude.ai/code/artifact/74011542-f3be-47bc-8317-d1c5cbc148ec
(private by default — share from the page's share menu if you want someone
else to see it without going through Claude).

## Files here

- `deck_template.html` — **the file to edit.** All slide text, the color/
  type/layout system, and the JS (canvas double-well drawing, SVG error
  budget ruler, slide navigation) live here. The two result PNGs are NOT
  in this file — they're referenced as `%%ARRHENIUS_B64%%` and
  `%%PHASE3_B64%%` placeholders so this file stays small and diffable in
  git.
- `build_deck.py` — reads `deck_template.html`, embeds the current
  `results/arrhenius.png` and `results/phase3_convergence_study.png` as
  base64 in place of the two placeholders, writes `presentation_deck.html`.
  Stdlib only (`base64`, `pathlib`), no dependencies.
- `presentation_deck.html` — **generated, do not hand-edit.** This is the
  file actually handed to the Artifact tool. Running `build_deck.py`
  overwrites it.

## Editing text, numbers, or slide content

Edit `deck_template.html` directly — it's plain HTML/CSS/JS, no build step
needed for this. Then regenerate the publishable file:

```bash
python presentation/build_deck.py
```

## Refreshing the embedded plots

If either `results/arrhenius.png` or `results/phase3_convergence_study.png`
is regenerated (e.g. after re-running `scripts/run_phase2_uq.py`), just
re-run `build_deck.py` — it always embeds whatever is currently in
`results/`.

## Publishing / updating

From a Claude Code session with the Artifact tool, republish
`presentation/presentation_deck.html` passing the URL above as the `url`
parameter (not `file_path` alone) — that updates the *same* link instead of
minting a new one. If the tool reports it hasn't seen the latest version,
`WebFetch` the URL once first (this "checks out" the current state to the
session), then retry the publish.

## Adding a new slide

Copy an existing `<section class="slide" data-title="...">...</section>`
block, edit its content, and place it wherever it belongs in the deck order
— nothing else needs updating; slide count, nav dots, and the counter are
all computed from `document.querySelectorAll('.slide')` at load time, not
hardcoded.

## Design system, in brief

- **Palette**: derives from the actual matplotlib colors already used in
  `results/*.png` (`tab:blue`/`tab:orange`/`tab:red`), not arbitrary —
  deepened slightly for on-page contrast. CSS custom properties on `:root`,
  redefined under `prefers-color-scheme: dark` and
  `:root[data-theme="dark"|"light"]` so both the OS setting and the
  viewer's in-page theme toggle work.
- **Type**: serif (`Iowan Old Style`/Palatino/Georgia stack) for headings —
  scientific-journal register; system sans for body; `ui-monospace` for
  numbers, code, and config tuples (with `tabular-nums`). No custom
  webfonts — deliberate, since the Artifact CSP blocks font CDNs and this is
  a utilitarian talk deck, not an editorial piece.
- **Layout**: one `<section class="slide">` per slide, full-viewport,
  vertical CSS scroll-snap. Navigate with arrow keys / space, click-through
  dots, or the up/down buttons (bottom-right). `@media print` collapses it
  to a normal scrolling document with page breaks, for a PDF export /
  leave-behind.
- Full narrative design plan (why this palette, why this layout) is in the
  Claude Code conversation that built it — see `HANDOFF_PROMPT.md` if
  you're picking this up in a fresh session without that history.

## What's deliberately NOT reflected in the deck yet

- The Lean oracle scaffold (Phase 3.5, `lean/`) has its own slide, but the
  root `README.md` itself doesn't mention Lean/ax-prover anywhere yet as of
  this writing — worth checking whether that's still true before treating
  the deck's Lean slide as "the same story the README tells."
- If `results/arrhenius.png` or `results/phase3_convergence_study.png`
  change shape/story again (not just get regenerated with the same
  content), the slide *captions* in `deck_template.html` may need a text
  edit too, not just a rebuild — `build_deck.py` only swaps the image
  bytes, it does not know whether the caption still describes what the new
  image shows.
