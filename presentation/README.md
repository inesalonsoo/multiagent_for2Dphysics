# presentation/

The slide deck for presenting the project: one self-contained HTML file
that opens in any browser and can be published with Claude Code's Artifact
tool.

**Live link:** https://claude.ai/code/artifact/74011542-f3be-47bc-8317-d1c5cbc148ec
(private by default; share it from the page's share menu).

## Files

- `deck_template.html`: **the file to edit.** All slide text, styling and
  scripts (the double-well drawing and slide navigation). The two figures
  are placeholders (`%%ARRHENIUS_B64%%`, `%%PHASE3_B64%%`), so the file
  stays small and easy to compare in git.
- `build_deck.py`: embeds `results/arrhenius.png` and
  `results/phase3_rescore.png` into the template and writes
  `presentation_deck.html`. Standard library only.
- `presentation_deck.html`: **generated; do not edit by hand.** This is the
  file to open or publish.

## Updating

```bash
python presentation/build_deck.py
```

Run it after editing the template or regenerating either figure. If a
figure changes what it shows, update its caption in the template too: the
script only swaps image bytes.

To update the live page, republish `presentation_deck.html` with the
Artifact tool, passing the link above as `url` so the same link is kept.

## Design

- One full-screen `<section class="slide">` per slide. Navigate with the
  arrow keys, the dots, or the buttons at the bottom right. Slide count and
  navigation are computed at load time, so adding a slide needs no other
  change.
- Colors follow the matplotlib colors used in the figures and adapt to
  light and dark mode. Printing gives a normal paged document.
