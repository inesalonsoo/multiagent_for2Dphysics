"""
Build presentation/presentation_deck.html from presentation/deck_template.html
by embedding the two current figures (results/arrhenius.png and
results/phase3_rescore.png) as base64 data URIs.

The template keeps placeholders (%%ARRHENIUS_B64%%, %%PHASE3_B64%%) instead of
the images, so it stays small and readable in git. Re-run this script after
regenerating either figure.

Usage (paths are relative to this file, not the working directory):
    python presentation/build_deck.py

Output: presentation/presentation_deck.html, fully self-contained (no
external requests).
"""

import base64
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRESENTATION_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

TEMPLATE_PATH = PRESENTATION_DIR / "deck_template.html"
OUTPUT_PATH = PRESENTATION_DIR / "presentation_deck.html"

ARRHENIUS_PNG = RESULTS_DIR / "arrhenius.png"
PHASE3_PNG = RESULTS_DIR / "phase3_rescore.png"


def b64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    arrhenius_b64 = b64_of(ARRHENIUS_PNG)
    phase3_b64 = b64_of(PHASE3_PNG)

    output = template.replace("%%ARRHENIUS_B64%%", arrhenius_b64)
    output = output.replace("%%PHASE3_B64%%", phase3_b64)

    if "%%" in output:
        raise RuntimeError(
            "A placeholder token was left unreplaced -- check deck_template.html "
            "for a %%...%% marker this script doesn't know about."
        )

    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(output):,} bytes)")
    print(f"  embedded {ARRHENIUS_PNG.name} ({ARRHENIUS_PNG.stat().st_size:,} bytes)")
    print(f"  embedded {PHASE3_PNG.name} ({PHASE3_PNG.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
