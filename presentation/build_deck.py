"""
Build presentation/presentation_deck.html from presentation/deck_template.html
by embedding the two current result figures (results/arrhenius.png and
results/phase3_convergence_study.png) as base64 data URIs.

Why this exists as a script rather than a one-off: the deck template keeps
two placeholder tokens (%%ARRHENIUS_B64%% and %%PHASE3_B64%%) instead of the
images themselves, so the source file stays small, human-readable, and easy
to diff in git. Whenever either PNG in results/ is regenerated (e.g. after
re-running scripts/run_phase2_uq.py), re-run this script to refresh the
deck's embedded copies -- nothing else needs to change by hand.

Usage (from anywhere; paths below are relative to this file, not to cwd):
    python presentation/build_deck.py

Output: presentation/presentation_deck.html -- this is the file to hand to
the Artifact tool (or open directly in a browser; it is fully self-contained,
no external requests).
"""

import base64
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRESENTATION_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

TEMPLATE_PATH = PRESENTATION_DIR / "deck_template.html"
OUTPUT_PATH = PRESENTATION_DIR / "presentation_deck.html"

ARRHENIUS_PNG = RESULTS_DIR / "arrhenius.png"
PHASE3_PNG = RESULTS_DIR / "phase3_convergence_study.png"


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
