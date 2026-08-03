# Handoff prompt — presentation deck edits

Copy everything below the line into a fresh Claude Code session (in this
repo, `C:\Users\Ines\dev\multiagent_for2Dphysics`) to bring it up to speed
on the presentation deck without re-deriving any of this from scratch.
This file is a snapshot as of 2026-07-30 — re-verify anything load-bearing
against the live repo before trusting it; the note at the bottom explains
why that matters here specifically.

---

## What you're picking up

I (a previous Claude Code session) built and am maintaining a slide-deck
Artifact that presents this repo — `moire-msm-engine`, a project
benchmarking a three-agent LLM verification loop against exact stochastic
double-well physics — to the user's PI at ICFO's Quantum Nano-Optoelectronics
group. The user is a student learning Claude Code / agentic workflows and
wants the deck to be editable by hand, not something only I can touch.

**Live artifact:** https://claude.ai/code/artifact/74011542-f3be-47bc-8317-d1c5cbc148ec
(private; 12 slides; a single self-contained HTML page — no external
requests, both light/dark themes supported, keyboard/click navigation).

**Durable source**, already committed-ready in this repo (not in any temp
directory — do not go looking in `AppData\Local\Temp` for this):

```
presentation/
├── deck_template.html    <- EDIT THIS for any text/number/slide change
├── build_deck.py         <- run this after editing to regenerate the output
├── presentation_deck.html <- GENERATED, do not hand-edit; this is what gets published
└── README.md             <- fuller editing guide, read this first
```

Read `presentation/README.md` now if you haven't — it has the mechanical
edit/rebuild/republish loop. This document adds the *content* context that
README doesn't cover: what each slide claims, where those numbers come
from, and what's fragile.

## How to make an edit, end to end

1. Edit `presentation/deck_template.html` (plain HTML/CSS/JS — the two
   embedded plots are `%%ARRHENIUS_B64%%` / `%%PHASE3_B64%%` placeholders,
   not inline in this file, so it stays small and diffable).
2. `python presentation/build_deck.py` — regenerates
   `presentation/presentation_deck.html` by embedding whatever is
   currently in `results/arrhenius.png` and
   `results/phase3_convergence_study.png`.
3. Publish with the Artifact tool: `file_path` =
   `presentation/presentation_deck.html`, `url` =
   `https://claude.ai/code/artifact/74011542-f3be-47bc-8317-d1c5cbc148ec`,
   `favicon` = `⚛️` (keep this exact emoji — favicon changes read as "new
   artifact" to the user, only change it on a genuine topic pivot).
   **If the tool refuses because "this session hasn't viewed the latest
   version"**: `WebFetch` the URL once first (any prompt works, e.g. "confirm
   what this page is"), then retry the publish. That fetch is expensive in
   context (the tool returns the full page source, ~200KB+, not a summary,
   because the page is an Artifact) — expect it, don't assume something
   broke.

## Design system (so edits stay consistent, not re-derived)

- **Palette is not arbitrary**: the blue/orange/red accents are pulled from
  the actual matplotlib `tab:blue`/`tab:orange`/`tab:red` colors already
  used in `results/*.png`, slightly deepened for on-page contrast — this
  was a deliberate choice so the deck and the embedded real plots read as
  one visual system. Don't introduce a new accent color without a reason;
  if you need a fourth semantic color, derive it the same way (from an
  existing plot convention) rather than picking arbitrarily.
- **Both themes are real, not an afterthought**: tokens are CSS custom
  properties on `:root`, overridden under `@media (prefers-color-scheme:
  dark)` and again under `:root[data-theme="dark"|"light"]` (the in-page
  toggle). If you add a new color anywhere, add both variants.
- **Typography**: serif stack (`Iowan Old Style`, Palatino, Georgia) for
  headings, system sans for body, `ui-monospace` for anything numeric
  (`tabular-nums` is set so columns of digits align). No custom webfonts —
  deliberate: the Artifact CSP blocks font CDNs, and this is a utilitarian
  talk deck, not an editorial piece where a bespoke typeface would be worth
  the embedding cost.
- **Layout**: one `<section class="slide" data-title="...">` per slide,
  full-viewport, CSS scroll-snap. The nav dots, counter, and keyboard
  handling all derive from `document.querySelectorAll('.slide')` at load
  — adding/removing/reordering `<section>` blocks needs no other change.
- **The double-well canvas drawings and the Phase 2 SVG "error budget
  ruler" are computed from the real formulas/numbers in JS**, not
  hand-drawn images — if a physics parameter changes (e.g. a different
  reference β), update the constants in the `<script>` block
  (`drawWell`'s `tilt`, or the ruler's `mean`/`statRel`/`sysRel`/
  `totalRel`/`analytical` values), don't redraw anything by hand.

## What each slide currently claims, and where the numbers come from

Cross-check these against the live repo before trusting them for a new
edit — this list is a snapshot, see the caution at the end.

1. **Title** — no factual claims.
2. **The question** — states the moiré-scope caveat up front (2-component
   field, 3-fold AA/AB/BA symmetry, elastic soliton network vs. this
   project's scalar double well). Wording should track `README.md`'s own
   "Where moiré materials fit" subsection — check that section still says
   this before assuming the slide is still accurate.
3. **Benchmark system** — `V(x) = A(x²−1)² + b·x`,
   `dx = −V'(x)dt + √(2/β)dW`. Static, unlikely to change unless the
   physics itself changes (it hasn't).
4. **Phase 1** — slope −0.981 vs. exact −1 (1.87% deviation, β≤7); Gate 1
   (2 macrostates, 6/6 replicas, all β) clean; two bugs found (sparse-count
   bias at high β, under-converged lag). Source: `PROJECT_STATE.md` §9,
   2026-07-11/07-12 entries; `README.md`'s "Phase 1" section.
   Image: `results/arrhenius.png`, now a two-panel figure (log-rate +
   linear residual-%) — this was a real fix (the original single-panel log
   plot hid Phase 2's error bars visually); if this image ever reverts to
   single-panel, the slide's caption text needs to change too, not just the
   embedded bytes.
5. **Phase 2** — error-budget ruler at β=5: statistical ±1.08%, systematic
   ±2.97%, total ±3.16%, analytical rate 0.012133, measured mean ≈0.011782.
   Full-sweep total bands at β=3..7: 7.12% / 6.23% / 3.16% / 3.22% / 3.66%.
   Source: `PROJECT_STATE.md` §9, 2026-07-12 "Phase 2 built and PASSED"
   entry (Part B); `scripts/run_phase2_uq.py`.
6. **Phase 3 architecture** — Orchestrator/Optimizer/Validator mapped to
   Ax-Prover's Orchestrator/Prover/Verifier (arXiv:2510.12787, Breen et
   al.). Static description, unlikely to go stale unless the architecture
   itself is refactored.
7. **Phase 3 result** — 4 runs, iterations 6/4/5/5, accepted configs
   (n=60,lag=20)/(n=50,lag=50)/(n=75,lag=20)/(n=100,lag=10), rates
   0.011853/0.011797/0.011774/0.011764, band [0.011749, 0.012517],
   analytical 0.012133. The "one-sided, checked directly" callout: all 4
   accepted + all 20 rejected configs measured LOW; a direct (non-agentic)
   check found a too-short lag overestimates the rate by up to 82%, and the
   real Validator function correctly rejects it — confirmed symmetric, but
   no real *agentic* run has produced an overestimate yet. Source:
   `PROJECT_STATE.md` §9, 2026-07-12 and 2026-07-17 entries;
   `results/phase3_convergence_study_report.md`; `README.md`'s Phase 3
   section (the "one asymmetry worth stating explicitly" paragraph).
8. **Lean oracle (Phase 3.5)** — 7 theorem statements in
   `lean/Oracle/Potential.lean`, all ending `sorry`; scope boundary
   ("Claude Code writes statements, ax-prover writes proofs") is in
   `CLAUDE.md`'s "Lean / ax-prover scope boundaries" section. **As of this
   writing, the root `README.md` does NOT mention the Lean oracle
   anywhere** — this slide is currently "ahead of" the public-facing
   README. Worth asking the human whether README should be updated to
   match before treating this slide as uncontroversial; don't just assume
   it should stay because it's already in the deck.
9. **What I decided** (authorship / AI-vs-human slide) — bullet points
   pulled from specific dated `PROJECT_STATE.md` §9 entries (the Δf=0
   diagnosis, the "test it directly, don't tighten the tolerance"
   correction made twice, the Lean scope boundary). If you add a new
   bullet here, it should cite an actual dated entry, not a generic
   restatement — the whole point of this slide is specificity as evidence.
10. **Known limitations** — split into "Fixed since the last internal
    review" (moiré scope, invisible error bars — both green/resolved) and
    "Still open" (one-sided-in-real-runs, N=4, unverified L-conversion,
    Lean statements-not-proofs — orange/caution). If any "still open" item
    gets resolved, move it to the fixed column with a one-line note on how,
    rather than deleting it — the deck's whole voice is "here's what
    changed and why," not just a static state.
11. **Next steps** — two live options per `PROJECT_STATE.md`'s own "Next"
    line: Phase 4 (2D deployment) or Module 3.9 (run `ax-prover prove` on
    the Lean scaffold). Both still open as of this writing.
12. **Close** — test count "103 passed, 2 skipped." This number moves
    almost every session; check `PROJECT_STATE.md`'s latest session-log
    entry or just run `pytest tests/ -q` yourself before trusting it.

## Things to verify, not assume, before you touch content

- **This repo moves fast and is sometimes ahead of its own git history.**
  As of the last check in this conversation, `PROJECT_STATE.md`,
  `README.md`, and `results/phase3_convergence_study_report.md` all showed
  as *modified but uncommitted* in `git status`, despite the most recent
  commit's message (`4279700`, "Archive pre-pivot dead ends, remove dead
  stub files, fix README overclaim") explicitly claiming to include the
  README wording fix. Run `git status` and `git diff --stat` yourself
  before assuming the working tree matches any specific commit's story —
  don't trust a commit message's claims over the actual diff.
- **There may be a second, stale copy of this project** at
  `C:\Users\Ines\OneDrive - ICFO\Documents\ICFO\multiagent_for2Dphysics`.
  The canonical, git-remote-tracked copy (as of this writing) is
  `C:\Users\Ines\dev\multiagent_for2Dphysics` (has `origin/main`); the
  OneDrive path was the original location before a move, and had no git
  remote when last checked. Confirm with the human which one is
  authoritative if you're ever unsure which repo you're in — don't infer
  it silently the way this note is doing.
- **Every number in the "what each slide claims" section above came from
  reading `PROJECT_STATE.md`/`README.md`/`results/*.png` directly** at one
  point in time. Re-read the source before repeating a number in a new
  edit, especially anything with a specific decimal (rates, percentages,
  test counts) — these are exactly the kind of detail that silently goes
  stale and reads as sloppy if wrong in front of the PI.
- Don't `git commit` any of this unless the human explicitly asks — the
  presentation files are currently untracked/uncommitted in this repo,
  which may be intentional (rehearsal-only) rather than an oversight.
