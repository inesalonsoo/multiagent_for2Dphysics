# Phase 3 convergence-robustness study — results and analysis

**Date:** 2026-07-12
**Model:** `anthropic:claude-sonnet-5` (both Optimizer and Validator)
**Reference:** β=5.0 (Phase 1/2's clean β≤7 range), one fixed 15,000,000-step
trajectory (seed=7) reused across all 8 repetitions
**Rate tolerance:** ±3.16% (Phase 2's own total statistical⊕systematic band,
reused via `agents.validator.load_rate_tolerance()`)
**Raw data:** `results/phase3_convergence_study/run_01..08_ledger.json`,
`results/phase3_convergence_study.png`

## Headline result

| Claim | Status |
|---|---|
| Convergence rate | **8/8 runs converged** (100%), every run on iteration 1 |
| Accepted physics inside Phase 2's UQ band | **8/8 inside** — trivially, see below |
| Search paths differ across runs | **NO — all 8 runs are byte-identical** |

The study was designed to demonstrate two things in tension: that the search
explores (paths differ) and that the outcome is bounded regardless (accepted
physics agrees). **Only the second half showed up, and it showed up in a way
that doesn't actually prove what it was meant to prove.** This is reported
here as a real finding, not smoothed over — exactly the failure mode flagged
in advance: *"if the outcomes were identical every time you couldn't tell
whether the physics gate was constraining anything."* That is precisely what
happened.

## What actually happened, in full

Every one of the 8 independent real-agent runs:
1. Proposed the **identical** `PipelineConfig`: `n_clusters=50,
   cluster_seed=42, msm_lagtime=20` — to full precision, every time.
2. Was run through the identical deterministic tool on the identical
   trajectory, producing the **identical** `PipelineResult` to full float
   precision: `relaxation_rate_mean=0.0121142034...`,
   `vamp2_score=2.449146`, `macrostate_populations=[0.445554, 0.554446]`.
3. Was validated identically: `two_states_recovered=True`,
   `rate_matches_analytical=True`, `is_ill_posed=False`, mechanical
   `verdict=ACCEPT`, `llm_verdict=ACCEPT` (never overridden).
4. Stopped at iteration 1 with `stop_reason=validator_accepted`.

**What did vary:** the natural-language reasoning text. Each run's Optimizer
wrote genuinely different prose explaining the same choice (e.g. run 1:
*"Using a moderate number of microstates (50) to reasonably resolve the
double-well free energy landscape..."*; run 5: *"Start from a well-motivated
baseline. Phase 1/2 findings indicate implied timescales plateau near
msm_lagtime=20..."*). This confirms the model is genuinely re-reasoning on
each call, not returning a cached response — but it converged on the same
structured numbers every single time regardless of phrasing.

Because `run_msm_pipeline` is provably deterministic (`tests/test_tools.py`)
and every run used the identical config on the identical trajectory, the
"accepted rate agrees across runs" result is a **direct, expected consequence
of that already-proven determinism** — not new evidence that the Validator's
gate constrains a genuinely varied search. The interesting version of the
claim (bounded outcome *despite divergent paths*) was never exercised,
because no paths diverged.

## Why this happened — a real diagnosis, not a shrug

Two design choices in this session's own build compound to make the first
proposal essentially the only proposal a well-calibrated model would make:

1. **The Optimizer's prompt hands it the answer.** `agents/optimizer.py`'s
   `SearchBounds.as_prompt_text()` explicitly tells the model *"Phase 1/2
   already established that this system's implied timescale plateaus near
   msm_lagtime=20... treat that as a well-motivated starting region, not a
   value you have to rediscover from scratch."* That's exactly what it was
   designed to do — hand the agent hard-won physics knowledge rather than
   make it rediscover it by trial and error (this session's own explicit
   design goal for `SearchBounds`). But it also means there is essentially
   one obviously-correct answer on iteration 1, and no reason for a
   reasoning model to deviate from it.
2. **Nothing ever forced a second iteration.** The Optimizer's own
   feedback-reacts-to-failure behavior (proven with fakes in
   `tests/test_optimizer.py`) never got exercised in this real study,
   because the first proposal was accepted every time. Diversity had only
   one possible source left: the model's own sampling stochasticity on a
   single, highly-anchored call — and at whatever default sampling settings
   `pydantic-ai`/the Anthropic API used here, that was not enough to move
   the structured output (`cluster_seed=42` in particular is a well-known
   "canonical placeholder" value with no physics grounding in the prompt at
   all, and the model picked it identically 8/8 times).

Neither of these is a bug. Both are consequences of choices made earlier in
this build for good reasons (hand the agent real physics knowledge; let the
Validator's gate, not brute-force search, be where the real judgment lives).
The honest conclusion is that **this specific experimental design cannot
distinguish "the search explores" from "the search is over-constrained"** —
it would show the same 8/8-identical result either way.

## What this study DID demonstrate

- **Zero errors, zero ill-posed configs, zero mechanical/LLM disagreement**
  across 8 independent real runs — the loop's machinery (already proven
  deterministic with fakes) behaves identically under real LLM calls.
- **The Optimizer, with no example numbers handed to it beyond the lagtime
  region, independently reconstructed Phase 1's own validated
  `n_clusters=50`** — a real, if modest, piece of evidence that its
  reasoning is well-calibrated to the problem, not just following a script.
- **The full real pipeline (real API calls, real 15M-step trajectory, real
  MSM/PCCA+) reproduces Phase 1/2's own numbers almost exactly**: measured
  rate 0.012114 vs. analytical 0.012133 (0.15% deviation) — comfortably
  inside the reused ±3.16% band, and visibly tighter than the band itself,
  consistent with everything Phase 1/2 already established about this
  config being well past the "trustworthy" cutoff.

## What it did NOT demonstrate

- That the agentic search explores a genuinely varied space of configs.
- That the Validator's gate is doing constraining work on divergent paths
  (it was never tested against a divergent path in this run).

## Recommended next step (not yet run — needs a decision, and more budget)

To actually test path diversity, at least one of the following needs to
change, deliberately, before spending more API budget:
1. **Raise sampling temperature** on the Optimizer's `Agent` (currently
   using pydantic-ai/Anthropic defaults) to increase the chance of genuine
   proposal variation on a first call.
2. **Weaken the prompt's anchor** — stop handing the Optimizer the
   converged lagtime as directly, forcing it to reason more independently
   (at the cost of the very design goal that motivated handing it that
   knowledge in the first place).
3. **Deliberately force at least one rejection** — e.g. run the study at a
   noticeably tighter rate tolerance than Phase 2's own band, so the
   "obvious" first guess is not guaranteed to pass, and the Optimizer must
   genuinely search and react to a real (not faked) rejection. This
   exercises the exact feedback-loop machinery `tests/test_optimizer.py`
   already proved works with fakes, but under real conditions.

Option 3 is the most direct test of the property this study actually set out
to prove, and doesn't require weakening a design choice (prompt anchoring)
that was made deliberately and for good reason. Not run here — this report
stops at the honest negative finding, per the instruction to report what was
actually observed rather than keep spending budget chasing the result we
expected to see.
