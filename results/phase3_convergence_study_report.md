# Phase 3 study: results, re-scored against the exact rate

**In short:** AI agents searched for good analysis settings, and a separate
checker accepted or rejected each one. The checker originally compared
against a textbook formula that is 8% off at this temperature, so its
verdicts were misleading. Re-checked against the exact answer, every
setting the agents tried falls inside the checker's ±5.6% band, so that
absolute check cannot tell good settings from slightly biased ones. A
check that the lag has converged would.

**Runs:** 4 real runs of the agentic loop, 20 iterations in total, July 2026.
**Model:** `anthropic:claude-sonnet-5` (Optimizer and Validator).
**Reference:** beta = 5, one fixed 15,000,000-step trajectory (seed 7) for all runs.
**Raw data:** `results/phase3_convergence_study/run_01..04_ledger.json`;
re-score in `results/phase3_rescore.json` and `results/phase3_rescore.png`.
**Earlier version:** `archive/results/phase3_convergence_study_v1_prompt_anchored/`
(the Optimizer was given the answer in its prompt; superseded).

## What was run

The Optimizer proposed (n_clusters, msm_lagtime) pairs; the Validator
accepted a config when its rate fell within ±3.16% of 2 x Eyring-Kramers.
That reference turned out to be 8% above the exact rate of the simulated
chain at beta = 5 (see `physics/known_answers.py`), so the original
verdicts are re-scored below.

| Run | Iterations | Path (n_clusters, lag) | Accepted | Accepted rate / exact |
|---|---|---|---|---|
| 1 | 6 | (50,100), (60,500), (60,1000), (60,50), (80,200), **(60,20)** | (60, 20) | 1.055 |
| 2 | 4 | (50,200), (75,1000), (60,400), **(50,50)** | (50, 50) | 1.050 |
| 3 | 5 | (50,200), (75,1000), (75,50), (75,500), **(75,20)** | (75, 20) | 1.048 |
| 4 | 5 | (50,100), (50,1000), (100,50), (100,300), **(100,10)** | (100, 10) | 1.047 |

"Exact" here is the exact rate of the simulated chain at dt = 0.01
(`euler_maruyama_relaxation_rate_0d`).

## Re-score (no LLM calls)

`scripts/rescore_phase3_ledgers.py` re-runs the deterministic pipeline on
all 17 distinct configs (every recorded rate is reproduced exactly) and
applies the current Validator checks: one slow process (t2/t3 above 10)
and the rate within 3 replica sigmas of the exact chain rate. Phase 1's
replica spread at beta = 5 is 1.87%, so the band is ±5.6%. That spread
comes from only 6 replicas; the trend over all beta suggests about 3.4%,
which would make the band near ±10%. Either way every config passes.

| Lag (frames) | Rate / exact chain rate | Original verdict | Current verdict |
|---|---|---|---|
| 10-20 | 1.047-1.055 | accepted (3 of 3) | accepted |
| 50 | 1.032-1.050 | 1 of 4 accepted | accepted |
| 100-400 | 1.005-1.032 | rejected | accepted |
| 500-1000 | 0.993-1.002 | rejected | accepted |

All 17 configs are accepted, with t2/t3 between 49 and 295.

## What the study shows

- **The search varies between runs.** Runs 2 and 3 share their first two
  proposals and runs 1 and 4 their first, then all four diverge, ending on
  four different configs.
- **The machinery works.** The verdict is computed from the physics checks,
  never from the LLM; no config was ill-posed (0 of 20).
- **The physics gate did not discriminate.** The original rejections came
  from the off-center reference: it rejected the converged configs
  (lags 500-1000, within 0.7% of the exact rate) and accepted short-lag
  ones 4.7-5.5% high. With the exact reference, every config passes: the
  band is wider than the ~5% short-lag bias. Within this one trajectory
  the rate's fall with lag is clearly resolved, so a lag-convergence check
  would catch it.
- **The Optimizer's key reasoning step was an artifact.** In run 1 it moved
  to short lags because VAMP-2 kept decreasing as the lag grew. VAMP-2
  falls with the lag for any model, so this was not evidence; the
  Validator's suggestion at that point (longer lags) was the right one. The
  Optimizer's prompt now says VAMP-2 is only comparable at equal lag.
- **The rate converges from above.** Across configs the measured rate falls
  toward the exact value as the lag grows, as the variational principle
  predicts for MSM implied timescales.

## Next step

Make the Validator also require a converged lag (at least 0.1 of the
slowest timescale, as Phase 1 does) and estimate its band from all
temperatures. Then run the agents again.

## Cost notes

Run 1 was a dry run reused in the study (same deterministic trajectory);
only 3 further runs were paid for. Each iteration is one clustering and MSM
pass plus 2 API calls.
