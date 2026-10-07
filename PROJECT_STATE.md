# PROJECT_STATE.md

Current state of the project. The full session-by-session log up to
2026-10-06 is archived unchanged in [docs/HISTORY.md](docs/HISTORY.md).
Append one dated entry per session to the Session log at the end.

## 1. What the project is

We simulate a particle hopping between the two valleys of a double-well
landscape and measure how often it hops, using a Markov State Model (MSM).
Because the true rate can be computed exactly, every result is checked
against it. Three agents (Orchestrator, Optimizer, Validator) run the
analysis under that check; only the Optimizer and the Validator's
commentary use an LLM. A Lean 4 file states 7 facts about the landscape
(2 proved). A 2D version (Phase 4) is planned, not started.

## 2. Ground truth used for checking

System: dx = -V'(x) dt + sqrt(2/beta) dW, V(x) = A(x^2 - 1)^2 + b x.
All in `physics/known_answers.py`; nothing is fitted to simulation data.

| Quantity | Function | Method |
|---|---|---|
| Relaxation rate lambda_2 (what an MSM measures) | `exact_relaxation_rate_0d` | eigenvalue of the discretized generator, beta <= 15 |
| Rate of the simulated chain at time step dt | `euler_maruyama_relaxation_rate_0d` | eigenvalue of the discretized transition kernel; +1.2% above lambda_2 at dt = 0.01 |
| Basin population ratio | `boltzmann_population_ratio` | quadrature of exp(-beta V), split at the barrier |
| Large-beta limit of the one-way rate | `eyring_kramers_rate_0d` | closed form; twice it is 7-11% above lambda_2 at beta = 3-7 |

Detailed balance is a modeling assumption: the MSM estimator imposes it, so
it is not tested.

## 3. Parameters (human decisions; do not change without asking)

Phases 1-3: A = 1, b = 0, dt = 0.01, 15,000,000 steps per trajectory,
6 replicas per beta, 50 k-means microstates, beta = 3-10, reference
beta = 5 (Phase 3, trajectory seed 7), agent model `anthropic:claude-sonnet-5`,
loop cap 15 iterations.

Phase 4 (recorded values, under review before any run): gamma = 1, 32x32
periodic grid, L = 2.5, beta = 20-30, b = 0.1, spatial-mean feature.
Why under review: the field's barrier scales with area, so the coherent-flip
barrier is beta L^2 A = 125-190 kT at these values and no switching would be
seen. The Rolland-Bouchet thresholds that motivated them are for a 1-D field.
The code's `DOMAIN_SIZE` is still 10. The 2D SPDE also needs renormalization,
so any result needs a grid-refinement check (32x32 vs 64x64).

## 4. Status

**Phase 1 (verified engine): passes.** Per replica, the lag is 0.1 of the
slowest implied timescale (self-consistent), with a Chapman-Kolmogorov check
and a timescale-separation check. Results (`results/arrhenius_sweep_raw.npz`):

| beta | resolved replicas | median lag (frames) | measured / exact chain rate (+- SEM) |
|---|---|---|---|
| 3 | 6/6 | 122 | 1.003 +- 0.006 |
| 4 | 6/6 | 334 | 0.991 +- 0.006 |
| 5 | 6/6 | 878 | 1.013 +- 0.008 |
| 6 | 6/6 | 2468 | 0.975 +- 0.025 |
| 7 | 6/6 | 6064 | 1.031 +- 0.022 |
| 8 | 6/6 | 18388 | 0.989 +- 0.079 |
| 9 | 6/6 | 42100 | 1.145 +- 0.155 |
| 10 | 0/6 | trajectory too short | not gated |

Two metastable states in 42/42 resolved replicas (t2/t3 >= 40.8 at the
chosen lag, a lower bound; a single-well control gives below 2;
Chapman-Kolmogorov error <= 0.029). At every resolved beta the mean is
consistent with the exact chain rate (not rejected at the 1% level); the
99% half-widths are 2-3% at beta 3-5 and tens of percent at beta 8-9.

**Phase 2 (uncertainty): negative result.** The BayesianMSM 90% intervals
contain the exact rate in 2 of 42 replicas. They are 11-193 times narrower
than the replica scatter, because deeptime's "effective" counts are only
about 20% below the correlated sliding counts. The project's uncertainty is
the replica spread.

**Phase 3 (agentic loop): machinery demonstrated; the rate band is too
loose to catch short-lag bias.** The recorded study (4 real runs,
20 iterations, July 2026) was judged against 2 x Eyring-Kramers, 8% too
high. Re-scored without LLM calls (`results/phase3_rescore.json`): the
tolerance is +-5.6% (3 x the spread of 6 replicas at beta = 5; the trend
over all beta suggests a spread near 3.4%, so a band near +-10%), and all
17 distinct
configs are accepted, at 0.993-1.055 of the exact chain rate. The original
gate accepted the 4 most biased configs (lags 10-50, 4.7-5.5% high) and
rejected converged ones. Within the one trajectory the rate's fall with
lag is clearly resolved, but the absolute band is too loose to reject a
~5% short-lag bias.

**Phase 3.5 (Lean oracle):** 2 of 7 statements proved in
`lean/Oracle/Potential.lean` (`potential_even_at_b0` by ax-prover,
`V_hasDerivAt` by Lemma at the human's direction); 5 open. Checked on the
pinned toolchain (v4.33.0-rc1, Mathlib 9d302fc): `#print axioms` shows only
the three standard axioms for both proofs
(`results/lean_v_hasDerivAt_check.log`). ax-prover 0.2.0 is installed; its
elaborator-based target selection was verified to queue exactly the 5 open
theorems. No paid 0.2.0 run yet.

**Phase 4 (2D field):** not started; parameters under review (section 3).

## 5. Known limitations

- Bayesian intervals are overconfident for this system (Phase 2).
- The Phase 3 Validator's absolute band cannot reject a ~5% short-lag bias,
  and its width rests on 6 replicas.
- Phase 1 takes about 5 hours: the Bayesian intervals are slow at long lags
  (about 28 minutes per replica at beta = 9).
- Detailed balance is imposed, not tested.
- Only 2 of 7 Lean statements are proved. They back the Eyring-Kramers
  cross-check and the V' formula, not the exact-rate oracles.

## 6. Next tasks

1. Lean: run ax-prover 0.2.0 on `barrier_height_eq`, then `V'_hasDerivAt`
   (DEBUG logging, low iteration cap, human-approved budget).
2. Phase 3: make the Validator require a converged lag (at least 0.1 of
   the slowest timescale, as Phase 1 does) and estimate its band from all
   beta; then fresh agent runs.
3. Phase 2: try deeptime's "sliding-effective" counts and re-measure coverage.
4. Phase 4: the human chooses parameters with the area scaling in mind.
5. A real detailed-balance test (non-reversible estimate plus a statistic).

## Session log

- **[2026-10-06] Science review and fixes.** A review found the asymptotic
  Eyring-Kramers rate used as an exact oracle (6-10% high at beta = 3-7).
  Built: exact rate oracles, a self-consistent lag rule, Chapman-Kolmogorov
  and timescale-separation checks, a Phase 1 gate against the exact chain
  rate, a Phase 2 coverage report, a re-score of the Phase 3 study, a Lean
  consistency test that reads the .lean file. Passed: Phase 1 gates (beta
  3-9), 112 tests (a flaky UQ test, which compared two random posterior
  draws, now reads both intervals from one draw), Lean check. Found:
  overconfident Bayesian intervals; a
  Phase 3 rate band too loose to catch short-lag bias. Docs rewritten;
  the old log moved to docs/HISTORY.md. Next task: Lean run on
  `barrier_height_eq` with ax-prover 0.2.0.
