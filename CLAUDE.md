# CLAUDE.md: Project Constitution

## What this project is
An autonomous multi-agent system that runs a Markov State Model (MSM)
pipeline on trajectories from a stochastic double well, verifies the
recovered physics against exactly computed answers, and reports
uncertainty. The benchmark physics is textbook and checkable.

Phase 1's engine is the 0-D double well dx = -V'(x)dt + sqrt(2/beta)dW,
whose relaxation rate and basin populations can be computed exactly
(physics/known_answers.py). The 2D stochastic Allen-Cahn field is Phase 4,
validated qualitatively against the 0-D reference. Moiré materials are the
motivating application, not a modeled system.

## Core principle: NOTHING IS A BLACK BOX
The human author is learning. Every function you write MUST have:
- A docstring explaining what it does in plain English.
- Named intermediate variables (no dense one-liners).
- A comment on any line doing non-obvious math.
If a simpler-but-longer version exists, write the longer version.

## Style (documentation, docstrings, comments)
- Short sentences in plain English, one idea each.
- No em dashes, and no " -- " used as a dash. Use a period, comma, colon or
  parentheses instead.
- Claim only what the code, a test, or a log in the repo supports. Say
  "exact", "proves" or "verified" only when literally true.
- No dated history notes in code or docs; history belongs in the
  PROJECT_STATE.md session log.

## HARD BOUNDARIES (never violate)
1. NEVER install a package not on the approved list below without asking.
   Approved: numpy, scipy, matplotlib, py-pde, deeptime, scikit-learn,
   pydantic, pydantic-ai, h5py, tqdm, pytest. Also ax-prover (optional,
   Phase 3.5 only, for Lean proofs; human-authorized).
2. NEVER change physics parameters (barrier height, temperature, grid
   size) on your own. These are the human's decisions. Ask.
3. NEVER add a new agent, tool, or pipeline stage that isn't in the
   current phase's task list. Ask first. (Phase 3's three-agent
   architecture is part of its task list.)
4. NEVER write a function longer than ~40 lines. Split it.
5. NEVER silently catch an exception. Every except block must log the
   full error. No bare `except:`.
6. NEVER use a real LLM API call inside a fast-running test. Tests use
   hardcoded fake responses.
7. If a known-answer check fails, STOP and report. Do not "fix" it by
   loosening the check.

## WORKFLOW RULES
- One module per request. Do not build ahead.
- After writing any module, write its known-answer test in the same turn
  and run it. Report the result.
- Before editing an existing file, show me the 3-line summary of what you
  will change and wait for confirmation.
- At the end of every session, update PROJECT_STATE.md (see below).

## FILE YOU MUST MAINTAIN: PROJECT_STATE.md
After each session, append a dated entry with: what was built, what
passed its check, current known bugs, and the single next task.
At the START of each session, read PROJECT_STATE.md first and confirm
your understanding before doing anything. The log before 2026-10-06 is
archived in docs/HISTORY.md.

## PHYSICS GROUND TRUTH (the checks that define "correct")
Phase 1 (0-D double well, primary benchmark):
- V(x) = A(x²−1)² [+ b·x if tilted] has exactly two minima for
  |b| < 8A/(3√3). The MSM must show two metastable states: one slow
  process, with t2/t3 above pipeline.msm.MIN_TIMESCALE_SEPARATION.
- Relaxation rate: the reference is the exact rate λ2
  (exact_relaxation_rate_0d) and, for simulated data, the exact rate of
  the Euler-Maruyama chain at the simulation's dt
  (euler_maruyama_relaxation_rate_0d; +1.2% at dt = 0.01). Twice the
  one-way Eyring-Kramers rate sqrt(V''(x0)|V''(xs)|)/(2π)·exp(−βΔV) is
  λ2's β→∞ limit (7-11% high at β = 3-7): a cross-check, not the oracle.
- Basin population ratio: P+/P− is the ratio of the integrals of
  exp(−βV) over the two basins (boltzmann_population_ratio). It is exactly
  1 for b = 0. exp(−βΔV) of the minima alone misses the well widths.
- Detailed balance holds at equilibrium. The reversible MSM estimator
  imposes it, so it is an assumption here, not a check.
If any of these fail, there is a BUG. Never present a failing result as
correct.

Phase 4 (2D stochastic Allen-Cahn field, deployment target):
- Validated qualitatively against the Phase 1 reference, not against a
  2D analytical rate: the Eyring-Kramers prefactor is not known
  analytically in 2D (Rolland, Bouchet & Simonnet, arXiv:1507.05577,
  §3.2.1).
- Energies scale with area: the coherent-flip barrier is about L²·A, so
  the field behaves like the 0-D system at β_eff = β·L². 0-D formulas and
  the paper's 1-D thresholds do not transfer directly.
- Unit conversion to Rolland-Bouchet: with A = 1, γ = 1, L_theirs =
  2·L_ours. Time and temperature also rescale; derive the full conversion
  before using any of their thresholds.
- The 2D continuum SPDE with additive white noise is ill-defined without
  renormalization (same §3.2.1). The grid is part of the model: every
  Phase 4 result needs a grid-refinement check (same L, 32×32 vs 64×64)
  and must be reported with its grid resolution.

## TECH STACK NOTES
- deeptime is the MSM library (successor to PyEMMA). Use
  deeptime.clustering.KMeans and deeptime.markov.msm.MaximumLikelihoodMSM
  and BayesianMSM. BayesianMSM with count_mode="effective" is
  overconfident for this system (Phase 2 measured 2/42 coverage).
- py-pde handles the stochastic PDE. There is no `NoiseTerm` class; add
  thermal noise via the `noise=` argument of `pde.PDE`, which takes a
  VARIANCE (pass `2*gamma/beta`, the square of the prefactor). No py-pde
  solver supports adaptive time-stepping on a noisy PDE; use a fixed-step
  solver (e.g. `solver="euler"`).
- pydantic-ai handles agents. Every agent output is a Pydantic model.
- Anthropic model string for agents: anthropic:claude-sonnet-5 (the
  "anthropic:" prefix is required by pydantic-ai's infer_model()).
- Phase 3 is a three-agent architecture mirroring Ax-Prover's
  Orchestrator / Prover / Verifier (Axiomatic AI, arXiv:2510.12787, Breen
  et al., §3.1):
  - The Orchestrator routes: task assignment, feedback, the stop decision.
    `agents/loop.py` stays thin (builds the agents, hands control to the
    Orchestrator, writes the JSON ledger).
  - The Optimizer (≙ Prover) proposes PipelineConfigs; the Orchestrator
    runs each through the deterministic `run_msm_pipeline` tool.
  - The Validator (≙ Verifier) computes the physics checks in Python before
    the LLM is called: timescale separation, and the rate within
    3 replica sigmas (Phase 1's 6-replica spread at β = 5) of the exact
    chain rate.
    It reports ill-posed configs (Ax-Prover Appendix C) separately from
    well-posed configs that fail physics. The verdict is recomputed in the
    schema, so the LLM cannot override it.
  - One deliberate departure from Ax-Prover: there the Prover and Verifier
    share one tool (Lean); here the Optimizer's guide (VAMP-2, comparable
    only at equal lag) and the Validator's oracle (exact physics) are
    different checks.

## Lean / ax-prover scope boundaries (Phase 3.5)

IN SCOPE: formally verifying the *algebraic landscape facts* that
physics/known_answers.py relies on: barrier height, critical-point
locations at b=0, second derivatives (8A at wells, -4A at saddle), and the
b=0 symmetry giving ΔF=0.

OUT OF SCOPE (do not attempt, do not scaffold toward):
- Proving the Eyring-Kramers rate formula itself (Fokker-Planck spectral
  asymptotics; not in Mathlib).
- Any theorem about the MSM estimator's statistical convergence. Not a
  theorem; a category error.
- 2D Allen-Cahn / Rolland-Bouchet Eq. 13 (infinite-dim Hessian dets).
- Exact well positions for b != 0: those are irrational cubic roots,
  which is *why* known_answers.py uses brentq. For the tilted case the only
  honest Lean statement is qualitative: sign(ΔF) vs sign(b), and ΔF=0 iff b=0.

Claude Code writes theorem STATEMENTS ending in `sorry`. It does not write
proof bodies. ax-prover writes proofs. One human-authorized exception:
`V_hasDerivAt`'s proof body was written by Lemma (an external tool) at the
human's explicit direction, outside the ax-prover workflow; its statement
is unchanged. Claude Code still writes no proof bodies.

## ARCHITECTURE

multiagent_for2Dphysics/
├── CLAUDE.md                  # this constitution
├── PROJECT_STATE.md           # current state + session log
├── README.md                  # project overview and results
├── requirements.txt           # pinned package versions
├── physics/                   # the environment (generates data)
│   ├── potential.py           # V and dV/dx (shared, 0-D + 2D)
│   ├── simulate_0d.py         # 0-D SDE integrator, Phase 1 engine
│   ├── simulate.py            # 2D stochastic Allen-Cahn integrator, Phase 4
│   └── known_answers.py       # exact reference values
├── pipeline/                  # the analysis (data -> MSM)
│   ├── features.py            # trajectory -> feature vectors
│   ├── cluster.py             # k-means microstates
│   ├── msm.py                 # MSM, lag choice, CK and separation checks
│   └── uq.py                  # BayesianMSM credible intervals
├── agents/                    # three-agent loop (see TECH STACK NOTES)
│   ├── schemas.py             # Pydantic contracts
│   ├── tools.py               # run_msm_pipeline (deterministic)
│   ├── optimizer.py           # Optimizer (≙ Prover)
│   ├── validator.py           # Validator (≙ Verifier)
│   ├── orchestrator.py        # routing and stop decision
│   └── loop.py                # thin entry point, writes the ledger
├── lean/                      # Lean 4 + Mathlib formalization (Phase 3.5)
│   └── Oracle/Potential.lean  # the 7 statements about V
├── tests/                     # known-answer tests, one file per module
├── scripts/                   # runnable phase entry points
├── results/                   # plots, raw data, ledgers, check logs
├── presentation/              # slide deck (built from deck_template.html)
├── docs/HISTORY.md            # archived session log
└── archive/                   # superseded artifacts, kept for the record
