# CLAUDE.md — Project Constitution

## What this project is
An autonomous multi-agent system that runs a Markov State Model (MSM)
pipeline on trajectories from a stochastic double-well system, verifies
the recovered physics against KNOWN ANALYTICAL ANSWERS, and reports
uncertainty. The benchmark physics is textbook and checkable.

**[2026-07-11] Phase 1/Phase 4 pivot** (see PROJECT_STATE.md §9 for full
reasoning): Phase 1's verified engine is now the 0-D stochastic double well
dx = -V'(x)dt + sqrt(2/beta)dW, where BOTH the Eyring-Kramers rate and the
Boltzmann well-population ratio are exact, closed-form, and observable —
cleaner and unambiguous, unlike the 2D field where the Eyring-Kramers
prefactor is not analytically known even in 2D (Rolland, Bouchet &
Simonnet 2015, arXiv:1507.05577, §3.2.1). The stochastic 2D Allen-Cahn
field moves to Phase 4 as the "interesting deployment": run at small L (a
few interface widths) so it switches coherently, validated qualitatively
against the 0-D reference rather than staked on a 2D analytical rate.
Moiré materials remain the motivating target application, now folded into
this Phase 4 deployment (tilt b) rather than a separate late demo.

## Core principle: NOTHING IS A BLACK BOX
The human author is learning. Every function you write MUST have:
- A docstring explaining what it does in plain English.
- Named intermediate variables (no dense one-liners).
- A comment on any line doing non-obvious math.
If a simpler-but-longer version exists, write the longer version.

## HARD BOUNDARIES (never violate)
1. NEVER install a package not on the approved list below without asking.
   Approved: numpy, scipy, matplotlib, py-pde, deeptime, scikit-learn,
   pydantic, pydantic-ai, h5py, tqdm, pytest.
2. NEVER change physics parameters (barrier height, temperature, grid
   size) on your own. These are the human's decisions. Ask.
3. NEVER add a new agent, tool, or pipeline stage that isn't in the
   current phase's task list. Ask first.
   **[2026-07-11] Phase 3's task list was updated to a three-agent
   architecture (Optimizer, Validator, Orchestrator) before any code
   referencing `agents/orchestrator.py` was written — see PROJECT_STATE.md
   §7/§9. This is pre-authorized, not scope creep to flag.**
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
your understanding before doing anything.

## PHYSICS GROUND TRUTH (the checks that define "correct")
Phase 1 (0-D double well, PRIMARY benchmark):
- V(x) = A(x²−1)² [+ b·x if tilted] has exactly TWO minima. The MSM must
  recover exactly two dominant macrostates.
- Eyring-Kramers rate (exact, closed-form, incl. prefactor):
  T = (2π/|λs|)·sqrt(|V''(xs)|/V''(x0))·exp(β(V(xs)−V(x0))) — see
  physics/known_answers.py. A log(rate) vs β plot MUST be a straight line
  of slope −ΔV.
- Boltzmann well-population ratio (exact): P(x_+)/P(x_-) = exp(−βΔF),
  ΔF = V(x_+)−V(x_-). For the symmetric well (b=0), ΔF=0 → ratio 1
  (symmetric populations). For the tilted well, ΔF≈2b (see
  physics/known_answers.py for the exact root-found value).
- At equilibrium the system is time-reversible: detailed balance holds.
If any of these fail, there is a BUG. Never present a failing result as
correct.

Phase 4 (2D stochastic Allen-Cahn field, deployment target):
- Same double well, now spatially extended. Validated QUALITATIVELY
  against the Phase 1 0-D reference (same MSM pipeline, same known-answer
  gates), not staked on a 2D analytical rate — the Eyring-Kramers prefactor
  is not known analytically in 2D even in the literature (see
  PROJECT_STATE.md §9, citing Rolland-Bouchet arXiv:1507.05577 §3.2.1).
- Unit conversion to the Rolland-Bouchet convention is NOT 1:1: with our
  A=1, gamma=1, their L equals 2x ours (L_theirs = 2*L_ours) — verified via
  both front-width and bifurcation-point matching. Always convert their
  quoted (β,L) thresholds through this factor before using them; see
  PROJECT_STATE.md §9 for the derivation and the corrected formulas.

## TECH STACK NOTES
- deeptime is the MSM library (successor to PyEMMA). Use
  deeptime.decomposition.TICA / VAMP, deeptime.clustering.KMeans,
  deeptime.markov.msm.MaximumLikelihoodMSM and BayesianMSM.
- py-pde handles the stochastic PDE. There is no `NoiseTerm` class; add thermal
  noise via the `noise=` argument of `pde.PDE`, which takes a VARIANCE (so pass
  `2*gamma/beta`, the square of the physical prefactor √(2γ/β)). No py-pde solver
  supports adaptive time-stepping on a noisy PDE — use a fixed-step solver
  (e.g. `solver="euler"`) for any stochastic run. See PROJECT_STATE.md §4/§8.
- pydantic-ai handles agents. Every agent output is a Pydantic model.
- Anthropic model string for agents: anthropic:claude-sonnet-5 (the
  "anthropic:" provider prefix is required by pydantic-ai's infer_model();
  a bare "claude-sonnet-5" raises "Unknown model"). **[2026-07-12]
  Corrected from the stale "claude-sonnet-4-6" — see PROJECT_STATE.md §9
  for the resolution of the standing "verify current model string"
  reminder.**
- **Phase 3 is a three-agent architecture — Orchestrator / Optimizer / Validator
  — mirroring Ax-Prover's Orchestrator/Prover/Verifier separation (Axiomatic AI,
  arXiv:2510.12787, Koppens et al., §3.1).** The Orchestrator is a REAL
  component (task assignment, feedback routing, owns the stop decision — Ax-Prover
  §3.1.1), not loop plumbing folded into `agents/loop.py`'s while-statement;
  `agents/loop.py` is deliberately thin (instantiates the three agents, hands
  control to the Orchestrator, writes the JSON ledger). The Optimizer (≙ Prover)
  proposes PipelineConfigs and calls the deterministic `run_msm_pipeline` tool
  (≙ Ax-Prover's Lean tool calls). The Validator (≙ Verifier) is the independent
  gatekeeper, grounded in `physics/known_answers.py`'s hardcoded Boolean physics
  checks, and additionally does ill-posedness detection (Ax-Prover Appendix C) —
  a config can be well-posed-but-wrong or ill-posed, and these must be reported
  distinctly, not conflated. One deliberate departure from Ax-Prover, stated
  explicitly rather than papered over: their Prover and Verifier share ONE tool
  (Lean) and the Verifier's value is independence of judgment; here the
  Optimizer's tool (VAMP-2, a statistical model-quality score) and the
  Validator's oracle (independent analytical physics: Kramers rate, Boltzmann
  ratio, two-state recovery) are genuinely different checks — a stronger
  verification setup than the one being borrowed from, not a weaker copy of it.
  Full reasoning: PROJECT_STATE.md §9 (2026-07-11 entry).

## ARCHITECTURE

moire-msm-engine/
├── CLAUDE.md                  # the constitution (already created)
├── PROJECT_STATE.md           # session log — author and Claude maintain this
├── README.md                  # one-paragraph project description
├── requirements.txt           # pinned package versions
│
├── physics/                   # THE ENVIRONMENT (generates data)
│   ├── __init__.py
│   ├── potential.py           # V(φ) and its derivative (shared, 0-D + 2D)
│   ├── simulate_0d.py         # 0-D SDE integrator -- Phase 1 primary engine
│   ├── simulate.py            # stochastic Allen-Cahn integrator -- Phase 4
│   └── known_answers.py       # analytical values for verification
│
├── pipeline/                  # THE ANALYSIS (data → MSM)
│   ├── __init__.py
│   ├── features.py            # field snapshots → feature vectors
│   ├── reduce.py              # TICA dimensionality reduction
│   ├── cluster.py             # k-means microstates
│   ├── msm.py                 # build MSM, extract timescales
│   └── uq.py                  # BayesianMSM confidence intervals
│
├── agents/                    # THREE-AGENT ARCHITECTURE (Ax-Prover pattern,
│   │                          # arXiv:2510.12787 -- see TECH STACK NOTES above)
│   ├── __init__.py
│   ├── schemas.py             # Pydantic models = the contracts
│   ├── tools.py               # deterministic functions agents call (run_msm_pipeline)
│   ├── optimizer.py           # Optimizer Agent (≙ Prover)
│   ├── validator.py           # Validator Agent (≙ Verifier) + ill-posedness checks
│   ├── orchestrator.py        # Orchestrator Agent: task assignment, feedback
│   │                          # routing, stop decision -- a real component
│   └── loop.py                # THIN: instantiates the three agents, hands
│                               # control to the Orchestrator, writes the JSON ledger
│
├── tests/                     # KNOWN-ANSWER CHECKS
│   ├── test_potential.py
│   ├── test_simulate_0d.py
│   ├── test_simulate.py
│   ├── test_msm_recovers_two_states.py
│   ├── test_arrhenius.py
│   └── test_agents_with_fake_llm.py
│
├── data/                      # trajectories (gitignored, streamed here)
│   └── .gitkeep
│
├── results/                   # plots, ledgers, final outputs
│   └── .gitkeep
│
└── scripts/                   # top-level runnable entry points
    ├── run_phase1_benchmark.py
    ├── run_phase2_uq.py
    ├── run_phase3_agentic.py
    └── run_phase4_moire_demo.py

