"""
agents/loop.py (PROJECT_STATE.md Sec 6/7, module 3.6) -- deliberately
THIN. Builds the fixed reference context (trajectory, search bounds, rate
tolerance) a real loop run needs, instantiates the two real LLM-backed
agents, and hands control to agents.orchestrator.
run_agentic_loop_with_real_agents(); writes the resulting AgenticRun to
disk as the JSON State Ledger. All the actual control-flow logic
(routing, stopping) lives in agents/orchestrator.py (module 3.5); all the
actual judgment lives in agents/optimizer.py and agents/validator.py
(modules 3.3/3.4). This module does none of that -- it only wires the
pieces together and persists the result, matching PROJECT_STATE.md
Sec 6's "now THIN" description.
"""

from pathlib import Path
from typing import Any, Optional

from agents.optimizer import SearchBounds, build_optimizer_agent
from agents.orchestrator import MAX_ITERATIONS, run_agentic_loop_with_real_agents
from agents.schemas import AgenticRun
from agents.validator import REFERENCE_BETA, build_validator_agent, load_rate_tolerance
from physics.simulate_0d import run_trajectory_0d

DT = 0.01  # PROJECT_STATE.md Sec 4, Phase 1's own baseline
# MUST match scripts.run_phase1_benchmark.N_STEPS (15,000,000), not a
# smaller "good enough for a quick test" value: load_rate_tolerance()'s
# statistical component (agents/validator.py) was measured on a
# trajectory of exactly that length. A shorter trajectory here has
# genuinely more statistical noise than that reused tolerance accounts
# for -- reusing Phase 2's number is only valid for the trajectory size
# it was calibrated against. Verified empirically while building this
# module: a 1,500,000-step trajectory at seed=7 measured a rate ~6% off
# analytical, outside the ~3.16% total tolerance, purely from the extra
# sampling noise of the shorter trajectory -- not a physics finding.
N_STEPS = 15_000_000
# [2026-07-12] Reference only -- NOT passed to SearchBounds/the Optimizer's
# prompt anymore. Phase 1's converged lag at REFERENCE_BETA
# (PROJECT_STATE.md Sec 9). Kept here purely so a human (or a post-hoc
# analysis script) can compare what the real search actually found against
# what Phase 1 already established -- handing it to the AGENT directly was
# the design mistake the real convergence study exposed (see
# agents/optimizer.py's module docstring for the full diagnosis).
KNOWN_CONVERGED_LAGTIME_FOR_REFERENCE_ONLY = 20
MAX_N_CLUSTERS = 100


def build_reference_context(seed: int = 7):
    """
    Build the fixed pieces every real loop run needs: a reference
    trajectory at REFERENCE_BETA, its search bounds, and Phase 2's
    already-validated rate tolerance. Deliberately separated from
    run_one_real_loop() below so a caller running MULTIPLE repetitions
    (scripts/run_phase3_agentic.py's convergence study) can build this
    ONCE and reuse the SAME trajectory across every repetition --
    otherwise trajectory-level sampling noise would become a second,
    confounding source of run-to-run variation, muddying the question
    the study actually asks (does the AGENTS' search vary?).

    Returns
    -------
    trajectory : np.ndarray
    search_bounds : agents.optimizer.SearchBounds
    rate_tolerance : float
    """
    trajectory = run_trajectory_0d(n_steps=N_STEPS, seed=seed, beta=REFERENCE_BETA, dt=DT)
    search_bounds = SearchBounds(
        trajectory_length_frames=len(trajectory),
        max_n_clusters=MAX_N_CLUSTERS,
    )
    rate_tolerance = load_rate_tolerance(reference_beta=REFERENCE_BETA)
    return trajectory, search_bounds, rate_tolerance


def run_one_real_loop(
    trajectory, search_bounds: SearchBounds, rate_tolerance: float,
    max_iterations: int = MAX_ITERATIONS,
    optimizer_agent: Optional[Any] = None, validator_agent: Optional[Any] = None,
) -> AgenticRun:
    """
    Run one complete agentic loop via agents.orchestrator.
    run_agentic_loop_with_real_agents(). optimizer_agent/validator_agent
    default to the real, production pydantic-ai Agents (built here, using
    agents/optimizer.py and agents/validator.py's default model string,
    PROJECT_STATE.md Sec 4) -- MAKING REAL API CALLS. Tests pass
    FunctionModel-backed fakes instead (see tests/test_loop.py), so this
    function itself stays testable without touching a real API, even
    though its default, production behavior does.
    """
    optimizer_agent = optimizer_agent or build_optimizer_agent()
    validator_agent = validator_agent or build_validator_agent()

    return run_agentic_loop_with_real_agents(
        optimizer_agent, validator_agent, trajectory, DT,
        search_bounds, rate_tolerance, REFERENCE_BETA, max_iterations,
    )


def main():
    """Single real run, written to results/ledger.json -- this module's
    literal roadmap description (PROJECT_STATE.md Sec 6). For the
    repeated-run convergence-robustness study, see
    scripts/run_phase3_agentic.py instead, which reuses the functions
    above rather than duplicating this wiring."""
    trajectory, search_bounds, rate_tolerance = build_reference_context()
    run = run_one_real_loop(trajectory, search_bounds, rate_tolerance)

    # encoding="utf-8" is required, not optional: Path.write_text() defaults to
    # the OS locale codec (cp1252 on this Windows setup), which cannot encode
    # characters an LLM's own reasoning text may contain (e.g. U+2248 "almost
    # equal to" -- caught when the real convergence study crashed on exactly
    # this mid-run; see PROJECT_STATE.md Sec 9).
    Path("results/ledger.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote results/ledger.json -- {len(run.entries)} iterations, stop_reason={run.stop_reason}")


if __name__ == "__main__":
    main()
