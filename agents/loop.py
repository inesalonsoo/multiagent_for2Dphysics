"""
Entry point for one run of the agentic loop. Deliberately thin: it builds
the fixed inputs (a reference trajectory, the search space, the rate
tolerance), creates the two LLM agents, hands control to
agents/orchestrator.py, and saves the result as a JSON ledger (a complete
record of every proposal and verdict). The decisions themselves live in
optimizer.py, validator.py and orchestrator.py.
"""

from pathlib import Path
from typing import Any, Optional

from agents.optimizer import SearchBounds, build_optimizer_agent
from agents.orchestrator import MAX_ITERATIONS, run_agentic_loop_with_real_agents
from agents.schemas import AgenticRun
from agents.validator import REFERENCE_BETA, build_validator_agent, load_rate_tolerance
from physics.simulate_0d import run_trajectory_0d

DT = 0.01  # same time step as Phase 1
# Must equal scripts.run_phase1_benchmark.N_STEPS: the Validator's rate
# tolerance is the replica spread of trajectories of exactly this length.
N_STEPS = 15_000_000
MAX_N_CLUSTERS = 100


def build_reference_context(seed: int = 7):
    """
    Build the fixed inputs of a run: the reference trajectory at
    REFERENCE_BETA, its search bounds, and the Validator's rate tolerance.

    Kept separate from run_one_real_loop() so a multi-run study can build
    them once and give every run the same trajectory. Then any difference
    between runs comes from the agents, not from simulation noise.

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
    Run one complete loop through the Orchestrator.

    By default the agents are the real LLM-backed ones, so this makes real
    API calls. Tests pass scripted fake agents instead (tests/test_loop.py).
    """
    optimizer_agent = optimizer_agent or build_optimizer_agent()
    validator_agent = validator_agent or build_validator_agent()

    return run_agentic_loop_with_real_agents(
        optimizer_agent, validator_agent, trajectory, DT,
        search_bounds, rate_tolerance, REFERENCE_BETA, max_iterations,
    )


def main():
    """One real run, saved to results/ledger.json. For the multi-run study
    see scripts/run_phase3_agentic.py, which reuses these functions."""
    trajectory, search_bounds, rate_tolerance = build_reference_context()
    run = run_one_real_loop(trajectory, search_bounds, rate_tolerance)

    # UTF-8 explicitly: Windows' default encoding cannot write some characters
    # that appear in LLM text (such as the "almost equal" sign).
    Path("results/ledger.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote results/ledger.json -- {len(run.entries)} iterations, stop_reason={run.stop_reason}")


if __name__ == "__main__":
    main()
