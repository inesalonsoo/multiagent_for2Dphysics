"""
Tests for agents/loop.py, which only wires things together: the reference
inputs are checked directly, and one run is smoke-tested with fake agents
(no API calls). Routing itself is tested in tests/test_orchestrator.py.
"""

import os

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agents.loop import MAX_N_CLUSTERS, build_reference_context, run_one_real_loop
from agents.optimizer import build_optimizer_agent
from agents.validator import build_validator_agent

_REQUIRES_PHASE2_CACHE = pytest.mark.skipif(
    not os.path.exists("results/arrhenius_sweep_raw.npz"),
    reason="requires results/arrhenius_sweep_raw.npz (run scripts.run_phase1_benchmark)",
)


@_REQUIRES_PHASE2_CACHE
def test_build_reference_context_returns_a_usable_trajectory_and_tolerance():
    trajectory, search_bounds, rate_tolerance = build_reference_context(seed=7)

    assert len(trajectory) == search_bounds.trajectory_length_frames
    assert search_bounds.max_n_clusters == MAX_N_CLUSTERS
    assert 0.0 < rate_tolerance < 0.20


@_REQUIRES_PHASE2_CACHE
def test_build_reference_context_is_deterministic_given_the_same_seed():
    first_trajectory, _, first_tolerance = build_reference_context(seed=7)
    second_trajectory, _, second_tolerance = build_reference_context(seed=7)

    assert (first_trajectory == second_trajectory).all()
    assert first_tolerance == second_tolerance


@_REQUIRES_PHASE2_CACHE
def test_run_one_real_loop_wires_fake_agents_together_correctly():
    """Smoke test of the wiring only, with fake agents (no API calls) on the
    real reference trajectory."""
    trajectory, search_bounds, rate_tolerance = build_reference_context(seed=7)

    def optimizer_fake_llm(messages, info):
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            config=dict(n_clusters=50, cluster_seed=42, msm_lagtime=1000),
            reasoning="A converged lag, about 0.1 of the slowest timescale.",
        ))])

    def validator_fake_llm(messages, info):
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            llm_verdict="ACCEPT", reasoning="Checks look consistent.", suggested_change=None,
        ))])

    optimizer_agent = build_optimizer_agent(model=FunctionModel(optimizer_fake_llm))
    validator_agent = build_validator_agent(model=FunctionModel(validator_fake_llm))

    run = run_one_real_loop(
        trajectory, search_bounds, rate_tolerance, max_iterations=3,
        optimizer_agent=optimizer_agent, validator_agent=validator_agent,
    )

    assert len(run.entries) == 1
    assert run.entries[0].result.error is None
    assert run.stop_reason == "validator_accepted"
