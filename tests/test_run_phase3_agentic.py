"""
Tests for scripts/run_phase3_agentic.py, with fake agents in a temporary
directory. The key property is resumability: if a paid study crashes
partway, finished runs must not be paid for again.
"""

import json

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.optimizer import SearchBounds, build_optimizer_agent
from agents.schemas import AgenticRun
from agents.validator import build_validator_agent
from physics.simulate_0d import run_trajectory_0d
from scripts.run_phase3_agentic import run_all_repetitions

_DT = 0.01
_TRAJECTORY = run_trajectory_0d(n_steps=750_000, seed=7, beta=5.0, dt=_DT)
_SEARCH_BOUNDS = SearchBounds(trajectory_length_frames=len(_TRAJECTORY), max_n_clusters=100)
_LOOSE_TOLERANCE = 0.5  # hermetic: no dependency on cached Phase 2 files for this plumbing test


def _accepting_agents(call_counter):
    """Fake Optimizer and Validator that always propose and accept the same
    config, counting Optimizer calls (the signal the test checks)."""
    def optimizer_fake_llm(messages, info: AgentInfo) -> ModelResponse:
        call_counter["n"] += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            config=dict(n_clusters=50, cluster_seed=42, msm_lagtime=20),
            reasoning="Fixed proposal for a resumability test.",
        ))])

    def validator_fake_llm(messages, info: AgentInfo) -> ModelResponse:
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            llm_verdict="ACCEPT", reasoning="Checks look consistent.", suggested_change=None,
        ))])

    optimizer_agent = build_optimizer_agent(model=FunctionModel(optimizer_fake_llm))
    validator_agent = build_validator_agent(model=FunctionModel(validator_fake_llm))
    return optimizer_agent, validator_agent


def test_run_all_repetitions_persists_every_run(tmp_path):
    call_counter = {"n": 0}
    optimizer_agent, validator_agent = _accepting_agents(call_counter)

    runs, rate_tolerance = run_all_repetitions(
        n_repetitions=2, ledger_dir=tmp_path,
        trajectory=_TRAJECTORY, search_bounds=_SEARCH_BOUNDS, rate_tolerance=_LOOSE_TOLERANCE,
        optimizer_agent=optimizer_agent, validator_agent=validator_agent,
    )

    assert len(runs) == 2
    assert call_counter["n"] == 2
    assert (tmp_path / "run_01_ledger.json").exists()
    assert (tmp_path / "run_02_ledger.json").exists()


def test_run_all_repetitions_resumes_without_recalling_the_optimizer_for_completed_runs(tmp_path):
    """
    An existing, non-empty run_NN_ledger.json must be reused, with no new
    Optimizer call. Run 1 is a complete ledger; run 2 is an empty file
    (as after a crash mid-write), so only run 2 should be redone.
    """
    call_counter = {"n": 0}
    optimizer_agent, validator_agent = _accepting_agents(call_counter)

    first_pass_runs, rate_tolerance = run_all_repetitions(
        n_repetitions=1, ledger_dir=tmp_path,
        trajectory=_TRAJECTORY, search_bounds=_SEARCH_BOUNDS, rate_tolerance=_LOOSE_TOLERANCE,
        optimizer_agent=optimizer_agent, validator_agent=validator_agent,
    )
    assert call_counter["n"] == 1
    completed_run_01 = first_pass_runs[0]

    # Simulate the exact failure mode observed in the real study: a
    # crash mid-write leaves a zero-byte ledger file for repetition 2.
    (tmp_path / "run_02_ledger.json").write_bytes(b"")

    second_pass_runs, _ = run_all_repetitions(
        n_repetitions=2, ledger_dir=tmp_path,
        trajectory=_TRAJECTORY, search_bounds=_SEARCH_BOUNDS, rate_tolerance=_LOOSE_TOLERANCE,
        optimizer_agent=optimizer_agent, validator_agent=validator_agent,
    )

    # Only run 2 needed a new Optimizer call; run 1 was loaded from disk
    assert call_counter["n"] == 2
    assert len(second_pass_runs) == 2
    assert second_pass_runs[0] == completed_run_01

    reloaded_run_01 = AgenticRun.model_validate(
        json.loads((tmp_path / "run_01_ledger.json").read_text(encoding="utf-8"))
    )
    assert reloaded_run_01 == completed_run_01
    assert (tmp_path / "run_02_ledger.json").stat().st_size > 0


def test_run_all_repetitions_treats_a_zero_byte_ledger_as_not_yet_done(tmp_path):
    """A bare zero-byte file (the exact artifact a mid-write crash leaves
    behind) must never be mistaken for a completed run."""
    (tmp_path / "run_01_ledger.json").write_bytes(b"")
    call_counter = {"n": 0}
    optimizer_agent, validator_agent = _accepting_agents(call_counter)

    runs, _ = run_all_repetitions(
        n_repetitions=1, ledger_dir=tmp_path,
        trajectory=_TRAJECTORY, search_bounds=_SEARCH_BOUNDS, rate_tolerance=_LOOSE_TOLERANCE,
        optimizer_agent=optimizer_agent, validator_agent=validator_agent,
    )

    assert call_counter["n"] == 1
    assert (tmp_path / "run_01_ledger.json").stat().st_size > 0
    assert len(runs[0].entries) >= 1
