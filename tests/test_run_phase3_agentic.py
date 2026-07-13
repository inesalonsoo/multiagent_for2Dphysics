"""
Known-answer tests for scripts/run_phase3_agentic.py.

The property that matters here, per the PI's explicit instruction before
spending more real API budget: resumability. A script crash partway
through a real, billed study must not force re-paying for repetitions
that already completed successfully. This is tested deliberately, with
fakes, in a temp directory -- not left to be discovered under a real
crash a second time.
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
    """Fake Optimizer + Validator that always propose/accept the same
    well-posed config, counting how many times the Optimizer is actually
    invoked -- the resumability signal this test checks."""
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
    The property that matters: a run_NN_ledger.json that already exists
    (and is non-empty) must be loaded and reused, NOT regenerated -- the
    Optimizer must not be called again for it. Pre-populates run_01 (a
    real, valid ledger from a first pass) and leaves run_02 as a 0-byte
    file (simulating the exact mid-write crash the real study hit), then
    confirms only run_02 triggers a fresh call.
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

    # Only run 2 should have triggered a new Optimizer call -- run 1 was
    # already complete and must have been loaded from disk, not re-run.
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
