"""
Known-answer tests for agents/orchestrator.py.

The Orchestrator has no LLM in it, so unlike test_optimizer.py/
test_validator.py this file never needs a pydantic_ai fake -- routing is
tested entirely with plain scripted Python closures, proving the
Orchestrator's decisions are a pure, deterministic function of the
verdict sequence and iteration count. The one exception is
test_run_agentic_loop_with_real_agents_wires_the_real_modules_together,
a lightweight smoke test of the real-agent adapter (still using
FunctionModel fakes for the two LLMs -- no real API calls anywhere in
this suite).
"""

import json

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agents.optimizer import SearchBounds, build_optimizer_agent
from agents.orchestrator import (
    decide_next_action,
    run_agentic_loop,
    run_agentic_loop_with_real_agents,
)
from agents.schemas import AgenticRun, OptimizerProposal, PipelineConfig, PipelineResult, ValidatorDecision
from agents.validator import build_validator_agent
from physics.simulate_0d import run_trajectory_0d

_STUB_CONFIG = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20)


def _stub_result(config, n_macrostates_recovered=2, relaxation_rate_mean=0.0117, error=None):
    if error is not None:
        return PipelineResult(config=config, error=error, trajectory_length_frames=100)
    return PipelineResult(
        config=config, n_macrostates_recovered=n_macrostates_recovered,
        macrostate_populations=[0.5, 0.5], slowest_implied_timescale=100.0,
        relaxation_rate_mean=relaxation_rate_mean, vamp2_score=1.9,
        trajectory_length_frames=750_000, n_visited_microstates=config.n_clusters,
        min_transition_count=30,
    )


def _reject_decision(reasoning="physics check failed"):
    return ValidatorDecision(
        two_states_recovered=False, rate_matches_analytical=True, is_ill_posed=False,
        llm_verdict="REJECT", reasoning=reasoning,
    )


def _accept_decision(reasoning="both checks passed"):
    return ValidatorDecision(
        two_states_recovered=True, rate_matches_analytical=True, is_ill_posed=False,
        llm_verdict="ACCEPT", reasoning=reasoning,
    )


def _ill_posed_decision(error_message):
    return ValidatorDecision(
        two_states_recovered=False, rate_matches_analytical=False, is_ill_posed=True,
        ill_posedness_reasons=[error_message], llm_verdict="REJECT", reasoning="ill-posed config",
    )


def _accept_with_llm_disagreement_decision():
    """Hard checks pass, but the LLM's own stated verdict disagreed --
    ValidatorDecision's own model_validator forces verdict=ACCEPT and
    llm_overridden=True regardless."""
    return ValidatorDecision(
        two_states_recovered=True, rate_matches_analytical=True, is_ill_posed=False,
        llm_verdict="REJECT", reasoning="LLM thought this should be rejected; hard checks disagree.",
    )


def _make_scripted_fns(rounds):
    """
    rounds: list of (PipelineConfig, PipelineResult, ValidatorDecision)
    tuples, one per iteration. Returns (propose_fn, run_pipeline_fn,
    validate_fn) that replay these in lockstep, driven by len(history)
    -- propose_fn is always called first each iteration by
    run_agentic_loop, so it alone needs to read the history.
    """
    current_index = {"i": 0}

    def propose_fn(history):
        current_index["i"] = len(history)
        config = rounds[current_index["i"]][0]
        return OptimizerProposal(config=config, reasoning=f"scripted round {current_index['i']}")

    def run_pipeline_fn(config):
        return rounds[current_index["i"]][1]

    def validate_fn(result):
        return rounds[current_index["i"]][2]

    return propose_fn, run_pipeline_fn, validate_fn


@pytest.mark.parametrize("verdict,iteration,max_iterations,expected", [
    ("ACCEPT", 1, 15, "stop_accepted"),
    ("ACCEPT", 15, 15, "stop_accepted"),  # success wins even exactly at the cap
    ("REJECT", 1, 15, "continue"),
    ("REJECT", 14, 15, "continue"),
    ("REJECT", 15, 15, "stop_iteration_cap_reached"),
])
def test_decide_next_action_is_a_pure_function_of_verdict_and_iteration(
    verdict, iteration, max_iterations, expected
):
    assert decide_next_action(verdict, iteration, max_iterations) == expected
    # Calling again with identical inputs must give the identical output --
    # no hidden state, no randomness.
    assert decide_next_action(verdict, iteration, max_iterations) == expected


def test_orchestrator_routing_is_deterministic_given_the_same_verdict_sequence():
    """Same sequence of verdicts in, same routing out -- run the identical
    scripted history through the loop twice and require an identical
    AgenticRun both times."""
    rounds = [
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _reject_decision()),
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _accept_decision()),
    ]
    propose_fn_a, run_pipeline_fn_a, validate_fn_a = _make_scripted_fns(rounds)
    propose_fn_b, run_pipeline_fn_b, validate_fn_b = _make_scripted_fns(rounds)

    run_a = run_agentic_loop(propose_fn_a, run_pipeline_fn_a, validate_fn_a, max_iterations=15)
    run_b = run_agentic_loop(propose_fn_b, run_pipeline_fn_b, validate_fn_b, max_iterations=15)

    assert run_a == run_b
    assert [entry.next_action for entry in run_a.entries] == ["continue", "stop_accepted"]


def test_loop_stops_at_the_iteration_the_validator_approves():
    """APPROVE-stop: a scripted sequence that rejects twice then approves
    must stop exactly at iteration 3, with a success status."""
    rounds = [
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _reject_decision()),
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _reject_decision()),
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _accept_decision()),
    ]
    propose_fn, run_pipeline_fn, validate_fn = _make_scripted_fns(rounds)

    run = run_agentic_loop(propose_fn, run_pipeline_fn, validate_fn, max_iterations=15)

    assert len(run.entries) == 3
    assert run.entries[-1].iteration == 3
    assert run.stop_reason == "validator_accepted"
    assert run.accepted_config == _STUB_CONFIG


def test_loop_exhausts_at_max_iterations_when_validator_never_approves():
    """max_iterations-stop: a sequence that never approves must stop
    exactly at the cap, with a DISTINCT "exhausted" status -- not
    conflated with the success exit above."""
    max_iterations = 3
    rounds = [
        (_STUB_CONFIG, _stub_result(_STUB_CONFIG), _reject_decision())
        for _ in range(max_iterations)
    ]
    propose_fn, run_pipeline_fn, validate_fn = _make_scripted_fns(rounds)

    run = run_agentic_loop(propose_fn, run_pipeline_fn, validate_fn, max_iterations=max_iterations)

    assert len(run.entries) == max_iterations
    assert run.entries[-1].next_action == "stop_iteration_cap_reached"
    assert run.stop_reason == "iteration_cap_reached"
    assert run.accepted_config is None


def test_ledger_is_faithful_not_flattering():
    """
    Nothing the loop did is missing, nothing recorded is fabricated: an
    ill-posed iteration, a valid-but-wrong iteration, and a valid-and-
    right iteration where the LLM's own verdict disagreed with the
    mechanical one must ALL survive intact, including through a full
    JSON round trip.
    """
    ill_posed_config = PipelineConfig(n_clusters=50, cluster_seed=1, msm_lagtime=2_000_000)
    wrong_config = PipelineConfig(n_clusters=50, cluster_seed=2, msm_lagtime=20)
    right_config = PipelineConfig(n_clusters=50, cluster_seed=3, msm_lagtime=20)

    rounds = [
        (ill_posed_config,
         _stub_result(ill_posed_config, error="msm_lagtime (2000000) >= trajectory_length_frames (100)"),
         _ill_posed_decision("msm_lagtime (2000000) >= trajectory_length_frames (100)")),
        (wrong_config, _stub_result(wrong_config, n_macrostates_recovered=1),
         _reject_decision("only one macrostate recovered")),
        (right_config, _stub_result(right_config, n_macrostates_recovered=2),
         _accept_with_llm_disagreement_decision()),
    ]
    propose_fn, run_pipeline_fn, validate_fn = _make_scripted_fns(rounds)

    run = run_agentic_loop(propose_fn, run_pipeline_fn, validate_fn, max_iterations=15)

    assert len(run.entries) == 3
    assert run.entries[0].decision.is_ill_posed is True
    assert run.entries[0].next_action == "continue"
    assert run.entries[1].decision.verdict == "REJECT"
    assert run.entries[1].decision.is_ill_posed is False
    assert run.entries[2].decision.verdict == "ACCEPT"
    assert run.entries[2].decision.llm_overridden is True
    assert run.stop_reason == "validator_accepted"
    assert run.accepted_config == right_config

    reloaded = AgenticRun.model_validate(json.loads(run.model_dump_json()))
    assert reloaded == run


def test_run_agentic_loop_with_real_agents_wires_the_real_modules_together():
    """
    Not a routing test (covered above with pure fakes) -- a lightweight
    smoke test that the adapter correctly wires the real Optimizer, tool,
    and Validator into run_agentic_loop(). FunctionModel fakes stand in
    for both LLMs (no real API calls); a small real trajectory and the
    real run_msm_pipeline exercise the actual deterministic pipeline. A
    loose, hermetic rate_tolerance keeps this test independent of cached
    Phase 2 files.
    """
    trajectory = run_trajectory_0d(n_steps=750_000, seed=7, beta=5.0, dt=0.01)
    search_bounds = SearchBounds(trajectory_length_frames=len(trajectory), max_n_clusters=100)

    def optimizer_fake_llm(messages, info):
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            config=dict(n_clusters=50, cluster_seed=42, msm_lagtime=20),
            reasoning="Using Phase 1's own validated config.",
        ))])

    def validator_fake_llm(messages, info):
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=dict(
            llm_verdict="ACCEPT", reasoning="Both checks look consistent.", suggested_change=None,
        ))])

    optimizer_agent = build_optimizer_agent(model=FunctionModel(optimizer_fake_llm))
    validator_agent = build_validator_agent(model=FunctionModel(validator_fake_llm))

    run = run_agentic_loop_with_real_agents(
        optimizer_agent, validator_agent, trajectory, dt=0.01,
        search_bounds=search_bounds, rate_tolerance=0.5, max_iterations=3,
    )

    assert len(run.entries) == 1
    assert run.entries[0].result.error is None
    assert run.entries[0].decision.verdict == "ACCEPT"
    assert run.stop_reason == "validator_accepted"
