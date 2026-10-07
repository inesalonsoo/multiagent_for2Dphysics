"""
Tests for agents/optimizer.py, all with scripted fake LLMs (no API calls).

They never judge whether a proposal is good (there is no known optimum).
They check the interface: malformed output is retried, the previous result
(including a failure) reaches the prompt, and an Optimizer that reacts to a
failure proposes something different.
"""

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.optimizer import (
    OPTIMIZER_SYSTEM_PROMPT,
    SearchBounds,
    _format_history_for_prompt,
    build_optimizer_agent,
    propose_next_config,
)
from agents.schemas import LedgerEntry, OptimizerProposal, PipelineConfig, PipelineResult, ValidatorDecision

_BOUNDS = SearchBounds(trajectory_length_frames=1_500_000, max_n_clusters=200)


def _failed_ledger_entry(iteration, config, error_message):
    result = PipelineResult(config=config, error=error_message, trajectory_length_frames=100)
    decision = ValidatorDecision(
        two_states_recovered=False,
        rate_matches_analytical=False,
        is_ill_posed=True,
        ill_posedness_reasons=[error_message],
        llm_verdict="REJECT",
        reasoning="The pipeline tool could not even complete on this config.",
    )
    proposal = OptimizerProposal(config=config, reasoning="Starting from the Phase 1 defaults.")
    return LedgerEntry(
        iteration=iteration, proposal=proposal, result=result, decision=decision, next_action="continue",
    )


def _tool_call_response(info: AgentInfo, **fields) -> ModelResponse:
    tool_name = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=fields)])


def test_search_bounds_prompt_does_not_reveal_a_solved_lag_value():
    """
    The prompt must describe the search space and the physics that bounds
    the lag, never a likely answer. When it once contained the converged
    lag, every real run proposed the same config and no search happened.
    """
    bounds = SearchBounds(trajectory_length_frames=1_500_000, max_n_clusters=200)
    prompt_text = bounds.as_prompt_text()

    assert "plateaus near msm_lagtime=" not in prompt_text
    assert "well-motivated starting region" not in prompt_text
    assert not hasattr(bounds, "known_converged_lagtime")


def test_system_prompt_states_vamp2_is_a_soft_guide_not_the_acceptance_criterion():
    """
    The system prompt must separate VAMP-2 (a soft guide) from the physics
    checks that decide acceptance.
    """
    assert "SOFT GUIDE" in OPTIMIZER_SYSTEM_PROMPT
    assert "two_states_recovered" in OPTIMIZER_SYSTEM_PROMPT
    assert "rate_matches_analytical" in OPTIMIZER_SYSTEM_PROMPT


def test_format_history_states_the_hard_physics_gates_explicitly():
    """
    The history shown to the Optimizer must list the two physics checks as
    explicit fields, not only inside the Validator's prose.
    """
    config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20)
    result = PipelineResult(config=config, error=None, n_macrostates_recovered=2,
                             vamp2_score=2.1, relaxation_rate_mean=0.0119,
                             trajectory_length_frames=1_500_000)
    decision = ValidatorDecision(
        two_states_recovered=True, rate_matches_analytical=False, is_ill_posed=False,
        llm_verdict="REJECT", reasoning="Rate did not match analytical.",
    )
    entry = LedgerEntry(
        iteration=1,
        proposal=OptimizerProposal(config=config, reasoning="First attempt."),
        result=result, decision=decision, next_action="continue",
    )

    rendered = _format_history_for_prompt([entry])

    assert "two_states_recovered=True" in rendered
    assert "rate_matches_analytical=False" in rendered


def test_format_history_is_empty_message_on_first_iteration():
    assert _format_history_for_prompt([]) == "No iterations yet -- this is the first proposal."


def test_format_history_includes_the_failed_result_error_message():
    failing_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=2_000_000)
    entry = _failed_ledger_entry(1, failing_config, "msm_lagtime (2000000) >= trajectory_length_frames (100)")

    rendered = _format_history_for_prompt([entry])

    assert "FAILED" in rendered
    assert "msm_lagtime (2000000) >= trajectory_length_frames (100)" in rendered


def test_format_history_surfaces_the_validators_suggested_change():
    """
    The Validator's suggested_change must reach the Optimizer's prompt, so
    the Optimizer can use the feedback.
    """
    config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=5)
    result = PipelineResult(config=config, error=None, n_macrostates_recovered=1,
                             trajectory_length_frames=1_500_000)
    decision = ValidatorDecision(
        two_states_recovered=False, rate_matches_analytical=False, is_ill_posed=False,
        llm_verdict="REJECT", reasoning="Only one macrostate recovered.",
        suggested_change="Try a longer msm_lagtime -- the dynamics may not be Markovian yet at lag=5.",
    )
    entry = LedgerEntry(
        iteration=1,
        proposal=OptimizerProposal(config=config, reasoning="First attempt."),
        result=result, decision=decision, next_action="continue",
    )

    rendered = _format_history_for_prompt([entry])

    assert "Try a longer msm_lagtime" in rendered


def test_previous_failed_result_reaches_the_prompt_sent_to_the_model():
    """End-to-end plumbing check: capture the actual messages a fake LLM
    receives and confirm the failed result's error text is really in
    there, not just present in some internal string nobody sends."""
    failing_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=2_000_000)
    entry = _failed_ledger_entry(1, failing_config, "msm_lagtime (2000000) >= trajectory_length_frames (100)")
    captured = {}

    def fake_llm(messages, info: AgentInfo) -> ModelResponse:
        captured["prompt_text"] = "\n".join(
            part.content for message in messages for part in message.parts
            if hasattr(part, "content") and isinstance(part.content, str)
        )
        return _tool_call_response(info, config=dict(n_clusters=60, cluster_seed=1, msm_lagtime=25),
                                    reasoning="Moving away from the failed lagtime.")

    agent = build_optimizer_agent(model=FunctionModel(fake_llm))
    propose_next_config(agent, [entry], _BOUNDS)

    assert "msm_lagtime (2000000) >= trajectory_length_frames (100)" in captured["prompt_text"]


def test_malformed_response_is_retried_not_silently_accepted():
    """pydantic-ai's own automatic output-validation retry: a first,
    structurally invalid tool call must not become the final result."""
    call_count = {"n": 0}

    def flaky_llm(messages, info: AgentInfo) -> ModelResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # malformed: config is missing required fields entirely
            return _tool_call_response(info, config={"n_clusters": 50}, reasoning="incomplete")
        return _tool_call_response(
            info, config=dict(n_clusters=50, cluster_seed=42, msm_lagtime=20), reasoning="valid retry",
        )

    agent = build_optimizer_agent(model=FunctionModel(flaky_llm))
    proposal = propose_next_config(agent, [], _BOUNDS)

    assert call_count["n"] == 2
    assert proposal.config.n_clusters == 50
    assert proposal.reasoning == "valid retry"


def test_exhausted_retries_raises_instead_of_returning_bad_data():
    """If the model NEVER produces valid structured output, the failure
    must surface as a clear exception (for the future Orchestrator to
    catch), not as a corrupted or default-valued OptimizerProposal."""

    def always_malformed_llm(messages, info: AgentInfo) -> ModelResponse:
        return _tool_call_response(info, config={"n_clusters": "not-an-int"}, reasoning="still broken")

    agent = build_optimizer_agent(model=FunctionModel(always_malformed_llm))

    with pytest.raises(UnexpectedModelBehavior):
        propose_next_config(agent, [], _BOUNDS)


def test_optimizer_proposes_a_different_config_after_a_failure():
    """
    With a fake LLM that reacts to the failure shown in the prompt, the
    next proposal must differ from the config that failed. This tests that
    the feedback is wired through, not that a real LLM is smart.
    """
    failing_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=1_499_999)
    entry = _failed_ledger_entry(
        1, failing_config, "msm_lagtime (1499999) >= trajectory_length_frames (1500000)"
    )

    def reacts_to_failure_llm(messages, info: AgentInfo) -> ModelResponse:
        prompt_text = "\n".join(
            part.content for message in messages for part in message.parts
            if hasattr(part, "content") and isinstance(part.content, str)
        )
        if "FAILED" in prompt_text:
            new_lagtime = 20  # move far away from the failing near-trajectory-length value
        else:
            new_lagtime = 1_499_999  # would only happen if the failure signal never arrived
        return _tool_call_response(
            info, config=dict(n_clusters=50, cluster_seed=42, msm_lagtime=new_lagtime),
            reasoning="Moving lagtime well below the trajectory length after the failure.",
        )

    agent = build_optimizer_agent(model=FunctionModel(reacts_to_failure_llm))
    proposal = propose_next_config(agent, [entry], _BOUNDS)

    assert proposal.config != failing_config
    assert proposal.config.msm_lagtime != failing_config.msm_lagtime
