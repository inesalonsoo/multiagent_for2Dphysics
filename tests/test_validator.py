"""
Known-answer tests for agents/validator.py.

The property that matters most: the hard Boolean checks feeding
ValidatorDecision are computed in deterministic Python, independently of
the LLM's opinion -- this is the validator-level proof that complements
agents/schemas.py's schema-level guarantee (a computed-False check must
survive even LLM enthusiasm). Also tested: the three-way ill-posed /
valid-but-wrong / valid-and-right branch, the reused Phase 2 rate
tolerance, and the dormant Boltzmann socket.
"""

import os

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.schemas import PipelineConfig, PipelineResult
from agents.validator import (
    REFERENCE_BETA,
    _check_boltzmann_ratio_matches_analytical,
    build_validator_agent,
    load_rate_tolerance,
    validate_pipeline_result,
)
from physics.known_answers import eyring_kramers_rate_0d

_CONFIG = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20)
_ANALYTICAL_RATE = 2.0 * eyring_kramers_rate_0d(beta=REFERENCE_BETA)
_LOOSE_TOLERANCE = 0.05  # a stand-in fixed tolerance for hermetic tests, not read from disk


def _tool_call_response(info: AgentInfo, **fields) -> ModelResponse:
    tool_name = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=fields)])


def _well_posed_result(n_macrostates_recovered, relaxation_rate_mean, **overrides):
    fields = dict(
        config=_CONFIG,
        n_macrostates_recovered=n_macrostates_recovered,
        macrostate_populations=[0.5, 0.5],
        slowest_implied_timescale=100.0,
        relaxation_rate_mean=relaxation_rate_mean,
        vamp2_score=1.9,
        trajectory_length_frames=750_000,
        n_visited_microstates=50,
        min_transition_count=30,
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def _agent_that_always_says(llm_verdict, reasoning="stub", suggested_change=None, call_counter=None):
    def fake_llm(messages, info: AgentInfo) -> ModelResponse:
        if call_counter is not None:
            call_counter["n"] += 1
        return _tool_call_response(
            info, llm_verdict=llm_verdict, reasoning=reasoning, suggested_change=suggested_change,
        )

    return build_validator_agent(model=FunctionModel(fake_llm))


def test_llm_enthusiasm_cannot_flip_a_computed_false_check():
    """
    The validator-level proof: even an LLM that always writes ACCEPT with
    glowing prose must not make validate_pipeline_result() return ACCEPT
    when the Python-computed physics check is False. Proves the
    computation feeding the schema's own guarantee is itself independent
    of the LLM's opinion, not just that the schema enforces it in
    isolation.
    """
    wrong_number_of_states_result = _well_posed_result(
        n_macrostates_recovered=1, relaxation_rate_mean=_ANALYTICAL_RATE
    )
    agent = _agent_that_always_says("ACCEPT", reasoning="looks great, ship it")

    decision = validate_pipeline_result(agent, wrong_number_of_states_result, _LOOSE_TOLERANCE)

    assert decision.two_states_recovered is False
    assert decision.verdict == "REJECT"
    assert decision.llm_overridden is True


def test_ill_posed_result_is_rejected_mechanically_without_calling_the_llm():
    """Outcome 1 of 3: ill-posed. Checked first, no physics checks
    computed, and the LLM is never even called -- there's no physics
    pattern for it to interpret."""
    ill_posed_result = PipelineResult(
        config=_CONFIG, error="msm_lagtime (500) >= trajectory_length_frames (100)",
        trajectory_length_frames=100,
    )
    call_counter = {"n": 0}
    agent = _agent_that_always_says("ACCEPT", call_counter=call_counter)

    decision = validate_pipeline_result(agent, ill_posed_result, _LOOSE_TOLERANCE)

    assert decision.is_ill_posed is True
    assert decision.verdict == "REJECT"
    assert call_counter["n"] == 0


def test_ill_posedness_overrides_even_suspiciously_passing_measurement_fields():
    """Defensive check: is_ill_posed must win even if a (shouldn't-happen)
    PipelineResult has both `error` set AND measurement fields that look
    fine -- ill-posedness is checked first, unconditionally."""
    contradictory_result = PipelineResult(
        config=_CONFIG, error="deeptime rejected a degenerate count matrix",
        n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE,
        trajectory_length_frames=750_000,
    )
    agent = _agent_that_always_says("ACCEPT")

    decision = validate_pipeline_result(agent, contradictory_result, _LOOSE_TOLERANCE)

    assert decision.is_ill_posed is True
    assert decision.verdict == "REJECT"


def test_valid_but_wrong_result_is_rejected_with_llm_interpretation():
    """Outcome 2 of 3: valid config, failed physics. LLM IS called here,
    to interpret which check failed -- unlike the ill-posed branch."""
    wrong_rate_result = _well_posed_result(
        n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE * 5.0
    )
    call_counter = {"n": 0}
    agent = _agent_that_always_says(
        "REJECT", reasoning="rate is far off analytical", call_counter=call_counter
    )

    decision = validate_pipeline_result(agent, wrong_rate_result, _LOOSE_TOLERANCE)

    assert decision.is_ill_posed is False
    assert decision.two_states_recovered is True
    assert decision.rate_matches_analytical is False
    assert decision.verdict == "REJECT"
    assert call_counter["n"] == 1


def test_valid_and_right_result_is_accepted():
    """Outcome 3 of 3: valid config, both physics checks pass -- ACCEPT,
    mechanically, regardless of what the LLM adds."""
    correct_result = _well_posed_result(
        n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE
    )
    agent = _agent_that_always_says("ACCEPT", reasoning="both checks passed cleanly")

    decision = validate_pipeline_result(agent, correct_result, _LOOSE_TOLERANCE)

    assert decision.is_ill_posed is False
    assert decision.two_states_recovered is True
    assert decision.rate_matches_analytical is True
    assert decision.verdict == "ACCEPT"
    assert decision.llm_overridden is False


def test_rate_tolerance_boundary_is_respected():
    """A rate just inside the fixed tolerance passes; just outside fails --
    confirms rate_matches_analytical actually uses the tolerance argument,
    not some hardcoded threshold."""
    tolerance = 0.10
    just_inside = _well_posed_result(
        n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE * 1.09
    )
    just_outside = _well_posed_result(
        n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE * 1.11
    )
    agent = _agent_that_always_says("ACCEPT")

    inside_decision = validate_pipeline_result(agent, just_inside, tolerance)
    outside_decision = validate_pipeline_result(agent, just_outside, tolerance)

    assert inside_decision.rate_matches_analytical is True
    assert outside_decision.rate_matches_analytical is False


def test_boltzmann_socket_is_dormant_not_silently_wrong():
    """The Phase 4 socket must fail loudly if invoked today, not return a
    fabricated-looking answer -- see the function's docstring for why."""
    dummy_result = _well_posed_result(n_macrostates_recovered=2, relaxation_rate_mean=_ANALYTICAL_RATE)
    with pytest.raises(NotImplementedError):
        _check_boltzmann_ratio_matches_analytical(dummy_result, REFERENCE_BETA, tilt_b=0.1, tolerance=0.1)


@pytest.mark.skipif(
    not (os.path.exists("results/arrhenius_sweep_raw.npz")
         and os.path.exists("results/uq_sweep_raw.npz")),
    reason="requires cached results/arrhenius_sweep_raw.npz and results/uq_sweep_raw.npz -- "
           "run scripts.run_phase1_benchmark then scripts.run_phase2_uq to generate them",
)
def test_load_rate_tolerance_reuses_the_real_phase2_total_band():
    """Integration check against real cached Phase 1/2 output: the
    tolerance at beta=5 must be a small positive number, not a bare
    statistical width (which was ~1.1% and known to be too tight -- see
    PROJECT_STATE.md Sec 9) -- it should sit close to Phase 2's own
    reported ~3.16% total band at this beta."""
    tolerance = load_rate_tolerance(reference_beta=5.0)

    assert 0.0 < tolerance < 0.20
    assert tolerance == pytest.approx(0.0316, abs=0.01)
