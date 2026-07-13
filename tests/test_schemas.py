"""
Known-answer tests for agents/schemas.py.

The one property that actually matters here is mechanical, not statistical:
ValidatorDecision.verdict must be governed by the hard Boolean checks alone,
regardless of what the LLM wrote for llm_verdict. Everything else is basic
construction/round-trip sanity for the ledger format.
"""

import json

import pytest
from pydantic import ValidationError

from agents.schemas import (
    AgenticRun,
    LedgerEntry,
    OptimizerProposal,
    PipelineConfig,
    PipelineResult,
    ValidatorDecision,
)


def _make_config(n_clusters=50, cluster_seed=42, msm_lagtime=20):
    return PipelineConfig(n_clusters=n_clusters, cluster_seed=cluster_seed, msm_lagtime=msm_lagtime)


def _make_result(config, **overrides):
    fields = dict(
        config=config,
        n_macrostates_recovered=2,
        macrostate_populations=[0.48, 0.52],
        slowest_implied_timescale=120.0,
        relaxation_rate_mean=0.0117,
        vamp2_score=1.85,
        trajectory_length_frames=1_500_000,
        n_visited_microstates=50,
        min_transition_count=40,
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def test_verdict_is_accept_when_every_hard_check_passes():
    decision = ValidatorDecision(
        two_states_recovered=True,
        rate_matches_analytical=True,
        is_ill_posed=False,
        llm_verdict="ACCEPT",
        reasoning="Two macrostates recovered; rate within tolerance of the analytical value.",
    )
    assert decision.verdict == "ACCEPT"
    assert decision.llm_overridden is False


def test_verdict_forced_to_reject_when_a_hard_check_fails_despite_llm_saying_accept():
    """
    The core guarantee: an LLM that writes ACCEPT next to a failing check
    cannot make the schema report ACCEPT. This is what makes "the
    Validator's hard Boolean gate overrides the LLM" a testable, not just
    aspirational, property.
    """
    decision = ValidatorDecision(
        two_states_recovered=False,
        rate_matches_analytical=True,
        is_ill_posed=False,
        llm_verdict="ACCEPT",
        reasoning="Only one macrostate was recovered, but the rate looked plausible.",
    )
    assert decision.verdict == "REJECT"
    assert decision.llm_overridden is True


def test_verdict_forced_to_reject_when_ill_posed_even_if_other_checks_pass():
    decision = ValidatorDecision(
        two_states_recovered=True,
        rate_matches_analytical=True,
        is_ill_posed=True,
        ill_posedness_reasons=["msm_lagtime >= trajectory_length_frames"],
        llm_verdict="ACCEPT",
        reasoning="Both physics checks technically passed on a degenerate lag time.",
    )
    assert decision.verdict == "REJECT"
    assert decision.llm_overridden is True


def test_llm_overridden_is_false_when_llm_correctly_predicts_reject():
    decision = ValidatorDecision(
        two_states_recovered=False,
        rate_matches_analytical=False,
        is_ill_posed=False,
        llm_verdict="REJECT",
        reasoning="Only one macrostate recovered and the rate was off by an order of magnitude.",
    )
    assert decision.verdict == "REJECT"
    assert decision.llm_overridden is False


def test_passed_in_verdict_field_is_ignored_not_just_defaulted():
    """
    Even if a caller explicitly passes verdict="ACCEPT" alongside a failing
    check, the model_validator must still overwrite it -- the guarantee
    cannot be bypassed by supplying the field directly.
    """
    decision = ValidatorDecision(
        two_states_recovered=False,
        rate_matches_analytical=True,
        is_ill_posed=False,
        llm_verdict="REJECT",
        reasoning="Physics check failed.",
        verdict="ACCEPT",
    )
    assert decision.verdict == "REJECT"


def test_pipeline_result_represents_a_failed_run_without_measurement_fields():
    config = _make_config()
    failed_result = PipelineResult(config=config, error="deeptime: lagtime >= trajectory length")
    assert failed_result.n_macrostates_recovered is None
    assert failed_result.relaxation_rate_mean is None


def test_contract_models_reject_unknown_fields():
    """extra='forbid' means a malformed/hallucinated field from an LLM's
    structured output raises at parse time instead of being silently dropped."""
    with pytest.raises(ValidationError):
        PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20, tica_lag=5)


def test_ledger_entry_round_trips_through_json():
    """
    The ledger is written to disk as JSON (results/ledger.json). A
    LedgerEntry built from real-looking data must serialize and parse back
    to an identical object -- this is the actual artifact end users read.
    """
    config = _make_config()
    result = _make_result(config)
    decision = ValidatorDecision(
        two_states_recovered=True,
        rate_matches_analytical=True,
        is_ill_posed=False,
        llm_verdict="ACCEPT",
        reasoning="Both hard checks passed.",
    )
    entry = LedgerEntry(
        iteration=1,
        proposal=OptimizerProposal(config=config, reasoning="Starting from the Phase 1 defaults."),
        result=result,
        decision=decision,
        next_action="stop_accepted",
        orchestrator_note="Validator accepted on the first iteration.",
    )

    round_tripped = LedgerEntry.model_validate(json.loads(entry.model_dump_json()))
    assert round_tripped == entry


def test_agentic_run_holds_multiple_entries_and_final_outcome():
    config = _make_config()
    result = _make_result(config)
    decision = ValidatorDecision(
        two_states_recovered=True,
        rate_matches_analytical=True,
        is_ill_posed=False,
        llm_verdict="ACCEPT",
        reasoning="Both hard checks passed.",
    )
    entry = LedgerEntry(
        iteration=1,
        proposal=OptimizerProposal(config=config, reasoning="Starting from the Phase 1 defaults."),
        result=result,
        decision=decision,
        next_action="stop_accepted",
    )
    run = AgenticRun(
        max_iterations=15,
        entries=[entry],
        stop_reason="validator_accepted",
        accepted_config=config,
    )

    assert run.entries[0].decision.verdict == "ACCEPT"
    assert run.accepted_config == config
