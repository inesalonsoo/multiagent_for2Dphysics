"""
Validator Agent (PROJECT_STATE.md Sec 6/7, module 3.4) -- Ax-Prover's
Verifier (arXiv:2510.12787 Sec 3.1). The independent gatekeeper: it
computes hard, deterministic physics checks in plain Python against
physics/known_answers.py BEFORE the LLM is ever called, and the LLM only
INTERPRETS those already-decided Booleans afterward -- it is never asked
for a yes/no on physics. agents/schemas.py's ValidatorDecision already
enforces this mechanically (a model_validator recomputes `verdict` from
the hard checks, ignoring whatever the LLM wrote); this module is what
proves the checks feeding that gate are themselves computed independently
of the LLM's opinion, not just that the gate is well-formed in isolation.

THREE OUTCOMES, NOT TWO -- the Ax-Prover Appendix C robustness behavior
this module exists to implement:
1. ILL-POSED: agents/tools.py's run_msm_pipeline already couldn't validly
   run the config (PipelineResult.error is set). Checked FIRST. The
   physics checks are never computed here -- they would be meaningless
   against None-valued measurement fields -- and the LLM is never called
   either, since there is no physics pattern for it to interpret. This
   branch is fully mechanical: REJECT, with a fixed message telling the
   Optimizer to move back inside the valid parameter ranges.
2. VALID BUT WRONG: the pipeline ran, but a hard physics check failed.
   REJECT, with the LLM interpreting WHICH check failed and suggesting
   what to try next.
3. VALID AND RIGHT: both hard checks pass. ACCEPT (mechanically, via
   agents/schemas.py's model_validator -- not because the LLM said so).
These three route differently once agents/orchestrator.py (module 3.5)
exists: (1) means "you're outside the valid region," (2) means "you're
inside it but this isn't the answer, keep searching" -- collapsing them
loses exactly the distinction Ax-Prover Appendix C is about.

RATE TOLERANCE -- reused, not reinvented. rate_matches_analytical is NOT
a bare equality: physics/known_answers.py's Eyring-Kramers rate is a
finite-beta asymptotic formula with a known, already-characterized
systematic bias on top of ordinary statistical noise (PROJECT_STATE.md
Sec 9). Phase 2 already learned, the hard way, that gating a measured
rate against a bare statistical confidence interval reproduces exactly
this failure: a tight CI can exclude a truth displaced by a real
systematic. This module's rate check therefore reuses Phase 2's own
already-validated TOTAL (statistical (+) systematic, in quadrature)
relative error band (see load_rate_tolerance() below), not a
freshly-invented, looser tolerance. That tolerance must be computed ONCE,
before evaluating any Phase 3 config, and threaded into every
validate_pipeline_result() call as a fixed number -- never recomputed
after seeing whether a particular result passes it.

BOLTZMANN CHECK -- dormant socket, not absent. See agents/schemas.py's
ValidatorDecision docstring and PROJECT_STATE.md Sec 9 (module 3.1):
omitted from Phase 3's active hard-check set because it is
non-discriminating at this loop's symmetric (b=0) reference (ratio ~1).
_check_boltzmann_ratio_matches_analytical() below is a real, named,
documented place for Phase 4's tilted-case reactivation to plug into --
deliberately left raising NotImplementedError (tested) rather than
silently absent, and rather than faked into "working" today.

[2026-07-12] Its prerequisite is now built, as its own first task of
Phase 4 (PROJECT_STATE.md Sec 9), BEFORE any tilted-potential run: this
function needed well-identity tracking (which PCCA+ macrostate label
corresponds to which physical well -- irrelevant while b=0, essential
once b!=0) that PipelineResult did not previously record.
agents/tools.py now computes `PipelineResult.macrostate_well_identity`
(one "x_plus"/"x_minus" per entry of macrostate_populations, in the same
order) via a new `_classify_well_identity()` helper, verified against the
installed deeptime's PCCA+ label ordering directly before writing it, not
assumed. Reactivating THIS function still requires wiring its computed
Boolean into ValidatorDecision (a new field there too) -- that's the
remaining, deliberate step left for the actual Phase 4 deployment, not
done here ahead of it.

ADVISORY, NOT VERIFIED: `suggested_change` (like the Optimizer's whole
proposal, agents/optimizer.py) is the LLM's best natural-language guess
at what might help -- it is never checked against physics and cannot be.
Only `verdict` carries a hard guarantee here; `suggested_change` carries
none. Keep that distinction visible to whoever reads a ValidatorDecision.
"""

import logging
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent

from agents.schemas import PipelineResult, ValidatorDecision
from physics.known_answers import eyring_kramers_rate_0d, expected_number_of_states

logger = logging.getLogger(__name__)

VALIDATOR_MODEL = "anthropic:claude-sonnet-5"  # PROJECT_STATE.md Sec 4 -- pydantic-ai
# requires the explicit "anthropic:" provider prefix; a bare "claude-sonnet-5" fails
# infer_model() with "Unknown model" (caught interactively before any real API call).
REFERENCE_BETA = 5.0  # Phase 1/2's baseline inverse temperature, PROJECT_STATE.md Sec 4

VALIDATOR_SYSTEM_PROMPT = """
You are the Validator in a three-agent verification loop (Orchestrator /
Optimizer / Validator) analyzing a stochastic double-well trajectory with
a Markov State Model pipeline.

You will be given the results of physics checks that have ALREADY BEEN
COMPUTED in plain Python against known analytical answers
(physics/known_answers.py). You cannot change, override, or recompute
these checks -- your only job is to interpret the pattern they show and
write:
- llm_verdict: "ACCEPT" only if BOTH checks given to you are True,
  "REJECT" otherwise. Even if you write ACCEPT, the checks -- not your
  verdict -- determine the actual outcome; get this right anyway, since
  it is recorded and compared against the mechanical result.
- reasoning: a short, concrete explanation of the pattern of pass/fail.
- suggested_change: if rejecting, a concrete suggestion for what the
  Optimizer might try differently next (e.g. a longer lag time, a
  different cluster count) -- your best guess, not itself verified
  against physics.
"""


class ValidatorLLMInterpretation(BaseModel):
    """
    The Validator LLM's own structured output -- deliberately NARROWER
    than ValidatorDecision (agents/schemas.py). It does NOT include
    two_states_recovered, rate_matches_analytical, or is_ill_posed: those
    are computed in plain Python (see _compute_physics_checks below) and
    never handed to the LLM to decide. This model exists only so the LLM
    call has a typed output at all; validate_pipeline_result() below
    combines it with the independently-computed hard checks to build the
    real ValidatorDecision.
    """

    model_config = ConfigDict(extra="forbid")

    llm_verdict: Literal["ACCEPT", "REJECT"]
    reasoning: str
    suggested_change: Optional[str] = None


def build_validator_agent(model: Any = None) -> Agent[None, ValidatorLLMInterpretation]:
    """
    Construct the Validator's pydantic-ai Agent. `model` defaults to this
    project's approved agent model string; tests pass a
    pydantic_ai.models.function.FunctionModel here instead, so nothing in
    tests/test_validator.py makes a real API call.
    """
    return Agent(
        model or VALIDATOR_MODEL,
        output_type=ValidatorLLMInterpretation,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
    )


def load_rate_tolerance(reference_beta: float = REFERENCE_BETA) -> float:
    """
    Load Phase 2's already-characterized, already-validated TOTAL
    (statistical (+) systematic, in quadrature) relative error band at
    reference_beta, and return it as a fixed tolerance for
    rate_matches_analytical. Reuses scripts.run_phase2_uq's own
    load_phase1_reference() and build_total_error_band() rather than
    re-deriving a looser version of the same idea -- see this module's
    docstring for why reuse matters here specifically.

    Call this ONCE, before evaluating any Phase 3 config, and thread the
    returned value into every validate_pipeline_result() call as
    `rate_tolerance` -- it must be fixed before seeing which results pass
    it, not recomputed per iteration.

    Requires results/uq_sweep_raw.npz and results/arrhenius_sweep_raw.npz
    (scripts.run_phase2_uq and scripts.run_phase1_benchmark's cached
    output) to already exist on disk.
    """
    from scripts.run_phase1_benchmark import BETA_VALUES
    from scripts.run_phase2_uq import build_total_error_band, load_phase1_reference

    uq_data = np.load("results/uq_sweep_raw.npz")
    assert np.array_equal(uq_data["beta_values"], BETA_VALUES), (
        "cached uq_sweep_raw.npz was computed for a different BETA_VALUES -- "
        "re-run scripts.run_phase2_uq"
    )
    beta_index = list(BETA_VALUES).index(reference_beta)

    phase1_mean_rate, systematic_relative = load_phase1_reference()
    _, _, statistical_relative = build_total_error_band(
        phase1_mean_rate, uq_data["rate_mean"], uq_data["rate_lower"], uq_data["rate_upper"],
        systematic_relative,
    )

    total_relative = np.sqrt(
        statistical_relative[beta_index] ** 2 + systematic_relative[beta_index] ** 2
    )
    return float(total_relative)


def _compute_physics_checks(result: PipelineResult, reference_beta: float, rate_tolerance: float):
    """
    Computes (two_states_recovered, rate_matches_analytical) in
    deterministic Python, grounded in physics/known_answers.py. Only
    called when result.error is None -- agents/tools.py guarantees the
    measurement fields used here are populated in that case.
    """
    two_states_recovered = result.n_macrostates_recovered == expected_number_of_states()

    analytical_rate = 2.0 * eyring_kramers_rate_0d(beta=reference_beta)
    relative_deviation = abs(result.relaxation_rate_mean - analytical_rate) / analytical_rate
    rate_matches_analytical = relative_deviation <= rate_tolerance

    return two_states_recovered, rate_matches_analytical


def _check_boltzmann_ratio_matches_analytical(result: PipelineResult, reference_beta: float,
                                                tilt_b: float, tolerance: float) -> bool:
    """
    DORMANT SOCKET -- not called from validate_pipeline_result() and not
    a field on ValidatorDecision yet. See this module's docstring for why
    it's dormant (non-discriminating at Phase 3's b=0 reference) and what
    Phase 4 needs to do to reactivate it: add
    boltzmann_ratio_matches_analytical to agents/schemas.py's
    ValidatorDecision AND wire this function's result into it here.

    [2026-07-12] Its data prerequisite is now built: `result.
    macrostate_well_identity` (agents/tools.py) records which PCCA+
    macrostate label, 0 or 1, corresponds to which physical well, x_plus
    or x_minus -- see PROJECT_STATE.md Sec 9. This function is still left
    raising NotImplementedError rather than actually implemented,
    deliberately: wiring a real tilt `b` and a third hard gate into
    ValidatorDecision/the active Phase 3 check set is Phase 4 deployment
    work, not infrastructure to build ahead of when it's needed. When
    that work happens, `result.macrostate_populations[i]` paired with
    `result.macrostate_well_identity[i]` (same index) is what makes the
    directional comparison against
    physics.known_answers.boltzmann_population_ratio(beta, A, b) possible.
    """
    raise NotImplementedError(
        "Phase 4 socket: needs well-identity tracking in PipelineResult before "
        "this can be computed correctly -- see this function's docstring."
    )


def _build_interpretation_prompt(result: PipelineResult, two_states_recovered: bool,
                                   rate_matches_analytical: bool) -> str:
    """Assemble the prompt for a WELL-POSED result: the already-computed
    checks first (framed as fixed, not up for debate), then the raw
    measurements for context."""
    return (
        "These physics checks have ALREADY BEEN COMPUTED in Python and cannot be "
        "changed by you -- interpret them, do not recompute them:\n"
        f"  two_states_recovered: {two_states_recovered}\n"
        f"  rate_matches_analytical: {rate_matches_analytical}\n\n"
        "Measured values, for context:\n"
        f"  n_macrostates_recovered: {result.n_macrostates_recovered}\n"
        f"  macrostate_populations: {result.macrostate_populations}\n"
        f"  relaxation_rate_mean: {result.relaxation_rate_mean}\n"
        f"  vamp2_score: {result.vamp2_score}\n\n"
        "Write your llm_verdict, reasoning, and (if rejecting) suggested_change."
    )


def validate_pipeline_result(
    validator_agent: Agent[None, ValidatorLLMInterpretation],
    result: PipelineResult,
    rate_tolerance: float,
    reference_beta: float = REFERENCE_BETA,
) -> ValidatorDecision:
    """
    The Validator's single step: check ill-posedness FIRST; if the
    config never validly ran, reject mechanically without computing
    meaningless physics checks or calling the LLM at all. Otherwise,
    compute the hard physics checks in Python, then ask the LLM to
    interpret (never decide) them.

    `rate_tolerance` must come from load_rate_tolerance(), called once
    up front -- see that function's docstring.
    """
    if result.error is not None:
        return ValidatorDecision(
            two_states_recovered=False,
            rate_matches_analytical=False,
            is_ill_posed=True,
            ill_posedness_reasons=[result.error],
            llm_verdict="REJECT",
            reasoning=f"Config was ill-posed, not a physics failure: {result.error}",
            suggested_change="Propose a config back inside the valid parameter ranges.",
        )

    two_states_recovered, rate_matches_analytical = _compute_physics_checks(
        result, reference_beta, rate_tolerance
    )
    prompt = _build_interpretation_prompt(result, two_states_recovered, rate_matches_analytical)
    llm_output = validator_agent.run_sync(prompt).output

    return ValidatorDecision(
        two_states_recovered=two_states_recovered,
        rate_matches_analytical=rate_matches_analytical,
        is_ill_posed=False,
        llm_verdict=llm_output.llm_verdict,
        reasoning=llm_output.reasoning,
        suggested_change=llm_output.suggested_change,
    )
