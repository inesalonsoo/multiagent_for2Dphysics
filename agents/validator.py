"""
Validator Agent: the independent gatekeeper of the Phase 3 loop (the
Verifier in Ax-Prover's Orchestrator / Prover / Verifier design,
arXiv:2510.12787 Sec 3.1).

It computes the physics checks in plain Python before the LLM is called.
The LLM only interprets those fixed results, and agents/schemas.py's
ValidatorDecision recomputes the verdict from them, so the LLM cannot
override a check.

Three outcomes:
1. Ill-posed: the pipeline could not run the config (PipelineResult.error).
   Rejected mechanically; no physics checks, no LLM call.
2. Valid but wrong: the pipeline ran but a physics check failed. Rejected,
   with the LLM's interpretation and a suggested change.
3. Valid and right: both checks pass. Accepted.

Physics checks:
- two_states_recovered: t_2 / t_3 above pipeline.msm.MIN_TIMESCALE_SEPARATION.
- rate_matches_analytical: the measured rate is within rate_tolerance of the
  exact rate of the simulated chain at the reference beta. The tolerance is
  fixed once, before the loop runs (load_rate_tolerance()).

suggested_change is advisory: like the Optimizer's proposals, it is never
checked against physics. Only `verdict` carries a guarantee.
"""

import logging
from functools import lru_cache
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent

from agents.schemas import PipelineResult, ValidatorDecision
from physics.known_answers import euler_maruyama_relaxation_rate_0d
from pipeline.msm import MIN_TIMESCALE_SEPARATION

logger = logging.getLogger(__name__)

# pydantic-ai needs the "anthropic:" prefix; a bare model name fails with "Unknown model"
VALIDATOR_MODEL = "anthropic:claude-sonnet-5"
REFERENCE_BETA = 5.0  # the loop's reference trajectory (agents/loop.py)
REFERENCE_DT = 0.01  # its time step; must equal agents/loop.py DT (tested)
PHASE1_RESULTS_PATH = "results/arrhenius_sweep_raw.npz"

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
    What the Validator LLM itself returns: a verdict, its reasoning and a
    suggestion. Deliberately narrower than ValidatorDecision: the physics
    checks are computed in code (_compute_physics_checks) and never asked
    of the LLM. validate_pipeline_result() combines the two.
    """

    model_config = ConfigDict(extra="forbid")

    llm_verdict: Literal["ACCEPT", "REJECT"]
    reasoning: str
    suggested_change: Optional[str] = None


def build_validator_agent(model: Any = None) -> Agent[None, ValidatorLLMInterpretation]:
    """
    Create the Validator's LLM agent. Tests pass a scripted fake `model`
    so they never call a real API.
    """
    return Agent(
        model or VALIDATOR_MODEL,
        output_type=ValidatorLLMInterpretation,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
    )


def load_rate_tolerance(reference_beta: float = REFERENCE_BETA) -> float:
    """
    Relative tolerance for rate_matches_analytical: three times the
    replica-to-replica relative spread of Phase 1's rates at reference_beta.

    One trajectory's rate scatters around the exact value by about that
    spread. The spread measures precision (how much independent
    trajectories disagree); it never uses the deviation from the exact
    rate. It comes from only 6 replicas, so it is itself uncertain: at
    beta = 5 it is 1.87%, while the trend over all beta (about
    1/sqrt(number of barrier crossings)) suggests about 3.4%. Call once, before the
    loop. Requires Phase 1's results (results/arrhenius_sweep_raw.npz).
    """
    phase1 = np.load(PHASE1_RESULTS_PATH)
    beta_index = list(phase1["beta_values"]).index(reference_beta)
    rates = phase1["rate"][beta_index]
    resolved_rates = rates[~np.isnan(rates)]

    replica_sigma = resolved_rates.std(ddof=1) / resolved_rates.mean()
    return float(3.0 * replica_sigma)


@lru_cache(maxsize=None)
def reference_rate(reference_beta: float = REFERENCE_BETA) -> float:
    """Exact rate of the simulated chain at reference_beta (computed once)."""
    return euler_maruyama_relaxation_rate_0d(beta=reference_beta, dt=REFERENCE_DT)


def _compute_physics_checks(result: PipelineResult, reference_beta: float, rate_tolerance: float):
    """
    Compute (two_states_recovered, rate_matches_analytical) in plain Python.
    Only called when result.error is None, so the measurements are present.
    """
    two_states_recovered = (result.timescale_separation is not None
                            and result.timescale_separation > MIN_TIMESCALE_SEPARATION)

    exact_rate = reference_rate(reference_beta)
    relative_deviation = abs(result.relaxation_rate_mean - exact_rate) / exact_rate
    rate_matches_analytical = relative_deviation <= rate_tolerance

    return two_states_recovered, rate_matches_analytical


def _check_boltzmann_ratio_matches_analytical(result: PipelineResult, reference_beta: float,
                                                tilt_b: float, tolerance: float) -> bool:
    """
    Placeholder for the tilted-well (b != 0) population check, for Phase 4.
    At Phase 3's symmetric reference the expected ratio is 1, so the check
    would not discriminate. When needed, compare result.macrostate_populations
    (paired with result.macrostate_well_identity) with
    physics.known_answers.boltzmann_population_ratio(beta, A, b), and add the
    result as a field on ValidatorDecision.
    """
    raise NotImplementedError("Phase 4 placeholder: tilted-well population check.")


def _build_interpretation_prompt(result: PipelineResult, two_states_recovered: bool,
                                   rate_matches_analytical: bool) -> str:
    """Prompt for a result that ran: the checks first, stated as fixed,
    then the measurements for context."""
    return (
        "These physics checks have ALREADY BEEN COMPUTED in Python and cannot be "
        "changed by you -- interpret them, do not recompute them:\n"
        f"  two_states_recovered: {two_states_recovered}\n"
        f"  rate_matches_analytical: {rate_matches_analytical}\n\n"
        "Measured values, for context:\n"
        f"  timescale_separation (t_2/t_3): {result.timescale_separation}\n"
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
    The Validator's single step. A config that could not run is rejected
    straight away, with no physics checks and no LLM call. Otherwise the
    physics checks are computed in code and the LLM is asked only to
    interpret them.

    `rate_tolerance` comes from load_rate_tolerance(), called once before
    the loop.
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
