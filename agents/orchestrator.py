"""
The Orchestrator: runs the loop. It is the one agent with no LLM, because
its job involves no judgment, only routing:
1. Ask the Optimizer for a proposal, run it through the pipeline tool, and
   pass the result to the Validator, without changing anything.
2. Add each round to the history the Optimizer sees next time.
3. Decide whether to stop (decide_next_action(), the only place this
   happens). It trusts the Validator's verdict and never re-judges it.

Being deterministic makes it fully testable: tests/test_orchestrator.py
uses plain scripted functions instead of agents and checks that the same
verdicts always give the same routing.

A run ends in one of two ways, recorded as different outcomes: success
(the Validator accepted) or exhaustion (the iteration cap was reached with
no acceptance). Keeping them apart is what makes "how often does the loop
converge?" measurable. Every iteration goes into the ledger as it
happened, including rejected and ill-posed ones.
"""

from typing import Callable, Literal, Optional

import numpy as np
from pydantic_ai import Agent

from agents.optimizer import SearchBounds, propose_next_config
from agents.schemas import (
    AgenticRun,
    LedgerEntry,
    OptimizerProposal,
    PipelineConfig,
    PipelineResult,
    ValidatorDecision,
)
from agents.tools import run_msm_pipeline
from agents.validator import REFERENCE_BETA, ValidatorLLMInterpretation, validate_pipeline_result

MAX_ITERATIONS = 15  # hard stop, the human's decision

ProposeFn = Callable[[list], OptimizerProposal]
RunPipelineFn = Callable[[PipelineConfig], PipelineResult]
ValidateFn = Callable[[PipelineResult], ValidatorDecision]


def decide_next_action(
    verdict: Literal["ACCEPT", "REJECT"], iteration: int, max_iterations: int
) -> Literal["continue", "stop_accepted", "stop_iteration_cap_reached"]:
    """
    The whole stop decision, from the verdict and the iteration count
    alone. An ACCEPT wins even on the last allowed iteration, because the
    run did converge there.
    """
    if verdict == "ACCEPT":
        return "stop_accepted"
    if iteration >= max_iterations:
        return "stop_iteration_cap_reached"
    return "continue"


def _orchestrator_note_for(next_action: str) -> str:
    """Short plain-text note stored with each ledger entry."""
    if next_action == "stop_accepted":
        return "Validator approved this config -- stopping. Run converged."
    if next_action == "stop_iteration_cap_reached":
        return "Reached max_iterations without approval -- stopping. Run did NOT converge."
    return "Validator rejected this config -- continuing to the next iteration."


def run_agentic_loop(
    propose_fn: ProposeFn,
    run_pipeline_fn: RunPipelineFn,
    validate_fn: ValidateFn,
    max_iterations: int = MAX_ITERATIONS,
) -> AgenticRun:
    """
    The Orchestrator's loop. It takes three plain functions rather than the
    real agents, so the routing can be tested with scripted fakes;
    run_agentic_loop_with_real_agents() plugs in the real ones.

    Parameters
    ----------
    propose_fn : history -> OptimizerProposal
    run_pipeline_fn : PipelineConfig -> PipelineResult
    validate_fn : PipelineResult -> ValidatorDecision
    max_iterations : int, optional

    Returns
    -------
    AgenticRun
        The complete ledger of every iteration run.
    """
    history: list[LedgerEntry] = []

    for iteration in range(1, max_iterations + 1):
        proposal = propose_fn(history)
        result = run_pipeline_fn(proposal.config)
        decision = validate_fn(result)

        next_action = decide_next_action(decision.verdict, iteration, max_iterations)
        entry = LedgerEntry(
            iteration=iteration,
            proposal=proposal,
            result=result,
            decision=decision,
            next_action=next_action,
            orchestrator_note=_orchestrator_note_for(next_action),
        )
        history.append(entry)

        if next_action != "continue":
            break

    return _build_agentic_run(history, max_iterations)


def _build_agentic_run(history: list[LedgerEntry], max_iterations: int) -> AgenticRun:
    """Package the history into the final ledger object. The stop reason
    comes from the last entry, which is never "continue"."""
    last_entry = history[-1]

    if last_entry.next_action == "stop_accepted":
        stop_reason: Optional[Literal["validator_accepted", "iteration_cap_reached"]] = (
            "validator_accepted"
        )
        accepted_config: Optional[PipelineConfig] = last_entry.proposal.config
    else:
        stop_reason = "iteration_cap_reached"
        accepted_config = None

    return AgenticRun(
        max_iterations=max_iterations, entries=history,
        stop_reason=stop_reason, accepted_config=accepted_config,
    )


def run_agentic_loop_with_real_agents(
    optimizer_agent: Agent[None, OptimizerProposal],
    validator_agent: Agent[None, ValidatorLLMInterpretation],
    trajectory: np.ndarray,
    dt: float,
    search_bounds: SearchBounds,
    rate_tolerance: float,
    reference_beta: float = REFERENCE_BETA,
    max_iterations: int = MAX_ITERATIONS,
) -> AgenticRun:
    """
    Connect the real Optimizer, pipeline tool and Validator to
    run_agentic_loop(), which does all the routing. Used for real runs;
    tests call run_agentic_loop() with fakes.
    """
    def propose_fn(history: list[LedgerEntry]) -> OptimizerProposal:
        return propose_next_config(optimizer_agent, history, search_bounds)

    def run_pipeline_fn(config: PipelineConfig) -> PipelineResult:
        return run_msm_pipeline(config, trajectory, dt)

    def validate_fn(result: PipelineResult) -> ValidatorDecision:
        return validate_pipeline_result(validator_agent, result, rate_tolerance, reference_beta)

    return run_agentic_loop(propose_fn, run_pipeline_fn, validate_fn, max_iterations)
