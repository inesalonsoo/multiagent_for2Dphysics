"""
Orchestrator Agent (PROJECT_STATE.md Sec 6/7, module 3.5) -- Ax-Prover's
missing piece (arXiv:2510.12787 Sec 3.1.1). The one agent with NO LLM in
it, and that is the point: it makes no physics judgment and no search
judgment, it only ROUTES what the Optimizer and Validator produce.

Three responsibilities, nothing more:
1. TASK ASSIGNMENT -- calls the Optimizer for a proposal, then the
   deterministic tool to run it, then the Validator to check it. It never
   edits the Optimizer's PipelineConfig and never recomputes the
   Validator's checks; it passes each straight through to the next step.
2. FEEDBACK ROUTING -- appends every iteration's full round to the
   history, which is exactly what the next propose_next_config() call
   receives. There is no separate "routing" logic beyond keeping that
   history complete and in order -- the Optimizer already knows how to
   read a rejected/ill-posed entry (agents/optimizer.py, module 3.3).
3. THE STOP DECISION -- decide_next_action() below, and nothing else in
   this module, decides whether to continue. It reads
   ValidatorDecision.verdict as a SETTLED Boolean and acts on it; it never
   second-guesses whether a REJECT "should really" have been an ACCEPT --
   that would be re-deciding the Validator's job here, collapsing two
   roles into one and losing the separation the whole architecture is
   built on.

DETERMINISTIC BY DESIGN, TESTED AS SUCH: every other agent in this
project is fronted by an LLM and is therefore non-deterministic by
nature (agents/optimizer.py, agents/validator.py both document this
explicitly). The Orchestrator has no LLM call anywhere in it -- its
entire behavior is decide_next_action(verdict, iteration, max_iterations),
a pure function of a verdict sequence and a count. tests/
test_orchestrator.py never touches a real or fake LLM: it scripts plain
Python closures for "propose," "run the tool," and "validate," and
proves the SAME sequence of verdicts always produces the SAME routing,
deterministically. That is what makes the loop's control flow testable
even though the agents inside it are not.

TWO STOP CONDITIONS, TWO DISTINCT MEANINGS -- both real, both recorded,
never conflated. APPROVE-stop (Validator accepted) is success. Reaching
max_iterations without an ACCEPT is exhaustion -- the run did NOT
converge. agents/schemas.py's AgenticRun.stop_reason ("validator_accepted"
vs "iteration_cap_reached") already has a slot for exactly this
distinction (module 3.1); this module is what actually sets it correctly,
tested for both exits explicitly. Conflating them would let a
non-converging loop masquerade as success -- and it is also the
prerequisite for the convergence-robustness study PROJECT_STATE.md Sec 9
flags as the natural next step once this module exists: you can only
count "how often does this converge" if converged and exhausted runs are
told apart cleanly in the ledger.

THE LEDGER IS WRITTEN HERE -- faithful, not flattering. Every iteration,
whatever actually happened (including an ill-posed config, a rejected
config, or a Validator LLM whose stated verdict disagreed with the
mechanical one) is appended to AgenticRun.entries as-is. Nothing is
dropped, filtered, or summarized away -- the whole value of the ledger as
this project's primary artifact is that it tells the true story of the
search, messy iterations included, not a cleaned-up version of it.
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

MAX_ITERATIONS = 15  # PROJECT_STATE.md Sec 4/Sec 6 -- hard stop, PI's decision

ProposeFn = Callable[[list], OptimizerProposal]
RunPipelineFn = Callable[[PipelineConfig], PipelineResult]
ValidateFn = Callable[[PipelineResult], ValidatorDecision]


def decide_next_action(
    verdict: Literal["ACCEPT", "REJECT"], iteration: int, max_iterations: int
) -> Literal["continue", "stop_accepted", "stop_iteration_cap_reached"]:
    """
    The entire stop decision, as a pure function of (verdict, iteration,
    max_iterations) -- no LLM, no history, nothing else consulted. ACCEPT
    always wins, even exactly at the iteration cap: success takes
    priority over exhaustion when both conditions coincide on the same
    iteration, since the run DID converge there.
    """
    if verdict == "ACCEPT":
        return "stop_accepted"
    if iteration >= max_iterations:
        return "stop_iteration_cap_reached"
    return "continue"


def _orchestrator_note_for(next_action: str) -> str:
    """Plain-text routing note recorded on the ledger entry -- part of the
    "read top to bottom as a narrative" design from agents/schemas.py."""
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
    The Orchestrator's loop. Takes three plain callables rather than the
    real Optimizer/tool/Validator directly, so this function -- the
    actual routing logic -- can be tested with fully scripted, fully
    deterministic fakes and never needs an LLM, real or fake, in its own
    test suite. See run_agentic_loop_with_real_agents() below for the
    thin adapter that wires the real agents into this same function.

    Parameters
    ----------
    propose_fn : history -> OptimizerProposal
    run_pipeline_fn : PipelineConfig -> PipelineResult
    validate_fn : PipelineResult -> ValidatorDecision
    max_iterations : int, optional

    Returns
    -------
    AgenticRun
        The complete, faithful ledger of every iteration actually run.
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
    """Package the completed history into the top-level ledger object,
    setting stop_reason/accepted_config from the LAST entry's next_action
    -- decide_next_action() guarantees the loop never ends on "continue"."""
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
    Thin adapter: builds the three closures run_agentic_loop() needs from
    the REAL Optimizer, tool, and Validator, and delegates every routing
    decision to run_agentic_loop() -- this function does no routing of
    its own, only wiring. This is the entry point scripts/
    run_phase3_agentic.py (not yet built) will call for a real run; tests
    exercise run_agentic_loop() directly with fakes instead.
    """
    def propose_fn(history: list[LedgerEntry]) -> OptimizerProposal:
        return propose_next_config(optimizer_agent, history, search_bounds)

    def run_pipeline_fn(config: PipelineConfig) -> PipelineResult:
        return run_msm_pipeline(config, trajectory, dt)

    def validate_fn(result: PipelineResult) -> ValidatorDecision:
        return validate_pipeline_result(validator_agent, result, rate_tolerance, reference_beta)

    return run_agentic_loop(propose_fn, run_pipeline_fn, validate_fn, max_iterations)
