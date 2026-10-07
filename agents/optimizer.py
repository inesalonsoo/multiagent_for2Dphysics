"""
The Optimizer agent (the "Prover" in Ax-Prover's design, arXiv:2510.12787
Sec 3.1). One step per call: given the history of settings tried so far and
what each one measured, an LLM proposes the next PipelineConfig (number of
k-means regions and MSM lag time). Looping and stopping belong to the
Orchestrator.

What tests can and cannot check: there is no known "best" config and the
LLM is not deterministic, so tests/test_optimizer.py never asserts that a
proposal is good (the Validator judges that). With scripted fake LLMs it
checks the interface: malformed output is rejected, failures reach the
prompt, and a failed config is not proposed again.

Prompt design:
- The Optimizer chooses where to sample next; it is never asked to predict
  a score, because only the pipeline tool can measure one.
- It is given the valid ranges and the physics that bounds a sensible lag
  time, but not the answer. (When the prompt once contained the lag then
  believed converged, every run proposed it and no real search happened.)
- VAMP-2, a model-quality score, is a soft guide that is only comparable at
  equal lag time. Acceptance is decided by the Validator's physics checks.
"""

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from agents.schemas import LedgerEntry, OptimizerProposal

logger = logging.getLogger(__name__)

# pydantic-ai needs the "anthropic:" prefix; a bare model name fails with "Unknown model"
OPTIMIZER_MODEL = "anthropic:claude-sonnet-5"

OPTIMIZER_SYSTEM_PROMPT = """
You are the Optimizer in a three-agent verification loop (Orchestrator /
Optimizer / Validator) analyzing a stochastic double-well trajectory with
a Markov State Model pipeline.

Your job: given the history of configs tried so far and their real,
measured results, propose the NEXT PipelineConfig to try.

Two different things are in play, and they are NOT the same:
- The cross-validated VAMP-2 score is a SOFT GUIDE for comparing configs
  AT THE SAME LAG TIME: there, a higher score suggests the microstates
  capture the slow dynamics better. Scores at different lag times are not
  comparable: VAMP-2 falls as the lag grows for any model, so a lower
  score at a longer lag is not evidence of a worse config.
- Whether a config is ACCEPTED is decided entirely by two hard physics
  gates -- two_states_recovered and rate_matches_analytical -- computed
  independently in Python against known analytical physics, and reported
  to you as already-decided Booleans in the Validator's feedback in the
  history below. VAMP-2 is BLIND to these gates and does not predict
  them: a config can score well on VAMP-2 and still be rejected for
  wrong physics, or score modestly and still be accepted for right
  physics. Never treat VAMP-2 as a proxy for correctness -- navigate
  with it, but let the physics gates decide acceptance.

Rules:
- You choose WHERE TO SAMPLE NEXT. Never predict or state what score a
  config will achieve -- you do not know that until the deterministic
  pipeline tool actually runs it. Reason only about scores you have
  already been shown in the history below.
- If the most recent result in the history FAILED (has an error), do NOT
  repeat that exact config. Propose something that addresses the stated
  error -- move away from whichever bound was violated.
- If the most recent result was REJECTED on physics grounds (the tool ran
  fine, but a hard gate was False), use the Validator's reasoning and
  suggested_change to inform your next proposal -- don't just chase a
  higher VAMP-2 score in a direction the physics gates have already
  ruled out.
- Stay inside the valid ranges given to you below. A config outside them
  will simply be rejected by the tool as ill-posed, wasting an iteration.
- Give a short, concrete reason for your choice, referencing the history.
"""


@dataclass
class SearchBounds:
    """
    Valid ranges for the next proposal, taken from the reference
    trajectory. They tell the Optimizer what is allowed, never what is
    correct. A plain dataclass, not a Pydantic contract: it only shapes
    the prompt and is never stored in the ledger.
    """

    trajectory_length_frames: int
    max_n_clusters: int
    min_n_clusters: int = 2
    min_msm_lagtime: int = 1

    def as_prompt_text(self) -> str:
        """The bounds as prompt text: the valid ranges, plus the physics
        that limits a sensible lag time (not a value for it)."""
        max_lagtime = self.trajectory_length_frames - 1
        return (
            f"n_clusters: integer in [{self.min_n_clusters}, {self.max_n_clusters}].\n"
            f"msm_lagtime: integer in [{self.min_msm_lagtime}, {max_lagtime}] (must be "
            f"strictly less than the trajectory length, {self.trajectory_length_frames} "
            f"frames). No specific value is given to you -- reason about it from first "
            f"principles: too SHORT a lag means the microstate dynamics have not yet "
            f"lost memory of their sub-lag history, which biases the extracted rate; "
            f"too LONG a lag leaves too few independent lag-multiples in the "
            f"trajectory for reliable transition-count statistics. The short-lag "
            f"bias shrinks slowly as the lag grows, so watch how the measured rate "
            f"changes between lags. The exact value is yours to find, and to revise "
            f"based on what the Validator's feedback below tells you about your "
            f"previous attempt.\n"
            f"cluster_seed: any integer (only affects k-means initialization, not physics)."
        )


def build_optimizer_agent(model: Any = None) -> Agent[None, OptimizerProposal]:
    """
    Create the Optimizer's LLM agent. Tests pass a scripted fake `model`
    so they never call a real API.
    """
    return Agent(
        model or OPTIMIZER_MODEL,
        output_type=OptimizerProposal,
        system_prompt=OPTIMIZER_SYSTEM_PROMPT,
    )


def _format_history_for_prompt(history: list[LedgerEntry]) -> str:
    """
    Write past iterations as plain text for the prompt: each proposal,
    what it measured (or the error it hit), and the Validator's verdict.
    """
    if not history:
        return "No iterations yet -- this is the first proposal."

    lines = []
    for entry in history:
        config = entry.proposal.config
        result = entry.result
        lines.append(
            f"Iteration {entry.iteration}: n_clusters={config.n_clusters}, "
            f"cluster_seed={config.cluster_seed}, msm_lagtime={config.msm_lagtime}"
        )
        if result.error is not None:
            lines.append(f"  -> FAILED: {result.error}")
        else:
            lines.append(
                f"  -> vamp2_score={result.vamp2_score} (soft guide only), "
                f"n_macrostates_recovered={result.n_macrostates_recovered}, "
                f"relaxation_rate_mean={result.relaxation_rate_mean}"
            )
            # The two physics checks that decided the verdict, stated explicitly
            lines.append(
                f"  Physics gates: two_states_recovered={entry.decision.two_states_recovered}, "
                f"rate_matches_analytical={entry.decision.rate_matches_analytical}"
            )
        lines.append(
            f"  Validator verdict: {entry.decision.verdict} "
            f"(llm said {entry.decision.llm_verdict}; overridden={entry.decision.llm_overridden}) "
            f"-- {entry.decision.reasoning}"
        )
        if entry.decision.suggested_change is not None:
            # The Validator's suggestion for what to try next (advisory only)
            lines.append(f"  Validator's suggested_change: {entry.decision.suggested_change}")
    return "\n".join(lines)


def _build_proposal_prompt(history: list[LedgerEntry], search_bounds: SearchBounds) -> str:
    """Assemble the full user prompt: history first, then the bounds the
    next proposal must respect."""
    return (
        f"History so far:\n{_format_history_for_prompt(history)}\n\n"
        f"Valid ranges for your next proposal:\n{search_bounds.as_prompt_text()}\n\n"
        f"Propose the next PipelineConfig to try."
    )


def propose_next_config(
    optimizer_agent: Agent[None, OptimizerProposal],
    history: list[LedgerEntry],
    search_bounds: SearchBounds,
) -> OptimizerProposal:
    """
    The Optimizer's single step: from everything measured so far, propose
    the next config. The Orchestrator decides whether to call it again.
    """
    prompt = _build_proposal_prompt(history, search_bounds)
    logger.info("Optimizer prompt (iteration %d):\n%s", len(history) + 1, prompt)
    result = optimizer_agent.run_sync(prompt)
    return result.output
