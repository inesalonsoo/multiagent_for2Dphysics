"""
Optimizer Agent (PROJECT_STATE.md Sec 6/7, module 3.3) -- Ax-Prover's
Prover (arXiv:2510.12787 Sec 3.1). A SINGLE-STEP proposer: given the run's
history so far, propose the next PipelineConfig to try. No loop control
lives here -- iterating, deciding when to stop, and routing feedback back
and forth are the Orchestrator's job (module 3.5, not yet built).
Collapsing that control into this module would undo the three-agent
separation the whole architecture is built on: one agent, one
responsibility, and this one's responsibility is "propose," full stop.

WHAT IS AND ISN'T TESTABLE HERE -- read before adding a test.
There is no closed-form "optimal" PipelineConfig, and the LLM's proposal
is non-deterministic, so nothing in tests/test_optimizer.py asserts a
proposed config is GOOD. That is demonstrated in real runs, reported in
the ledger, and judged by the Validator against physics -- it cannot be a
unit-test assertion. What IS deterministically testable, with no real API
call (pydantic_ai.models.function.FunctionModel / .test.TestModel stand
in for the LLM): that a malformed structured-output response is
rejected/retried rather than corrupting the loop (pydantic-ai retries
output validation automatically; exhausting retries raises a clear
pydantic_ai.exceptions.UnexpectedModelBehavior, not a silently-wrong
value -- verified interactively against the installed pydantic-ai 2.7.0
before relying on it here); that the previous PipelineResult -- including
a FAILED one -- actually reaches the prompt; and that, given a scripted
fake LLM that reacts sensibly to a failure signal in the prompt, the next
proposal differs from the config that just failed. That last property is
the real behavioral guarantee worth locking down: an Optimizer that
re-proposes an already-failed config is stuck. In short: this module's
INTERFACE CONTRACT is verified; its REASONING QUALITY is demonstrated,
not proven -- the Phase 3 analogue of known-answer discipline, adapted to
a domain that has no known answer.

PROMPT DISCIPLINE (Ax-Prover's lesson, carried over from the tool
design, PROJECT_STATE.md Sec 9 / agents/tools.py): the Optimizer chooses
WHERE TO SAMPLE NEXT. It never predicts what VAMP-2 score a config will
get -- that would be reasoning about an outcome it didn't measure, the
exact failure mode the deterministic run_msm_pipeline tool (module 3.2)
exists to prevent. The system prompt below asks for a config and a
reason, never a score estimate; it is also handed explicit valid ranges
(SearchBounds) so it proposes inside the well-posed region instead of
guessing wildly outside it.

[2026-07-12 REDESIGN -- SearchBounds no longer hands over the converged
msm_lagtime.] The real convergence-robustness study (PROJECT_STATE.md
Sec 9) showed that an earlier version of SearchBounds.as_prompt_text()
handed the Optimizer Phase 1/2's own converged lag as "a well-motivated
starting region" -- and every one of 8 independent real runs proposed the
byte-identical config on iteration 1 as a direct result. That collapsed
the search entirely: there was only one obviously-correct answer to give,
so the Validator's gate was never exercised against a genuinely wrong
config, and the Optimizer's own reacts-to-rejection behavior (real, and
proven with fakes in tests/test_optimizer.py) never got to run under real
conditions either. SearchBounds now states only the PHYSICAL REASONING
that bounds a sensible lag time (short lags bias the rate; long lags
starve transition-count statistics) -- not the solved value -- so the
Optimizer has to actually search, and a too-short lag proposal now gets
rejected by the Validator on real physics grounds (a genuinely biased
rate falling outside Phase 2's honest tolerance band), not a rigged one.

[2026-07-12] VAMP-2 IS A SOFT GUIDE, NOT THE ACCEPTANCE CRITERION -- made
explicit in the system prompt below, not left implicit. Before this, the
system prompt said the Optimizer's job was to "maximize the cross-
validated VAMP-2 score," full stop -- true as far as it went, but it never
stated the more important fact: acceptance is decided entirely by the
Validator's two hard physics gates (two_states_recovered,
rate_matches_analytical), which are BLIND to VAMP-2 entirely. A config
can have a great VAMP-2 score and still be rejected (wrong physics), or a
middling one and still be accepted (right physics). The real convergence
study showed the Optimizer already using VAMP-2 sensibly in practice
(reasoning like "VAMP-2 declining while the rate stays flat as lag grows"
to navigate between candidate lag times) -- this update makes that
relationship a stated rule instead of something the model had to infer
correctly on its own every time.
"""

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from agents.schemas import LedgerEntry, OptimizerProposal

logger = logging.getLogger(__name__)

OPTIMIZER_MODEL = "anthropic:claude-sonnet-5"  # PROJECT_STATE.md Sec 4 -- pydantic-ai
# requires the explicit "anthropic:" provider prefix; a bare "claude-sonnet-5" fails
# infer_model() with "Unknown model" (caught interactively before any real API call).

OPTIMIZER_SYSTEM_PROMPT = """
You are the Optimizer in a three-agent verification loop (Orchestrator /
Optimizer / Validator) analyzing a stochastic double-well trajectory with
a Markov State Model pipeline.

Your job: given the history of configs tried so far and their real,
measured results, propose the NEXT PipelineConfig to try.

Two different things are in play, and they are NOT the same:
- The cross-validated VAMP-2 score is a SOFT GUIDE for navigating between
  candidate configs -- a higher score suggests a config captures the slow
  dynamics better, and you may use it to judge which direction to try
  next among configs that are otherwise physically plausible.
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
    Valid ranges for the next proposal, computed from the loop's fixed
    reference trajectory -- handed to the Optimizer as a CONSTRAINT
    (what's well-posed), not a pre-solved ANSWER (what's correct). See
    this module's docstring for why that distinction was tightened on
    2026-07-12: handing over the converged lag value directly made every
    real run propose the identical config, which meant the search never
    happened and the Validator's gate was never tested against a wrong
    answer. Deliberately NOT a Pydantic contract: this never crosses an
    agent boundary or gets written to the ledger, it only shapes the
    prompt built in this module.
    """

    trajectory_length_frames: int
    max_n_clusters: int
    min_n_clusters: int = 2
    min_msm_lagtime: int = 1

    def as_prompt_text(self) -> str:
        """Render these bounds as the constraint block of the user prompt.
        States the PHYSICAL REASONING that bounds a sensible lag time, not
        a pre-solved number -- see this class's docstring."""
        max_lagtime = self.trajectory_length_frames - 1
        return (
            f"n_clusters: integer in [{self.min_n_clusters}, {self.max_n_clusters}].\n"
            f"msm_lagtime: integer in [{self.min_msm_lagtime}, {max_lagtime}] (must be "
            f"strictly less than the trajectory length, {self.trajectory_length_frames} "
            f"frames). No specific value is given to you -- reason about it from first "
            f"principles: too SHORT a lag means the microstate dynamics have not yet "
            f"lost memory of their sub-lag history, which biases the extracted rate; "
            f"too LONG a lag leaves too few independent lag-multiples in the "
            f"trajectory for reliable transition-count statistics. A lag time for a "
            f"system like this is typically a small integer, orders of magnitude "
            f"below the trajectory length -- but the exact value is yours to find, "
            f"and to revise based on what the Validator's feedback below tells you "
            f"about your previous attempt.\n"
            f"cluster_seed: any integer (only affects k-means initialization, not physics)."
        )


def build_optimizer_agent(model: Any = None) -> Agent[None, OptimizerProposal]:
    """
    Construct the Optimizer's pydantic-ai Agent. `model` defaults to this
    project's approved agent model string (PROJECT_STATE.md Sec 4); tests
    pass a pydantic_ai.models.function.FunctionModel or .test.TestModel
    here instead, so nothing in tests/test_optimizer.py makes a real API
    call.
    """
    return Agent(
        model or OPTIMIZER_MODEL,
        output_type=OptimizerProposal,
        system_prompt=OPTIMIZER_SYSTEM_PROMPT,
    )


def _format_history_for_prompt(history: list[LedgerEntry]) -> str:
    """
    Render past iterations as plain text for the prompt: what was
    proposed, and what it actually measured (or the error it hit) -- the
    "reasons about the previous result, including parsing errors"
    requirement from PROJECT_STATE.md Sec 6, made concrete as text an LLM
    call actually receives.
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
            # The two hard gates, stated explicitly rather than left for the
            # model to infer from prose -- these, not vamp2_score, decided
            # the verdict below.
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
            # The Validator's own concrete suggestion for what to try next --
            # computed but previously never surfaced back to the Optimizer,
            # which meant real, usable feedback was going unread. Advisory
            # only (agents/schemas.py), but still real signal worth showing.
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
    The Optimizer's single step: given everything measured so far, choose
    the next config to try. No looping and no stop decision here -- the
    caller (eventually agents/orchestrator.py, module 3.5) decides what to
    do with the result and whether to call this again.
    """
    prompt = _build_proposal_prompt(history, search_bounds)
    logger.info("Optimizer prompt (iteration %d):\n%s", len(history) + 1, prompt)
    result = optimizer_agent.run_sync(prompt)
    return result.output
