"""
The data formats ("contracts") the three agents pass to each other, as
Pydantic models (arXiv:2510.12787 Sec 3.1 for the architecture).

An LLM returns free text; a Pydantic model turns it into a typed object
that code and tests can check. The key guarantee lives here:
ValidatorDecision.verdict is never taken from the LLM. It is recomputed
from the physics and ill-posedness checks every time a decision is built,
so an LLM writing "ACCEPT" next to a failed check cannot turn it into a
pass.

PipelineConfig holds only the settings the analysis actually exposes
(number of k-means regions, seed, lag time). There is no dimensionality-
reduction stage (TICA) to configure, because the 0-D trajectory is already
one number per frame.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """
    Shared base for every schema here. `extra="forbid"` makes any
    unexpected field (for example, one an LLM invented) an error instead of
    being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(ContractModel):
    """
    One proposal for how to analyze the fixed reference trajectory
    (pipeline/cluster.py and pipeline/msm.py). Only analysis settings
    appear here: physics parameters (beta, dt, A) are never the agents' to
    change (CLAUDE.md, hard boundary 2).
    """

    n_clusters: int = Field(
        gt=0, description="Number of k-means microstates (pipeline.cluster.cluster_trajectory)."
    )
    cluster_seed: int = Field(
        description="Random seed for the k-means fit, for reproducibility across iterations."
    )
    msm_lagtime: int = Field(
        gt=0, description="Lag time, in frames, at which the MSM is built (pipeline.msm.build_msm)."
    )


class PipelineResult(ContractModel):
    """
    What the pipeline tool (agents/tools.py) measured for one config. Only
    code writes these numbers, never an LLM. The measurements are optional
    so that a failed run can still be recorded and reasoned about, with
    `error` saying why.
    """

    config: PipelineConfig
    error: Optional[str] = Field(
        default=None,
        description="Why the run failed (for example, a lag/cluster combination the "
                    "MSM estimator rejects); the measurements are then None.",
    )

    n_macrostates_recovered: Optional[int] = None
    macrostate_populations: Optional[list[float]] = None
    macrostate_well_identity: Optional[list[Literal["x_plus", "x_minus"]]] = Field(
        default=None,
        description="Which physical well each entry of macrostate_populations belongs to "
                    "(same order). PCCA+'s 0/1 labels are arbitrary; a tilted-well "
                    "population check needs this mapping.",
    )
    slowest_implied_timescale: Optional[float] = None
    timescale_separation: Optional[float] = Field(
        default=None,
        description="t_2 / t_3, the evidence for two metastable states "
                    "(pipeline.msm.timescale_separation).",
    )
    relaxation_rate_mean: Optional[float] = None
    vamp2_score: Optional[float] = Field(
        default=None,
        description="Cross-validated VAMP-2 score: a soft guide for the Optimizer, "
                    "comparable only at equal lag. It never decides acceptance.",
    )

    # Diagnostics for spotting ill-posed configs (Ax-Prover Appendix C),
    # recorded while the pipeline runs
    trajectory_length_frames: Optional[int] = None
    n_visited_microstates: Optional[int] = None
    min_transition_count: Optional[int] = None


class ValidatorDecision(ContractModel):
    """
    The Validator's decision on one PipelineResult. Two kinds of field:
    - Checks (two_states_recovered, rate_matches_analytical, is_ill_posed),
      computed in plain code, never by asking an LLM.
    - The LLM's own reading (llm_verdict, reasoning, suggested_change),
      kept for the ledger but never authoritative.

    `verdict` is recomputed from the checks on every construction, so an
    LLM's "ACCEPT" can never outvote a failed check. A failed run has every
    check False and is_ill_posed True.

    There is no population-ratio check yet: at Phase 3's symmetric (b=0)
    reference the expected ratio is 1, so it would not discriminate. A
    tilted (b != 0) deployment should add one, checked against
    known_answers.boltzmann_population_ratio().
    """

    two_states_recovered: bool = Field(
        description="Two metastable states: timescale_separation above "
                    "pipeline.msm.MIN_TIMESCALE_SEPARATION."
    )
    rate_matches_analytical: bool = Field(
        description="Measured relaxation rate within tolerance of the exact chain rate "
                    "(known_answers.euler_maruyama_relaxation_rate_0d) at the reference beta."
    )
    is_ill_posed: bool = Field(
        description="True if the config itself could not run (for example, a lag at least "
                    "as long as the trajectory), as opposed to running and failing physics."
    )
    ill_posedness_reasons: list[str] = Field(
        default_factory=list,
        description="Empty unless is_ill_posed is True; one short string per reason found.",
    )

    llm_verdict: Literal["ACCEPT", "REJECT"] = Field(
        description="The Validator LLM's own verdict; recorded, never authoritative "
                    "(see `verdict` and `llm_overridden`)."
    )
    reasoning: str = Field(description="The Validator LLM's interpretation of the pass/fail pattern.")
    suggested_change: Optional[str] = Field(
        default=None,
        description="What the Validator suggests trying next, if rejected. Advisory only: "
                    "never checked against physics.",
    )

    verdict: Literal["ACCEPT", "REJECT"] = "REJECT"
    llm_overridden: bool = False

    @model_validator(mode="after")
    def enforce_hard_gate(self) -> "ValidatorDecision":
        """
        Recompute `verdict` from the checks, ignoring any value passed in.
        ACCEPT needs every check to pass and the config to be well-posed.
        """
        all_hard_checks_passed = (
            self.two_states_recovered
            and self.rate_matches_analytical
            and not self.is_ill_posed
        )
        self.verdict = "ACCEPT" if all_hard_checks_passed else "REJECT"
        self.llm_overridden = self.llm_verdict != self.verdict
        return self


class OptimizerProposal(ContractModel):
    """
    The Optimizer's proposed config and its reason for it, so the ledger
    records why a config was chosen, not only its numbers.
    """

    config: PipelineConfig
    reasoning: str = Field(
        description="Why this config, given the previous iteration's result (if any)."
    )


class LedgerEntry(ContractModel):
    """
    One round of the loop, with fields in reading order: what was
    proposed, what it measured, what the checks said, what happens next.
    The ledger can be read top to bottom like a transcript.
    """

    iteration: int = Field(ge=1)
    proposal: OptimizerProposal
    result: PipelineResult
    decision: ValidatorDecision
    next_action: Literal["continue", "stop_accepted", "stop_iteration_cap_reached"]
    orchestrator_note: Optional[str] = Field(
        default=None,
        description="The Orchestrator's short note on next_action, such as why it stopped.",
    )


class AgenticRun(ContractModel):
    """
    The whole run, written to results/ledger.json as one self-contained
    record (the "JSON ledger").
    """

    max_iterations: int = Field(gt=0)
    entries: list[LedgerEntry] = Field(default_factory=list)
    stop_reason: Optional[Literal["validator_accepted", "iteration_cap_reached"]] = None
    accepted_config: Optional[PipelineConfig] = None
