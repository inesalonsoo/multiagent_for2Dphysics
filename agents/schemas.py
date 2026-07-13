"""
Pydantic contracts for the Phase 3 agentic loop (Optimizer / Validator /
Orchestrator, mirroring Ax-Prover, arXiv:2510.12787 Sec 3.1 -- see
PROJECT_STATE.md Sec 6/7/9 for the full architecture and the reasoning
behind each design choice below).

These schemas exist to make agent output CHECKABLE. An LLM call by itself
returns free text; a Pydantic model turns that into a typed object with
fields a test can assert on. The single most important property enforced
here: ValidatorDecision.verdict is NEVER taken directly from the LLM. It
is recomputed, every time a ValidatorDecision is built, from the hard
Boolean physics/ill-posedness checks alone -- so a Validator LLM call that
writes "ACCEPT" next to a failing check cannot make the pipeline treat a
physics failure as a pass. See ValidatorDecision below: that guarantee
lives in the schema itself (a model_validator that recomputes `verdict`
unconditionally), not in a convention some future caller has to remember
to follow.

Scope note on PipelineConfig: PROJECT_STATE.md Sec 6 describes the
Optimizer's proposal generically as "(tica_lag, msm_lag, n_clusters)",
mirroring a typical deeptime pipeline. This project's actual 0-D pipeline
has no TICA stage: pipeline/features.py is a trivial reshape of the 1-D
trajectory, and pipeline/reduce.py was confirmed unnecessary and never
built (PROJECT_STATE.md Sec 7, module 1.5). PipelineConfig below is
adapted to the fields the REAL, already-built pipeline exposes --
n_clusters, cluster_seed, msm_lagtime -- with no tica_lag field, since
there is nothing for it to configure.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """
    Shared base for every schema in this file. `extra="forbid"` means a
    malformed or hallucinated extra field from an LLM's structured output
    raises immediately at parse time, instead of being silently dropped --
    "nothing is a black box" applies to what agents hand each other, not
    just to our own code.
    """

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(ContractModel):
    """
    One proposal for how to run the analysis pipeline (pipeline/cluster.py
    + pipeline/msm.py) on the fixed reference trajectory the loop reuses
    across iterations. Physics parameters (beta, dt, A, ...) are never
    part of this config -- CLAUDE.md HARD BOUNDARY 2 forbids an agent from
    touching them; only analysis-side knobs are tunable here.
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
    The deterministic output of agents/tools.py's run_msm_pipeline tool
    (module 3.2, not yet built) for one PipelineConfig. Every numeric
    field here is a real measured number -- no LLM ever writes to this
    model. The measurement fields are Optional (default None) so a FAILED
    run (e.g. deeptime itself rejecting a degenerate lag/cluster
    combination) can still be represented and reasoned about instead of
    crashing the loop -- PROJECT_STATE.md Sec 6 requires the Optimizer to
    "reason about the previous result -- including parsing errors -- and
    adjust."
    """

    config: PipelineConfig
    error: Optional[str] = Field(
        default=None,
        description="Set, with the measurement fields below left None, if the tool call "
                    "itself raised -- e.g. an ill-posed lag/cluster combination deeptime rejects.",
    )

    n_macrostates_recovered: Optional[int] = None
    macrostate_populations: Optional[list[float]] = None
    macrostate_well_identity: Optional[list[Literal["x_plus", "x_minus"]]] = Field(
        default=None,
        description="For each entry in macrostate_populations (same order, same index), "
                    "which physical well it corresponds to. None unless exactly 2 "
                    "macrostates were recovered. PCCA+'s 0/1 macrostate labels are "
                    "otherwise physically arbitrary -- irrelevant for Phase 3's symmetric "
                    "(b=0) reference (the two wells are interchangeable), but this is the "
                    "Phase 4 prerequisite (PROJECT_STATE.md Sec 9): the tilted-potential "
                    "Boltzmann-ratio check needs to know WHICH measured population belongs "
                    "to WHICH well, which this field, not the raw PCCA+ label, provides.",
    )
    slowest_implied_timescale: Optional[float] = None
    relaxation_rate_mean: Optional[float] = None
    vamp2_score: Optional[float] = Field(
        default=None,
        description="Cross-validated VAMP-2 score. A SOFT guide for the Optimizer's own "
                    "navigation between candidate configs (agents/optimizer.py's system "
                    "prompt) -- it does NOT decide acceptance. Acceptance is decided "
                    "entirely by the Validator's hard physics gates, which are blind to "
                    "this number.",
    )

    # Cheapest computed once, here, while the pipeline is already running,
    # rather than re-derived later from raw arrays by the Validator.
    # Feeds ill-posedness detection (Sec 7, 3.4.1 / Ax-Prover Appendix C).
    trajectory_length_frames: Optional[int] = None
    n_visited_microstates: Optional[int] = None
    min_transition_count: Optional[int] = None


class ValidatorDecision(ContractModel):
    """
    The Validator's output for one PipelineResult. Two families of field,
    kept structurally separate on purpose:

    - Hard, deterministic checks (two_states_recovered, rate_matches_
      analytical, is_ill_posed) are Booleans grounded in
      physics/known_answers.py (or, for is_ill_posed, in PipelineResult's
      own diagnostics). Whoever builds a ValidatorDecision computes these
      in plain Python -- never asks an LLM for a yes/no on physics.
    - llm_verdict and reasoning are the Validator LLM's own read of the
      pattern of pass/fail, kept for the ledger's narrative value, but
      NEVER authoritative.

    `verdict` is not set by the caller in any way that survives: the
    model_validator below recomputes it from the hard checks alone on
    every construction, so it is IMPOSSIBLE to build a ValidatorDecision
    where an LLM's "ACCEPT" outvotes a failing physics check. If the
    underlying PipelineResult itself failed (error is not None), the
    caller is expected to set every hard check to False and is_ill_posed
    to True -- a failed run can never be mechanically accepted.

    DORMANT, NOT DELETED: there is deliberately no boltzmann_ratio_matches_
    analytical field here. For Phase 3's symmetric (b=0) reference
    trajectory, known_answers.boltzmann_population_ratio() is trivially
    ~1 -- a non-discriminating check, not a useful gate. Phase 4 runs the
    same pipeline on a TILTED potential, where that ratio becomes the
    PRIMARY discriminating known-answer (population imbalance =
    exp(-beta*deltaF), see PROJECT_STATE.md Sec 3). Reactivate this field
    when building Phase 4's agentic deployment -- do not let a tilted run
    silently proceed without its most important physics check just
    because it was dropped here for an unrelated reason three phases
    earlier. See PROJECT_STATE.md Sec 9 (module 3.1 entry) for this same
    marker.
    """

    two_states_recovered: bool = Field(
        description="known_answers.expected_number_of_states() == 2 macrostates recovered."
    )
    rate_matches_analytical: bool = Field(
        description="Measured relaxation rate within tolerance of 2*known_answers."
                    "eyring_kramers_rate_0d() at the loop's fixed reference beta."
    )
    is_ill_posed: bool = Field(
        description="True if the config itself was degenerate (lag >= trajectory length, "
                    "n_clusters > distinct visited microstates, too few transition counts) -- "
                    "distinct from a well-posed config that simply failed a physics check."
    )
    ill_posedness_reasons: list[str] = Field(
        default_factory=list,
        description="Empty unless is_ill_posed is True; one short string per reason found.",
    )

    llm_verdict: Literal["ACCEPT", "REJECT"] = Field(
        description="The Validator LLM's own verdict from reading the checks above -- "
                    "recorded for the ledger's narrative, but never authoritative; see "
                    "`verdict` and `llm_overridden`."
    )
    reasoning: str = Field(description="The Validator LLM's interpretation of the pass/fail pattern.")
    suggested_change: Optional[str] = Field(
        default=None,
        description="What the Validator suggests the Optimizer try next, if rejected -- "
                    "ADVISORY ONLY, never checked against physics (it can't be: it's a "
                    "natural-language hint about what to try, the same demonstrated-not-"
                    "proven status as the Optimizer's own proposals). `verdict` above is "
                    "the only field on this model that carries a hard guarantee.",
    )

    verdict: Literal["ACCEPT", "REJECT"] = "REJECT"
    llm_overridden: bool = False

    @model_validator(mode="after")
    def enforce_hard_gate(self) -> "ValidatorDecision":
        """
        Recompute `verdict` from the hard Boolean checks alone, ignoring
        whatever value was passed in for it. ACCEPT requires every hard
        check to pass AND the config to be well-posed -- a single False
        anywhere in that chain forces REJECT, full stop.
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
    The Optimizer's proposed config, plus its own reasoning for proposing
    it -- typically a reaction to the previous iteration's PipelineResult
    and ValidatorDecision. Kept as its own model, not a bare
    PipelineConfig, so the ledger records WHY a config was chosen, not
    just its numbers.
    """

    config: PipelineConfig
    reasoning: str = Field(
        description="Why this config, given the previous iteration's result (if any)."
    )


class LedgerEntry(ContractModel):
    """
    One full round of the Optimizer/Validator debate, with fields in the
    exact order a reader should follow them: what was proposed, what
    running it produced, what the checks said, what was decided, and what
    happens next. This ordering is deliberate -- someone should be able to
    read results/ledger.json top to bottom and follow the "debate" without
    a decoder (PROJECT_STATE.md Sec 6's "auditable record" framing).
    """

    iteration: int = Field(ge=1)
    proposal: OptimizerProposal
    result: PipelineResult
    decision: ValidatorDecision
    next_action: Literal["continue", "stop_accepted", "stop_iteration_cap_reached"]
    orchestrator_note: Optional[str] = Field(
        default=None,
        description="Free-text context from the Orchestrator for next_action -- e.g. why it "
                    "stopped, or what it told the Optimizer to try differently next.",
    )


class AgenticRun(ContractModel):
    """
    The top-level object written whole to results/ledger.json -- the JSON
    State Ledger (PROJECT_STATE.md Sec 6). One self-contained transcript
    of an entire run, so the outcome never has to be reconstructed by
    cross-referencing other files.
    """

    max_iterations: int = Field(gt=0)
    entries: list[LedgerEntry] = Field(default_factory=list)
    stop_reason: Optional[Literal["validator_accepted", "iteration_cap_reached"]] = None
    accepted_config: Optional[PipelineConfig] = None
