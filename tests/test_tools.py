"""
Known-answer tests for agents/tools.py's run_msm_pipeline -- the
deterministic tool that is the seam between verified physics and LLM
reasoning. Two properties matter most and are tested explicitly, per
PROJECT_STATE.md Sec 7 module 3.2: it is PURE AND DETERMINISTIC given
(config, trajectory, dt), and it NEVER RAISES on an ill-posed config
(returns a PipelineResult with `error` set instead).
"""

import numpy as np

from agents.schemas import PipelineConfig
from agents.tools import run_msm_pipeline
from physics.simulate_0d import run_trajectory_0d

DT = 0.01
# beta=5.0, converged msm_lagtime=20 -- reusing Phase 1's own validated
# per-beta lag (PROJECT_STATE.md Sec 9), not re-deriving it here.
_TRAJECTORY = run_trajectory_0d(n_steps=750_000, seed=7, beta=5.0, dt=DT)
_WELL_POSED_CONFIG = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20)


def test_run_msm_pipeline_is_deterministic_given_identical_inputs():
    """Same config, same trajectory, same dt -- must return field-for-field
    identical PipelineResult objects. This is what lets the loop-integrity
    tests (module 3.7) replay a config without a real API call."""
    first_result = run_msm_pipeline(_WELL_POSED_CONFIG, _TRAJECTORY, DT)
    second_result = run_msm_pipeline(_WELL_POSED_CONFIG, _TRAJECTORY, DT)

    assert first_result == second_result


def test_run_msm_pipeline_recovers_two_macrostates_on_a_well_posed_config():
    """Happy-path sanity check on a real trajectory, at Phase 1's own
    validated (n_clusters, msm_lagtime) choice."""
    result = run_msm_pipeline(_WELL_POSED_CONFIG, _TRAJECTORY, DT)

    assert result.error is None
    assert result.n_macrostates_recovered == 2
    assert result.macrostate_populations is not None
    assert len(result.macrostate_populations) == 2
    assert result.relaxation_rate_mean > 0.0
    assert result.n_visited_microstates == _WELL_POSED_CONFIG.n_clusters
    assert result.min_transition_count > 0
    assert result.trajectory_length_frames == len(_TRAJECTORY)


def test_run_msm_pipeline_reports_vamp2_score_on_a_well_posed_config():
    """The Optimizer's own optimization objective: a real number, not None,
    on a well-posed config with enough data for the train/test split."""
    result = run_msm_pipeline(_WELL_POSED_CONFIG, _TRAJECTORY, DT)

    assert result.vamp2_score is not None
    assert result.vamp2_score > 0.0


def test_run_msm_pipeline_reports_well_identity_on_a_well_posed_config():
    """
    Known-answer check: the two macrostates of a real double-well
    trajectory must map to exactly one x_plus and one x_minus well --
    the Phase 4 prerequisite (PROJECT_STATE.md Sec 9) this field exists
    for (the tilted-potential Boltzmann-ratio check needs to know WHICH
    population belongs to WHICH well; PCCA+'s 0/1 labels alone don't say).
    Same order as macrostate_populations, index-for-index.
    """
    result = run_msm_pipeline(_WELL_POSED_CONFIG, _TRAJECTORY, DT)

    assert result.macrostate_well_identity is not None
    assert len(result.macrostate_well_identity) == 2
    assert sorted(result.macrostate_well_identity) == ["x_minus", "x_plus"]


def test_run_msm_pipeline_leaves_well_identity_none_on_ill_posed_config():
    tiny_trajectory = _TRAJECTORY[:100]
    degenerate_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=200)

    result = run_msm_pipeline(degenerate_config, tiny_trajectory, DT)

    assert result.macrostate_well_identity is None


def test_run_msm_pipeline_reports_lagtime_ill_posedness_without_raising():
    """A lag time at least as long as the trajectory itself must come back
    as a structured, flagged PipelineResult -- not an exception."""
    tiny_trajectory = _TRAJECTORY[:100]
    degenerate_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=200)

    result = run_msm_pipeline(degenerate_config, tiny_trajectory, DT)

    assert result.error is not None
    assert "msm_lagtime" in result.error
    assert result.n_macrostates_recovered is None
    assert result.relaxation_rate_mean is None
    assert result.trajectory_length_frames == 100


def test_run_msm_pipeline_reports_clustering_ill_posedness_without_raising():
    """Asking for far more clusters than a short trajectory can support
    must come back as a structured, flagged PipelineResult -- not an
    exception from deeptime's KMeans."""
    tiny_trajectory = _TRAJECTORY[:10]
    degenerate_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=2)

    result = run_msm_pipeline(degenerate_config, tiny_trajectory, DT)

    assert result.error is not None
    assert result.n_macrostates_recovered is None
    assert result.relaxation_rate_mean is None
