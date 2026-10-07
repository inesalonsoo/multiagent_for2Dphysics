"""
Tests for agents/tools.py's run_msm_pipeline, the analysis tool the agents
call. Two guarantees are checked: the same inputs always give the same
result, and a bad config returns a flagged result instead of crashing.
"""

import numpy as np

from agents.schemas import PipelineConfig
from agents.tools import run_msm_pipeline
from physics.known_answers import euler_maruyama_relaxation_rate_0d
from physics.simulate_0d import run_trajectory_0d

DT = 0.01
# beta=5 test trajectory. Lag 20 is short (biased) but fine here: these tests
# check determinism and structure, not accuracy.
_TRAJECTORY = run_trajectory_0d(n_steps=750_000, seed=7, beta=5.0, dt=DT)
_WELL_POSED_CONFIG = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=20)


def test_run_msm_pipeline_is_deterministic_given_identical_inputs():
    """Same config, trajectory and dt must give identical results, so a
    config can be replayed without any API call."""
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
    The two macrostates of a real double-well trajectory must map to one
    x_plus and one x_minus well, in the same order as the populations.
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


def test_run_msm_pipeline_can_overestimate_the_rate_at_a_too_short_lag():
    """
    A lag that is far too short makes the MSM underestimate the slowest
    timescale, so it overestimates the rate (rate = 1/timescale). At
    msm_lagtime=1 the measured rate is about 1.8 times the exact value: a
    config that runs fine but that the Validator must reject on physics.
    """
    config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=1)
    analytical_rate = euler_maruyama_relaxation_rate_0d(beta=5.0, dt=DT)

    result = run_msm_pipeline(config, _TRAJECTORY, DT)

    assert result.error is None
    assert result.n_macrostates_recovered == 2
    assert result.relaxation_rate_mean > analytical_rate * 1.10, (
        "expected a real overestimate at this deliberately too-short lag"
    )


def test_run_msm_pipeline_reports_lagtime_ill_posedness_without_raising():
    """A lag at least as long as the trajectory must return a flagged
    result, not raise."""
    tiny_trajectory = _TRAJECTORY[:100]
    degenerate_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=200)

    result = run_msm_pipeline(degenerate_config, tiny_trajectory, DT)

    assert result.error is not None
    assert "msm_lagtime" in result.error
    assert result.n_macrostates_recovered is None
    assert result.relaxation_rate_mean is None
    assert result.trajectory_length_frames == 100


def test_run_msm_pipeline_reports_clustering_ill_posedness_without_raising():
    """Far more clusters than a short trajectory can fill must return a
    flagged result, not an exception from k-means."""
    tiny_trajectory = _TRAJECTORY[:10]
    degenerate_config = PipelineConfig(n_clusters=50, cluster_seed=42, msm_lagtime=2)

    result = run_msm_pipeline(degenerate_config, tiny_trajectory, DT)

    assert result.error is not None
    assert result.n_macrostates_recovered is None
    assert result.relaxation_rate_mean is None
