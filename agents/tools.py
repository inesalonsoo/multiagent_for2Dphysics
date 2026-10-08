"""
The analysis tool the loop runs on each proposal. It is ordinary code, not
an LLM: it
runs the pipeline (k-means regions, then the MSM) on the reference
trajectory with the settings in a PipelineConfig and reports what it
measures. It sits between the verified analysis code and the LLM agents,
so it guarantees two things (both tested):

1. Deterministic: the same config, trajectory and time step always give
   the same result. The only randomness (k-means) is seeded from the
   config.
2. Never crashes on a bad config. Problems such as a lag longer than the
   trajectory, unvisited regions, too few regions to measure t_3, or the
   MSM estimator rejecting the data are logged in full and returned as a
   PipelineResult with `error` set, so the agents can reason about them.

It does not judge the physics: comparing with the exact answers is the
Validator's job.
"""

import logging

import numpy as np
from deeptime.markov import TransitionCountEstimator

from agents.schemas import PipelineConfig, PipelineResult
from pipeline.cluster import cluster_trajectory
from pipeline.features import compute_features
from pipeline.msm import build_msm, recover_two_macrostates, timescale_separation

logger = logging.getLogger(__name__)


def _pre_check_lagtime(lagtime, trajectory_length_frames):
    """
    Return an error message if the lag time is not shorter than the
    trajectory (then no transitions can be observed), else None. Checked
    first because it is free.
    """
    if lagtime >= trajectory_length_frames:
        return (
            f"msm_lagtime ({lagtime}) >= trajectory_length_frames "
            f"({trajectory_length_frames}) -- no transitions are observable at this lag."
        )
    return None


def _cluster_and_check_coverage(trajectory, n_clusters, cluster_seed):
    """
    Cluster the trajectory into k-means regions (seeded from cluster_seed)
    and check that every requested region was visited at least once.

    Returns
    -------
    discrete_trajectory : np.ndarray or None (None on any failure)
    cluster_centers : np.ndarray or None (None on any failure)
        Position of each region's center, shape (n_clusters, 1). Kept so
        each macrostate can be matched to its physical well.
    n_visited_microstates : int or None (None only if clustering itself raised)
    error_message : str or None (None on success)
    """
    try:
        features = compute_features(trajectory)
        discrete_trajectory, cluster_centers = cluster_trajectory(
            features, n_clusters=n_clusters, seed=cluster_seed
        )
    except Exception as exc:
        message = f"clustering failed for n_clusters={n_clusters}: {exc}"
        logger.error(message, exc_info=True)
        return None, None, None, message

    n_visited_microstates = int(len(np.unique(discrete_trajectory)))
    if n_visited_microstates < n_clusters:
        # Ill-posed config (Ax-Prover Appendix C): more regions requested than
        # the data fills. Not a physics failure.
        message = (
            f"n_clusters={n_clusters} requested but only "
            f"{n_visited_microstates} microstates were actually visited."
        )
        logger.error(message)
        return None, None, n_visited_microstates, message

    return discrete_trajectory, cluster_centers, n_visited_microstates, None


def _estimate_msm_and_macrostates(discrete_trajectory, msm_lagtime):
    """
    Build the transition counts, the MSM, and its 2-macrostate PCCA+
    grouping, all at msm_lagtime. The MSM estimator can reject degenerate
    data (for example disconnected regions); that is caught and returned
    as an error. An MSM with fewer than 3 states is also an error: it has
    no t_3, so the timescale separation check cannot run.

    Returns
    -------
    msm, pcca_model : deeptime model objects, or None, None on failure
    min_transition_count : int or None
        Smallest number of observed transitions out of any region: the
        region whose statistics are weakest.
    error_message : str or None
    """
    try:
        count_matrix = (
            TransitionCountEstimator(lagtime=msm_lagtime, count_mode="sliding")
            .fit(discrete_trajectory)
            .fetch_model()
            .count_matrix
        )
        min_transition_count = int(count_matrix.sum(axis=1).min())
        msm, pcca_model = recover_two_macrostates(discrete_trajectory, msm_lagtime)
    except Exception as exc:
        message = f"MSM/PCCA+ estimation failed at msm_lagtime={msm_lagtime}: {exc}"
        logger.error(message, exc_info=True)
        return None, None, None, message

    # Each state after the first adds one implied timescale, so t_3 needs 3 states
    if msm.n_states < 3:
        message = (
            f"the MSM has only {msm.n_states} connected microstates, "
            f"but the timescale separation check needs at least 3."
        )
        logger.error(message)
        return None, None, None, message

    return msm, pcca_model, min_transition_count, None


def _classify_well_identity(cluster_centers, assignments):
    """
    Match each PCCA+ macrostate label (0, 1) to its physical well: the
    label whose region centers sit at positive x on average is the x ~ +1
    well ("x_plus"), the other is "x_minus". The labels themselves are
    arbitrary; a tilted-well population check needs this mapping. Order
    matches pcca_model.coarse_grained_stationary_probability.

    Parameters
    ----------
    cluster_centers : np.ndarray, shape (n_clusters, 1)
    assignments : np.ndarray, shape (n_clusters,)
        pcca_model.assignments: the macrostate label of each region.

    Returns
    -------
    list of "x_plus" or "x_minus", one per macrostate label, in label order.
    """
    well_identity = []
    for label in sorted(np.unique(assignments)):
        centers_in_macrostate = cluster_centers[assignments == label]
        mean_position = float(centers_in_macrostate.mean())
        well_identity.append("x_plus" if mean_position > 0 else "x_minus")
    return well_identity


def _cross_validated_vamp2_score(discrete_trajectory, msm_lagtime):
    """
    VAMP-2 score with two-fold cross-validation: fit an MSM on the first
    half of the trajectory and score it on the second half. VAMP-2 measures
    how well the model captures the slow dynamics; it guides the
    Optimizer, is comparable only at equal lag, and never decides
    acceptance.

    Returns None (logged) if scoring fails; the rest of the result is
    still valid.
    """
    halfway_index = len(discrete_trajectory) // 2
    train_trajectory = discrete_trajectory[:halfway_index]
    test_trajectory = discrete_trajectory[halfway_index:]

    try:
        train_msm = build_msm(train_trajectory, msm_lagtime)
        vamp2_score = float(train_msm.score(dtrajs=test_trajectory, r=2))
    except Exception as exc:
        message = f"cross-validated VAMP-2 scoring failed at msm_lagtime={msm_lagtime}: {exc}"
        logger.error(message, exc_info=True)
        return None

    return vamp2_score


def run_msm_pipeline(config: PipelineConfig, trajectory: np.ndarray, dt: float) -> PipelineResult:
    """
    Run the analysis (clustering, MSM, PCCA+) on `trajectory` with the
    settings in `config` and return a PipelineResult. Deterministic and
    never raises (see the module docstring).

    Parameters
    ----------
    config : agents.schemas.PipelineConfig
        n_clusters, cluster_seed, msm_lagtime to run with.
    trajectory : np.ndarray
        The loop's fixed reference trajectory, made once and reused. This
        tool never changes physics parameters.
    dt : float
        Time step of `trajectory`, to convert the MSM's timescale from
        frames to time.

    Returns
    -------
    PipelineResult
    """
    trajectory_length_frames = len(trajectory)

    lag_error = _pre_check_lagtime(config.msm_lagtime, trajectory_length_frames)
    if lag_error is not None:
        return PipelineResult(
            config=config, error=lag_error, trajectory_length_frames=trajectory_length_frames
        )

    discrete_trajectory, cluster_centers, n_visited_microstates, cluster_error = (
        _cluster_and_check_coverage(trajectory, config.n_clusters, config.cluster_seed)
    )
    if cluster_error is not None:
        return PipelineResult(
            config=config, error=cluster_error, trajectory_length_frames=trajectory_length_frames,
            n_visited_microstates=n_visited_microstates,
        )

    msm, pcca_model, min_transition_count, msm_error = _estimate_msm_and_macrostates(
        discrete_trajectory, config.msm_lagtime
    )
    if msm_error is not None:
        return PipelineResult(
            config=config, error=msm_error, trajectory_length_frames=trajectory_length_frames,
            n_visited_microstates=n_visited_microstates,
        )

    # PCCA+ labels actually used. Informational only: PCCA+ returns as many
    # sets as asked, so the Validator gates on timescale_separation instead.
    n_macrostates_recovered = int(len(np.unique(pcca_model.assignments)))
    macrostate_populations = (
        pcca_model.coarse_grained_stationary_probability.tolist()
        if n_macrostates_recovered == 2 else None
    )
    # Which physical well each macrostate is (same order as the populations)
    macrostate_well_identity = (
        _classify_well_identity(cluster_centers, pcca_model.assignments)
        if n_macrostates_recovered == 2 else None
    )
    slowest_implied_timescale = float(msm.timescales()[0])
    separation = float(timescale_separation(msm))
    relaxation_rate_mean = 1.0 / (slowest_implied_timescale * dt)
    vamp2_score = _cross_validated_vamp2_score(discrete_trajectory, config.msm_lagtime)

    return PipelineResult(
        config=config,
        n_macrostates_recovered=n_macrostates_recovered,
        macrostate_populations=macrostate_populations,
        macrostate_well_identity=macrostate_well_identity,
        slowest_implied_timescale=slowest_implied_timescale,
        timescale_separation=separation,
        relaxation_rate_mean=relaxation_rate_mean,
        vamp2_score=vamp2_score,
        trajectory_length_frames=trajectory_length_frames,
        n_visited_microstates=n_visited_microstates,
        min_transition_count=min_transition_count,
    )
