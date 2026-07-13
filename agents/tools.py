"""
The deterministic tool the Optimizer calls (PROJECT_STATE.md Sec 6/7,
module 3.2) -- NOT an LLM. This is the seam between the two halves of the
project: everything this module calls into (pipeline/cluster.py,
pipeline/msm.py) is verified, deterministic physics/statistics code;
everything that calls this module (agents/optimizer.py, agents/
validator.py) is LLM reasoning. run_msm_pipeline() has to be the
trustworthy surface both sides can rely on, which means two properties,
deliberately built in and both tested below:

1. PURE AND DETERMINISTIC given (config, trajectory, dt): the same inputs
   always produce an identical PipelineResult. The only randomness in the
   whole pipeline (k-means initialization and its fitting subsample) is
   seeded from config.cluster_seed -- nothing here draws fresh randomness
   of its own. This is what lets tests/test_tools.py,
   tests/test_orchestrator.py, and tests/test_loop.py replay a config and
   compare against a known result without touching a real API.
2. NEVER RAISES on an ill-posed or failing config. A degenerate lag time,
   a clustering that never visits every requested microstate, or deeptime
   itself rejecting a degenerate count matrix are all caught and reported
   as a PipelineResult with `error` set and the measurement fields left
   None -- structured data the Optimizer/Validator can reason about, not
   a crash. Per CLAUDE.md HARD BOUNDARY 5, every except block below logs
   the full error (via the standard `logging` module) before returning;
   nothing is swallowed silently.

Scope note: run_msm_pipeline() does NOT compare anything against
physics/known_answers.py. It reports raw, real measurements only --
deciding whether they PASS a physics check is the Validator's job (module
3.4), grounded in the Booleans on agents/schemas.py's ValidatorDecision.
Keeping that judgment out of this module is what lets it stay a pure,
deterministic function agents can trust.

[2026-07-12, Phase 4 prerequisite] macrostate_well_identity is now
computed here (_classify_well_identity below), tying each PCCA+
macrostate label back to its physical well (x_plus/x_minus) via the mean
position of its constituent microstate cluster centers. This is the
"well-identity tracking" agents/validator.py's dormant Boltzmann-ratio
socket was found to need before any Phase 4 tilted-potential run --
see PROJECT_STATE.md Sec 9.
"""

import logging

import numpy as np
from deeptime.markov import TransitionCountEstimator

from agents.schemas import PipelineConfig, PipelineResult
from pipeline.cluster import cluster_trajectory
from pipeline.features import compute_features
from pipeline.msm import build_msm, recover_two_macrostates

logger = logging.getLogger(__name__)


def _pre_check_lagtime(lagtime, trajectory_length_frames):
    """
    Returns an error message if the lag time is not even shorter than the
    trajectory (no transitions are observable at all), else None. Cheap:
    catching this needs no deeptime call, so there is no reason to run
    the rest of the pipeline first only to fail later.
    """
    if lagtime >= trajectory_length_frames:
        return (
            f"msm_lagtime ({lagtime}) >= trajectory_length_frames "
            f"({trajectory_length_frames}) -- no transitions are observable at this lag."
        )
    return None


def _cluster_and_check_coverage(trajectory, n_clusters, cluster_seed):
    """
    Clusters the trajectory (pipeline.cluster.cluster_trajectory, seeded
    from cluster_seed for determinism) and checks that every requested
    microstate was actually visited by at least one frame.

    Returns
    -------
    discrete_trajectory : np.ndarray or None (None on any failure)
    cluster_centers : np.ndarray or None (None on any failure)
        Position of each microstate's centroid, shape (n_clusters, 1) --
        kept (not discarded) because well-identity tracking (see
        _classify_well_identity below) needs to know WHERE each
        microstate sits, not just which macrostate PCCA+ assigned it to.
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
        # Ax-Prover Appendix C-style ill-posedness: the config asked for
        # more microstates than the data actually populated -- a
        # degenerate clustering, not a physics failure to be judged later.
        message = (
            f"n_clusters={n_clusters} requested but only "
            f"{n_visited_microstates} microstates were actually visited."
        )
        logger.error(message)
        return None, None, n_visited_microstates, message

    return discrete_trajectory, cluster_centers, n_visited_microstates, None


def _estimate_msm_and_macrostates(discrete_trajectory, msm_lagtime):
    """
    Builds the transition count matrix (for the min-transition-count
    diagnostic), the MSM, and its 2-macrostate PCCA+ coarse-graining, all
    at msm_lagtime. deeptime itself can raise for a degenerate count
    matrix (e.g. disconnected states) -- caught here, not propagated.

    Returns
    -------
    msm, pcca_model : deeptime model objects, or None, None on failure
    min_transition_count : int or None
        Smallest total outgoing transition count of any microstate -- the
        state whose rate estimate is least statistically supported.
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

    return msm, pcca_model, min_transition_count, None


def _classify_well_identity(cluster_centers, assignments):
    """
    For each macrostate label (in the same 0, 1, ... order as
    deeptime's own pcca_model.coarse_grained_stationary_probability --
    verified directly against the installed deeptime before writing
    this: coarse_grained_stationary_probability[i] and
    cluster_centers[assignments == i] are consistently indexed by the
    same macrostate label i), classify which PHYSICAL WELL it
    corresponds to: the mean position of its constituent microstate
    cluster centers being positive means the x~+1 well ("x_plus"),
    negative means the x~-1 well ("x_minus").

    This exists because PCCA+'s 0/1 macrostate labels are otherwise
    physically arbitrary -- irrelevant for Phase 3's symmetric (b=0)
    reference, where the two wells are interchangeable, but essential
    for Phase 4's tilted deployment, where the Boltzmann population
    ratio (the primary discriminating check there, see agents/
    validator.py's dormant _check_boltzmann_ratio_matches_analytical)
    needs to know WHICH measured population belongs to WHICH well.

    Parameters
    ----------
    cluster_centers : np.ndarray, shape (n_clusters, 1)
    assignments : np.ndarray, shape (n_clusters,)
        pcca_model.assignments -- the macrostate label of each microstate.

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
    A simple two-fold cross-validated VAMP-2 score: fit an MSM on the
    first half of the discrete trajectory, score it (deeptime's own
    MarkovStateModel.score(r=2), the VAMP-2 definition) against the
    held-out second half. This is the Optimizer's optimization objective
    (PROJECT_STATE.md Sec 6) -- a measure of how well this (n_clusters,
    msm_lagtime) choice generalizes, independent of and blind to the
    Validator's physics checks.

    Returns None (logged, not raised) if scoring fails -- a missing
    optimization score should not invalidate an otherwise-valid
    PipelineResult; only the Optimizer needs it, and it can treat None as
    "try a different config."
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
    Run the analysis pipeline (clustering + MSM + PCCA+) on `trajectory`
    with the knobs in `config`, and return a PipelineResult. Pure and
    deterministic given (config, trajectory, dt); never raises -- see the
    module docstring for both guarantees and why they matter here.

    Parameters
    ----------
    config : agents.schemas.PipelineConfig
        n_clusters, cluster_seed, msm_lagtime to run with.
    trajectory : np.ndarray
        The loop's fixed reference 0-D trajectory (physics.simulate_0d.
        run_trajectory_0d output), generated once, reused across
        iterations -- this tool never generates or touches physics
        parameters (CLAUDE.md HARD BOUNDARY 2).
    dt : float
        Time step used to generate `trajectory`, needed to convert the
        MSM's implied timescale (in frames) into a rate (in 1/time).

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

    # Same diagnostic Phase 1 uses (scripts/run_phase1_benchmark.py): PCCA+
    # was ASKED for 2 macrostates, but only really recovered 2 if both
    # labels are actually used in the microstate assignment.
    n_macrostates_recovered = int(len(np.unique(pcca_model.assignments)))
    macrostate_populations = (
        pcca_model.coarse_grained_stationary_probability.tolist()
        if n_macrostates_recovered == 2 else None
    )
    # Phase 4 prerequisite (PROJECT_STATE.md Sec 9): PCCA+'s 0/1 labels are
    # otherwise physically arbitrary. Same index order as macrostate_populations.
    macrostate_well_identity = (
        _classify_well_identity(cluster_centers, pcca_model.assignments)
        if n_macrostates_recovered == 2 else None
    )
    slowest_implied_timescale = float(msm.timescales()[0])
    relaxation_rate_mean = 1.0 / (slowest_implied_timescale * dt)
    vamp2_score = _cross_validated_vamp2_score(discrete_trajectory, config.msm_lagtime)

    return PipelineResult(
        config=config,
        n_macrostates_recovered=n_macrostates_recovered,
        macrostate_populations=macrostate_populations,
        macrostate_well_identity=macrostate_well_identity,
        slowest_implied_timescale=slowest_implied_timescale,
        relaxation_rate_mean=relaxation_rate_mean,
        vamp2_score=vamp2_score,
        trajectory_length_frames=trajectory_length_frames,
        n_visited_microstates=n_visited_microstates,
        min_transition_count=min_transition_count,
    )
