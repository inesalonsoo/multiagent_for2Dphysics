"""
Build a Markov State Model (MSM) from a discrete (clustered) trajectory,
and check it against physics/known_answers.py's ground truth: exactly two
macrostates, recovered via PCCA+ coarse-graining of the many microstates
from pipeline/cluster.py, at a lag time where the implied timescales (ITS)
have plateaued (the standard sign that the model is genuinely Markovian
at that lag, not an artifact of a too-short lag time).
"""

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM


def build_msm(discrete_trajectory, lagtime):
    """
    Estimate a reversible maximum-likelihood MSM from a discrete
    (microstate-labeled) trajectory at a given lag time.

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Array of shape (n_frames,): microstate index per frame, e.g.
        from pipeline.cluster.cluster_trajectory().
    lagtime : int
        Lag time (in frames) at which to count transitions.

    Returns
    -------
    deeptime MSM model
        Fitted reversible MSM, with .timescales(), .pcca(), etc.
    """
    count_model = TransitionCountEstimator(
        lagtime=lagtime, count_mode="sliding",
    ).fit(discrete_trajectory).fetch_model()

    msm = MaximumLikelihoodMSM(reversible=True).fit(count_model).fetch_model()
    return msm


def implied_timescales(discrete_trajectory, lagtimes):
    """
    Compute the SLOWEST implied timescale of the MSM at each of several
    lag times (an "ITS plot"). A plateau -- the slowest timescale
    stabilizing as lag time increases -- is the standard evidence that
    the microstate dynamics are Markovian at that lag: the system has
    "forgotten" its sub-lag-time history.

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Array of shape (n_frames,): microstate index per frame.
    lagtimes : sequence of int
        Lag times (in frames) to evaluate.

    Returns
    -------
    np.ndarray
        Array of shape (len(lagtimes),): the slowest implied timescale
        at each lag time.
    """
    slowest_timescales = []
    for lagtime in lagtimes:
        msm = build_msm(discrete_trajectory, lagtime)
        timescales = msm.timescales()
        slowest_timescales.append(timescales[0])

    return np.array(slowest_timescales)


def find_converged_lagtime(discrete_trajectory, candidate_lags, plateau_tolerance=0.03):
    """
    Find the smallest lag time at which the slowest implied timescale has
    genuinely plateaued, by walking candidate_lags (must be increasing,
    e.g. a doubling sequence [10, 20, 40, 80, ...]) and returning the
    smallest lag L such that doubling to the NEXT candidate lag changes
    the slowest timescale by less than plateau_tolerance (relative).

    A loose tolerance is dangerous here: it can call a curve that is
    still visibly climbing a "plateau" (e.g. a 25% tolerance would accept
    a +12.84% step as converged, which it plainly isn't -- this is what
    went wrong the first time this project picked a lag time, see
    PROJECT_STATE.md Sec 10). plateau_tolerance=0.03 (3%) is tight enough
    to actually distinguish "still climbing" from "flat".

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Array of shape (n_frames,): microstate index per frame.
    candidate_lags : sequence of int
        Increasing lag times (in frames) to test, e.g. a doubling
        sequence. Must have at least 2 entries.
    plateau_tolerance : float, optional
        Maximum allowed relative change between successive candidate
        lags to call the smaller one "converged". Default 0.03 (3%).

    Returns
    -------
    converged_lag : int or None
        The smallest candidate lag meeting the plateau criterion, or
        None if NO candidate lag in the list satisfies it -- this is a
        real, reportable finding (not enough data to resolve the slow
        timescale at any tested lag), not an error to hide.
    timescales : list of float
        The slowest implied timescale at each candidate lag, in the same
        order as candidate_lags, for inspection or plotting.
    """
    timescales = list(implied_timescales(discrete_trajectory, candidate_lags))

    converged_lag = None
    for i in range(1, len(candidate_lags)):
        relative_change = abs(timescales[i] - timescales[i - 1]) / timescales[i - 1]
        if relative_change < plateau_tolerance:
            converged_lag = candidate_lags[i - 1]
            break

    return converged_lag, timescales


def recover_two_macrostates(discrete_trajectory, lagtime):
    """
    Coarse-grain the many microstates into exactly 2 metastable
    macrostates via PCCA+, at the given lag time.

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Array of shape (n_frames,): microstate index per frame.
    lagtime : int
        Lag time (in frames), ideally chosen where implied_timescales()
        has plateaued.

    Returns
    -------
    msm : deeptime MSM model
        The fitted microstate-level MSM (see build_msm()).
    pcca_model : deeptime.markov.PCCAModel
        The 2-macrostate coarse-graining. `.assignments` gives the
        macrostate (0 or 1) each MICROSTATE belongs to;
        `.coarse_grained_stationary_probability` gives the population of
        each of the 2 macrostates.
    """
    msm = build_msm(discrete_trajectory, lagtime)
    pcca_model = msm.pcca(n_metastable_sets=2)
    return msm, pcca_model
