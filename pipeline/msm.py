"""
Build a Markov State Model (MSM) from a discrete (clustered) trajectory and
validate it: choose a converged lag time, check Markovianity with a
Chapman-Kolmogorov test, check timescale separation, and coarse-grain the
microstates into 2 macrostates with PCCA+.
"""

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

# A trajectory spanning N slowest timescales contains about N barrier
# crossings, so its rate estimate has roughly 1/sqrt(N) relative error.
# Below 20 (about 20% error) t_2 is not considered resolved.
MIN_RELAXATIONS_PER_TRAJECTORY = 20

# Gate thresholds for "two metastable states". A single well gives t_2/t_3
# below 2 and the double well tens or more (tests/test_msm.py). Observed
# Chapman-Kolmogorov errors at a converged lag are about 0.01.
MIN_TIMESCALE_SEPARATION = 10.0
MAX_CK_ERROR = 0.05


def build_msm(discrete_trajectory, lagtime):
    """
    Estimate a reversible maximum-likelihood MSM at a given lag time.

    Reversibility (detailed balance) is imposed by the estimator. It is a
    modeling assumption here, justified because the simulated dynamics are
    at equilibrium; it is not something this MSM can test.

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Microstate index per frame, shape (n_frames,).
    lagtime : int
        Lag time in frames.

    Returns
    -------
    deeptime MSM model, with .timescales(), .pcca(), etc.
    """
    count_model = TransitionCountEstimator(
        lagtime=lagtime, count_mode="sliding",
    ).fit(discrete_trajectory).fetch_model()

    msm = MaximumLikelihoodMSM(reversible=True).fit(count_model).fetch_model()
    return msm


def implied_timescales(discrete_trajectory, lagtimes):
    """
    Slowest implied timescale t_2 (in frames) at each lag time.

    An MSM underestimates t_2 at short lags and approaches the true value
    from below as the lag grows (the variational principle).
    """
    slowest_timescales = []
    for lagtime in lagtimes:
        msm = build_msm(discrete_trajectory, lagtime)
        slowest_timescales.append(msm.timescales()[0])

    return np.array(slowest_timescales)


def choose_lagtime(discrete_trajectory, fraction_of_slowest=0.1, initial_lag=10, max_rounds=8):
    """
    Choose the lag time as a fixed fraction of the slowest implied timescale
    t_2, found self-consistently: estimate t_2 at the current lag, set
    lag = fraction * t_2, and repeat until the lag changes by less than 10%.

    Why this rule: the short-lag bias of t_2 shrinks slowly (roughly as
    1/lag), so waiting for t_2 to stop changing between lag doublings stops
    too early. A lag of about 0.1 * t_2 removes the bias to about 1% for
    this system while still resolving t_2.

    Returns the lag in frames, or None if t_2 cannot be resolved: the lag
    does not settle within max_rounds, or the trajectory spans fewer than
    MIN_RELAXATIONS_PER_TRAJECTORY slowest timescales.
    """
    n_frames = len(discrete_trajectory)
    lagtime = initial_lag
    for _ in range(max_rounds):
        slowest_timescale = build_msm(discrete_trajectory, lagtime).timescales()[0]
        if slowest_timescale * MIN_RELAXATIONS_PER_TRAJECTORY > n_frames:
            return None
        next_lagtime = max(1, int(round(fraction_of_slowest * slowest_timescale)))
        if abs(next_lagtime - lagtime) <= 0.1 * lagtime:
            return next_lagtime
        lagtime = next_lagtime

    return None


def chapman_kolmogorov_error(discrete_trajectory, lagtime, n_multiples=4):
    """
    Largest gap between predicted and directly estimated 2-macrostate
    transition probabilities at lags k * lagtime, for k = 1..n_multiples.

    The prediction propagates the model at lagtime k times; the estimate
    builds a new model at k * lagtime. A Markovian model makes them agree,
    so a small value (well below 0.05) supports the chosen lag.
    """
    models = [build_msm(discrete_trajectory, lagtime * k) for k in range(1, n_multiples + 1)]
    ck_result = models[0].ck_test(models, n_metastable_sets=2)
    largest_gap = np.max(np.abs(ck_result.predictions - ck_result.estimates))
    return float(largest_gap)


def timescale_separation(msm):
    """
    Ratio t_2 / t_3 of the two slowest implied timescales.

    A double well has one slow process (crossing the barrier) and fast
    in-well relaxation, so t_2 / t_3 is large (tens or more; about 44 at
    beta=5 and the chosen lag). A single well has no slow process and gives
    a ratio below 2. This is
    the evidence for two metastable states; PCCA+ alone is not, because it
    always returns as many sets as it is asked for.
    """
    timescales = msm.timescales()
    return timescales[0] / timescales[1]


def recover_two_macrostates(discrete_trajectory, lagtime):
    """
    Coarse-grain the microstates into 2 macrostates with PCCA+.

    Returns
    -------
    msm : deeptime MSM model
        The microstate MSM (see build_msm()).
    pcca_model : deeptime.markov.PCCAModel
        `.assignments` gives each microstate's macrostate (0 or 1);
        `.coarse_grained_stationary_probability` gives the 2 populations.
    """
    msm = build_msm(discrete_trajectory, lagtime)
    pcca_model = msm.pcca(n_metastable_sets=2)
    return msm, pcca_model
