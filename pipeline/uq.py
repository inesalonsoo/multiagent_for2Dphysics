"""
Bayesian credible interval for the MSM relaxation rate, using
deeptime.markov.msm.BayesianMSM with count_mode="effective", the mode
deeptime recommends for Bayesian MSMs.

Known limitation, measured in scripts/run_phase2_uq.py: for this system
the "effective" counts are only about 20% below the overlapping "sliding"
counts, although those are correlated over a whole lag time. The posterior
therefore acts as if it had about a million independent transitions, and
its intervals are 11-190 times narrower than the real replica-to-replica
spread (2 of 42 intervals contain the exact rate). The project's
uncertainty is the replica spread, not these intervals.
"""

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM


def compute_rate_credible_interval(discrete_trajectory, lagtime, dt,
                                    confidence=0.90, n_samples=100):
    """
    Bayesian credible interval for the MSM relaxation rate
    (1 / slowest implied timescale), via BayesianMSM's posterior over
    transition matrices sampled from the given discrete trajectory.

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Microstate-labeled trajectory, e.g. from
        pipeline.cluster.cluster_trajectory().
    lagtime : int
        Lag time in frames (same convention as pipeline.msm.build_msm).
    dt : float
        Physical time per frame, to convert the implied timescale (in
        frames) into physical time before inverting it into a rate.
    confidence : float, optional
        Credible interval width, e.g. 0.90 for a 90% interval. Default 0.90.
    n_samples : int, optional
        Number of posterior transition-matrix samples to draw. Default 100.

    Returns
    -------
    rate_mean, rate_lower, rate_upper : float
        Posterior mean relaxation rate and the [lower, upper] edges of the
        credible interval. Note that rate = 1/timescale is a DECREASING
        function, so the lower rate bound corresponds to the UPPER
        timescale bound and vice versa -- handled here so the caller
        can't get this backwards.
    """
    count_model = TransitionCountEstimator(
        lagtime=lagtime, count_mode="effective",
    ).fit(discrete_trajectory).fetch_model()
    bayesian_msm = BayesianMSM(n_samples=n_samples).fit(count_model).fetch_model()

    timescale_stats = bayesian_msm.gather_stats("timescales", confidence=confidence)
    slowest_timescale_mean = timescale_stats.mean[0]
    slowest_timescale_lower = timescale_stats.L[0]
    slowest_timescale_upper = timescale_stats.R[0]

    rate_mean = 1.0 / (slowest_timescale_mean * dt)
    rate_lower = 1.0 / (slowest_timescale_upper * dt)
    rate_upper = 1.0 / (slowest_timescale_lower * dt)

    return rate_mean, rate_lower, rate_upper
