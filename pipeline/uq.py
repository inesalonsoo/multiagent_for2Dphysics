"""
Bayesian credible interval for the MSM relaxation rate, using
deeptime.markov.msm.BayesianMSM with count_mode="effective", the mode
deeptime recommends for Bayesian MSMs.

Known limitation, measured in scripts/run_phase2_uq.py: for this system
the "effective" counts are only about 20% below the overlapping "sliding"
counts, although those are correlated over a whole lag time. The posterior
therefore acts as if it had millions of independent transitions, and
its intervals are 11-190 times narrower than the real replica-to-replica
spread (2 of 42 intervals contain the exact rate). The project's
uncertainty is the replica spread, not these intervals.
"""

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM


def sample_rate_posterior(discrete_trajectory, lagtime, n_samples=100):
    """
    Fit a Bayesian MSM: n_samples transition matrices drawn from the
    posterior given the observed ("effective") transition counts at this
    lag. Sampling once lets several intervals be read from the same draws.
    """
    count_model = TransitionCountEstimator(
        lagtime=lagtime, count_mode="effective",
    ).fit(discrete_trajectory).fetch_model()
    return BayesianMSM(n_samples=n_samples).fit(count_model).fetch_model()


def rate_credible_interval(posterior, dt, confidence=0.90):
    """
    Posterior mean and credible interval of the relaxation rate
    (1 / slowest implied timescale) from already-drawn posterior samples.

    Returns
    -------
    rate_mean, rate_lower, rate_upper : float
        Since rate = 1/timescale, the lower rate bound comes from the upper
        timescale bound and vice versa. dt converts frames to time.
    """
    timescale_stats = posterior.gather_stats("timescales", confidence=confidence)
    slowest_timescale_mean = timescale_stats.mean[0]
    slowest_timescale_lower = timescale_stats.L[0]
    slowest_timescale_upper = timescale_stats.R[0]

    rate_mean = 1.0 / (slowest_timescale_mean * dt)
    rate_lower = 1.0 / (slowest_timescale_upper * dt)
    rate_upper = 1.0 / (slowest_timescale_lower * dt)
    return rate_mean, rate_lower, rate_upper


def compute_rate_credible_interval(discrete_trajectory, lagtime, dt,
                                    confidence=0.90, n_samples=100):
    """
    Sample the posterior once and return the rate's mean and credible
    interval at this confidence (see the two functions above).

    Parameters
    ----------
    discrete_trajectory : np.ndarray
        Microstate index per frame (pipeline.cluster.cluster_trajectory).
    lagtime : int
        Lag time in frames.
    dt : float
        Time per frame.
    confidence : float, optional
        Interval probability, e.g. 0.90. Default 0.90.
    n_samples : int, optional
        Number of posterior samples. Default 100.
    """
    posterior = sample_rate_posterior(discrete_trajectory, lagtime, n_samples)
    return rate_credible_interval(posterior, dt, confidence)
