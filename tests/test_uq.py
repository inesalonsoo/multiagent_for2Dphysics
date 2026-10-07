"""
Tests for pipeline/uq.py on a real 0-D trajectory at beta=5: the Bayesian
credible interval must be well ordered, shrink with more data, and widen
with higher confidence. Whether these intervals are honest (their coverage
of the exact rate) is measured in scripts/run_phase2_uq.py.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.uq import compute_rate_credible_interval, rate_credible_interval, sample_rate_posterior

DT = 0.01
LAGTIME = 20


def _sample_discrete_trajectory(n_steps=1_500_000, seed=7, beta=5.0):
    trajectory = run_trajectory_0d(n_steps=n_steps, seed=seed, beta=beta, dt=DT)
    features = compute_features(trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=50, seed=42)
    return discrete_trajectory


def test_credible_interval_is_well_ordered_and_contains_the_mean():
    """lower <= mean <= upper must hold for a sane credible interval."""
    discrete_trajectory = _sample_discrete_trajectory()

    rate_mean, rate_lower, rate_upper = compute_rate_credible_interval(
        discrete_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )

    assert rate_lower < rate_mean < rate_upper


def test_credible_interval_scales_down_with_more_data():
    """
    Doubling the trajectory length must shrink (or at worst not grow) the
    interval, since it describes sampling noise for a fixed model.
    """
    beta = 5.0
    short_trajectory = _sample_discrete_trajectory(n_steps=750_000, seed=7, beta=beta)
    long_trajectory = _sample_discrete_trajectory(n_steps=1_500_000, seed=7, beta=beta)

    _, short_lower, short_upper = compute_rate_credible_interval(
        short_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )
    _, long_lower, long_upper = compute_rate_credible_interval(
        long_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )

    assert (long_upper - long_lower) <= (short_upper - short_lower)


def test_wider_confidence_gives_wider_interval():
    """
    On the same posterior samples, a 95% interval must contain the 90% one.
    (Comparing two separate random draws would make this test flaky.)
    """
    posterior = sample_rate_posterior(_sample_discrete_trajectory(), lagtime=LAGTIME)

    _, lower_90, upper_90 = rate_credible_interval(posterior, dt=DT, confidence=0.90)
    _, lower_95, upper_95 = rate_credible_interval(posterior, dt=DT, confidence=0.95)

    assert lower_95 <= lower_90 and upper_90 <= upper_95
