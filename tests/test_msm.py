"""
Known-answer tests for pipeline/msm.py, on real 0-D trajectories.

The shared beta=5 trajectory (1.5M steps of dt=0.01) spans about 170
slowest timescales, enough to resolve the barrier-crossing process.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d
from physics.known_answers import euler_maruyama_relaxation_rate_0d
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.msm import (
    build_msm, implied_timescales, choose_lagtime, chapman_kolmogorov_error,
    timescale_separation, recover_two_macrostates,
)

DT = 0.01


def _discrete_trajectory(beta, n_steps, seed, b=0.0):
    """Simulate a 0-D trajectory and cluster it into 50 microstates."""
    trajectory = run_trajectory_0d(n_steps=n_steps, seed=seed, beta=beta, dt=DT, b=b)
    features = compute_features(trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=50, seed=42)
    return discrete_trajectory


_DISCRETE_TRAJECTORY = _discrete_trajectory(beta=5.0, n_steps=1_500_000, seed=7)


def test_build_msm_returns_valid_model():
    """The fitted MSM's state count must not exceed the number of microstates."""
    msm = build_msm(_DISCRETE_TRAJECTORY, lagtime=10)

    assert 1 < msm.n_states <= 50


def test_implied_timescales_increase_with_lag():
    """
    Variational principle: an MSM underestimates the slowest timescale at
    short lags and approaches the true value from below as the lag grows.
    """
    timescales = implied_timescales(_DISCRETE_TRAJECTORY, [10, 100, 500])

    assert timescales[0] < timescales[1] < timescales[2]


def test_chosen_lag_is_a_tenth_of_slowest_timescale():
    """choose_lagtime() must return a lag self-consistent with its own rule."""
    lagtime = choose_lagtime(_DISCRETE_TRAJECTORY)
    slowest_timescale = implied_timescales(_DISCRETE_TRAJECTORY, [lagtime])[0]

    assert abs(lagtime / (0.1 * slowest_timescale) - 1.0) < 0.1


def test_rate_at_chosen_lag_matches_exact_chain_rate():
    """
    At beta=3, a 1.5M-step trajectory sees about 600 barrier crossings, so
    the rate's statistical error is about 4%. At the chosen lag the MSM
    rate must match the exact rate of the simulated Euler-Maruyama chain
    within 6% (the trajectory is seeded, so the test is deterministic).
    """
    discrete_trajectory = _discrete_trajectory(beta=3.0, n_steps=1_500_000, seed=11)
    lagtime = choose_lagtime(discrete_trajectory)
    slowest_timescale = build_msm(discrete_trajectory, lagtime).timescales()[0]

    measured_rate = 1.0 / (slowest_timescale * DT)
    exact_chain_rate = euler_maruyama_relaxation_rate_0d(beta=3.0, dt=DT)

    assert abs(measured_rate / exact_chain_rate - 1.0) < 0.06


def test_choose_lagtime_refuses_unresolvable_trajectory():
    """
    At beta=9 the slowest timescale is about 480,000 frames, so a
    200,000-frame trajectory cannot resolve it. The function must say so
    (None) instead of returning a lag.
    """
    discrete_trajectory = _discrete_trajectory(beta=9.0, n_steps=200_000, seed=3)

    assert choose_lagtime(discrete_trajectory) is None


def test_chapman_kolmogorov_error_is_small_at_chosen_lag():
    """A Markovian model predicts its own longer-lag estimates (gap < 0.05)."""
    lagtime = choose_lagtime(_DISCRETE_TRAJECTORY)

    assert chapman_kolmogorov_error(_DISCRETE_TRAJECTORY, lagtime) < 0.05


def test_timescale_separation_distinguishes_double_from_single_well():
    """
    The double well has one slow process, so t_2 / t_3 is large. Tilting
    with b=3 (beyond the spinodal, about 1.54) leaves a single well with no
    slow process, so the ratio must be small. PCCA+ would still report
    2 sets for the single well; this check does not.
    """
    double_well_msm = build_msm(_DISCRETE_TRAJECTORY, choose_lagtime(_DISCRETE_TRAJECTORY))
    single_well_trajectory = _discrete_trajectory(beta=5.0, n_steps=1_500_000, seed=7, b=3.0)
    single_well_msm = build_msm(single_well_trajectory, lagtime=10)

    assert timescale_separation(double_well_msm) > 10.0
    assert timescale_separation(single_well_msm) < 2.0


def test_macrostate_populations_are_symmetric():
    """
    For b=0 the two wells hold equal populations. The trajectory has about
    85 crossings, so the population difference fluctuates by roughly
    1/sqrt(42), about 15%; 0.15 is the tolerance.
    """
    _, pcca_model = recover_two_macrostates(_DISCRETE_TRAJECTORY, lagtime=20)
    populations = pcca_model.coarse_grained_stationary_probability

    assert abs(populations.sum() - 1.0) < 1e-6
    assert abs(populations[0] - populations[1]) < 0.15
