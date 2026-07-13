"""
Known-answer tests for physics/simulate_0d.py.

These check the 0-D Euler-Maruyama integrator's basic numerical sanity
(shape, finiteness, reproducibility), one deterministic-limit physics
check (near-zero noise relaxes to the correct well), and one noise-scale
check that is EXACT in 0-D (no spatial discretization to complicate it,
unlike the 2D field's cell-volume subtlety in test_simulate.py): with the
potential turned off entirely (A=0, b=0), dx = sqrt(2/beta)*dW is pure
Brownian motion, so Var[x(t)] = (2/beta)*t exactly.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d


def test_trajectory_has_expected_shape_and_is_finite():
    """run_trajectory_0d should return shape (n_steps,) with no NaN/Inf."""
    n_steps = 1000

    trajectory = run_trajectory_0d(n_steps=n_steps, seed=0)

    assert trajectory.shape == (n_steps,)
    assert np.all(np.isfinite(trajectory))


def test_same_seed_gives_identical_trajectory():
    """Reusing the same seed must reproduce the exact same noise realization."""
    first_run = run_trajectory_0d(n_steps=500, seed=42)
    second_run = run_trajectory_0d(n_steps=500, seed=42)

    assert np.array_equal(first_run, second_run)


def test_different_seeds_give_different_trajectories():
    """Different seeds must actually randomize the thermal noise."""
    run_with_seed_a = run_trajectory_0d(n_steps=500, seed=1)
    run_with_seed_b = run_trajectory_0d(n_steps=500, seed=2)

    assert not np.array_equal(run_with_seed_a, run_with_seed_b)


def test_low_noise_relaxes_toward_nearest_well():
    """
    With beta very large, the noise prefactor sqrt(2/beta) is negligible,
    so starting away from either minimum (x=0.5, closer to the x=+1 well)
    the particle should deterministically relax toward x=+1, exactly like
    potential_derivative() predicts for the noiseless equation.
    """
    near_zero_noise_beta = 1.0e8

    trajectory = run_trajectory_0d(
        n_steps=2000, seed=0, beta=near_zero_noise_beta, x0=0.5,
    )

    final_value = trajectory[-1]

    assert abs(final_value - 1.0) < 1.0e-3


def test_pure_noise_variance_matches_prediction():
    """
    With the potential entirely switched off (A=0, b=0), the equation
    reduces to dx = sqrt(2/beta)*dW, exact Brownian motion. Its variance
    grows exactly linearly: Var[x(t) - x(0)] = (2/beta)*t. This is a
    cleaner, exact version of the analogous check in test_simulate.py
    (which needs a domain-area correction because the 2D field is
    spatially discretized; a single 0-D point has no such correction).
    """
    beta = 5.0
    n_steps = 2000
    dt = 0.01
    n_replicas = 200
    total_time = n_steps * dt

    predicted_variance = (2.0 / beta) * total_time

    final_positions = np.array([
        run_trajectory_0d(n_steps=n_steps, seed=seed, beta=beta, A=0.0, b=0.0, x0=0.0)[-1]
        for seed in range(n_replicas)
    ])
    empirical_variance = np.var(final_positions, ddof=1)

    assert 0.7 * predicted_variance < empirical_variance < 1.3 * predicted_variance
