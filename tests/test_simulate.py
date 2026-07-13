"""
Known-answer tests for physics/simulate.py.

These check the stochastic Allen-Cahn integrator's basic numerical
sanity (correct shape, no blow-ups, reproducibility, CFL guard) and two
physics checks: (1) with the temperature effectively turned off (beta
very large, so the noise variance 2*gamma/beta is negligible), the
field must deterministically relax toward the nearest potential
minimum, exactly like the bare potential's derivative predicts; and
(2) with the potential turned off entirely (pure diffusion + noise),
the domain-mean order parameter's variance must grow at the rate
predicted by the noise amplitude we set, catching a mis-scaled noise
term in isolation before it could contaminate a switching-rate
measurement.
"""

import os
import tempfile

import numpy as np
import pde

from physics.simulate import (
    run_trajectory,
    GRID_SHAPE,
    DOMAIN_SIZE,
    _build_grid,
    _build_equation,
    _make_storage,
    _extract_trajectory_array,
)

TOLERANCE = 1e-6


def test_trajectory_has_expected_shape_and_is_finite():
    """run_trajectory should return (n_steps, 32, 32) with no NaN/Inf."""
    n_steps = 10

    trajectory = run_trajectory(n_steps=n_steps, dt=0.01, seed=0)

    assert trajectory.shape == (n_steps,) + GRID_SHAPE
    assert np.all(np.isfinite(trajectory))


def test_same_seed_gives_identical_trajectory():
    """Reusing the same seed must reproduce the exact same noise realization."""
    first_run = run_trajectory(n_steps=15, dt=0.01, seed=42)
    second_run = run_trajectory(n_steps=15, dt=0.01, seed=42)

    assert np.array_equal(first_run, second_run)


def test_different_seeds_give_different_trajectories():
    """Different seeds must actually randomize the thermal noise."""
    run_with_seed_a = run_trajectory(n_steps=15, dt=0.01, seed=1)
    run_with_seed_b = run_trajectory(n_steps=15, dt=0.01, seed=2)

    assert not np.array_equal(run_with_seed_a, run_with_seed_b)


def test_low_noise_relaxes_toward_nearest_well():
    """
    With beta very large, the noise variance 2*gamma/beta is negligible,
    so starting away from either minimum (phi=0.5, closer to the phi=+1
    well) the field should deterministically relax toward phi=+1, just
    like potential_derivative() predicts for the noiseless equation.
    """
    near_zero_noise_beta = 1.0e8

    trajectory = run_trajectory(
        n_steps=200, dt=0.01, seed=0,
        beta=near_zero_noise_beta, initial_phi=0.5,
    )

    final_frame_mean = trajectory[-1].mean()

    assert abs(final_frame_mean - 1.0) < 1.0e-3


def test_cfl_violation_raises_clear_error():
    """
    A dt above the CFL stability bound dt < dx**2/(4*gamma) must raise a
    ValueError before any integration happens, not silently blow up.
    dx=0.3125 here, so the bound is 0.3125**2/4 = ~0.02441; dt=0.03
    violates it.
    """
    dt_above_cfl_bound = 0.03

    try:
        run_trajectory(n_steps=5, seed=0, dt=dt_above_cfl_bound)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_pure_diffusion_noise_variance_matches_prediction():
    """
    Cheap sanity check for py-pde's noise scaling, run in isolation from
    the double-well potential (include_potential=False), so a wrong
    noise amplitude can't hide behind the potential's own dynamics.

    With no potential term, d(phi)/dt = gamma*laplacian(phi) + noise is
    pure linear diffusion plus additive noise -- no metastability, no
    wells. Under periodic boundaries the discrete Laplacian conserves
    the grid SUM exactly on every step (it only moves phi between
    cells), so the domain-mean order parameter's fluctuations come only
    from the noise term. This gives a closed-form prediction:

        Var[mean(phi(t)) - mean(phi(0))] = noise_variance * t / domain_area

    (derived from py-pde's per-cell noise increment variance,
    dt*noise_variance/cell_volume, summed over all independent cells and
    divided by the number of cells squared). Comparing many independent
    replicas against this prediction will immediately reveal a wrong
    noise_variance (e.g. passing the amplitude sqrt(2*gamma/beta)
    instead of its square by mistake) -- long before it could
    contaminate a switching-rate measurement downstream.

    Replicas must be genuinely independent trajectories (each is a
    fresh, separately-integrated realization of the noise), not frames
    of a single trajectory -- that would not test the same thing and,
    for the full potential-on dynamics, would also be statistically
    invalid (the domain mean is not a simple random walk once the
    nonlinear force is present). To keep this cheap, we build ONE
    equation object and reuse it across replicas (giving each a fresh
    initial field): py-pde re-JIT-compiles its grid operators for every
    new equation object (~12s), but a rebuilt equation's own rng keeps
    advancing across repeated solve() calls, so reusing it still yields
    independent noise draws per replica, at near-zero extra cost.
    """
    gamma = 1.0
    beta = 5.0
    n_steps = 40
    dt = 0.005
    n_replicas = 150

    noise_variance = 2.0 * gamma / beta
    domain_area = DOMAIN_SIZE[0] * DOMAIN_SIZE[1]
    total_time = n_steps * dt
    predicted_variance = noise_variance * total_time / domain_area

    grid = _build_grid()
    equation = _build_equation(
        gamma=gamma, A=1.0, noise_variance=noise_variance, seed=2026,
        include_potential=False,
    )

    final_domain_means = []
    for _ in range(n_replicas):
        initial_state = pde.ScalarField(grid, data=0.0)
        storage = _make_storage(None)
        equation.solve(
            initial_state, t_range=total_time, dt=dt, solver="euler",
            tracker=storage.tracker(dt), backend="numpy",
        )
        trajectory = _extract_trajectory_array(storage)
        final_domain_means.append(trajectory[-1].mean())

    empirical_variance = np.var(final_domain_means, ddof=1)

    assert 0.6 * predicted_variance < empirical_variance < 1.6 * predicted_variance


def test_storage_path_streams_to_disk_and_matches_memory_run():
    """
    Streaming to an HDF5 file via storage_path must give the exact same
    trajectory (same seed) as keeping everything in memory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "trajectory.h5")

        in_memory_trajectory = run_trajectory(n_steps=10, dt=0.01, seed=7)
        on_disk_trajectory = run_trajectory(
            n_steps=10, dt=0.01, seed=7, storage_path=file_path
        )

        assert os.path.exists(file_path)
        assert np.array_equal(in_memory_trajectory, on_disk_trajectory)
