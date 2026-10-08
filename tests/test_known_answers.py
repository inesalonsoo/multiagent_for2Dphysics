"""
Known-answer tests for physics/known_answers.py.

Each test compares a function against a value derived by hand or by an
independent method: exact symmetric-case results, a finite-difference
check of the root finding, the small-tilt expansion, the Eyring-Kramers
formula, and two independent routes to the exact relaxation rate.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from physics.potential import potential, potential_derivative
from physics.known_answers import (
    MAX_BETA_FOR_EXACT_RATE,
    expected_number_of_states,
    barrier_height,
    eyring_kramers_rate_0d,
    exact_relaxation_rate_0d,
    euler_maruyama_relaxation_rate_0d,
    find_well_positions,
    energy_difference_of_minima,
    boltzmann_population_ratio,
)

TOLERANCE = 1e-8


def test_expected_number_of_states_is_two():
    """The double well always has exactly two macrostates."""
    assert expected_number_of_states() == 2


def test_barrier_height_equals_A():
    """barrier_height(A) must just return A."""
    assert barrier_height(A=1.0) == 1.0
    assert barrier_height(A=2.5) == 2.5


def test_symmetric_well_positions_are_exactly_plus_minus_one():
    """With b=0, root-finding must recover x=-1 and x=+1 exactly."""
    x_minus, x_plus = find_well_positions(A=1.0, b=0.0)

    assert abs(x_minus - (-1.0)) < TOLERANCE
    assert abs(x_plus - 1.0) < TOLERANCE


def test_symmetric_energy_difference_is_exactly_zero():
    """For b=0, V(x_plus) = V(x_minus) = 0, so the difference is zero."""
    delta_V = energy_difference_of_minima(A=1.0, b=0.0)

    assert abs(delta_V) < TOLERANCE


def test_symmetric_population_ratio_is_exactly_one():
    """For b=0 the two basins are mirror images, so the ratio is 1."""
    for beta in (3.0, 5.0, 20.0):
        ratio = boltzmann_population_ratio(beta=beta, A=1.0, b=0.0)
        assert abs(ratio - 1.0) < TOLERANCE


def test_root_positions_satisfy_derivative_via_finite_difference():
    """
    A minimum of V must have zero slope. The slope is checked with an
    independent central finite difference, not with potential_derivative(),
    which find_well_positions() used to locate the root.
    """
    barrier_height_value = 1.0
    tilt = 0.15
    step_size = 1e-6

    for b in (0.0, tilt):
        x_minus, x_plus = find_well_positions(A=barrier_height_value, b=b)
        for root in (x_minus, x_plus):
            analytical_slope = potential_derivative(root, A=barrier_height_value, b=b)
            potential_above = potential(root + step_size, A=barrier_height_value, b=b)
            potential_below = potential(root - step_size, A=barrier_height_value, b=b)
            numerical_slope = (potential_above - potential_below) / (2 * step_size)

            assert abs(analytical_slope) < 1e-6
            assert abs(numerical_slope) < 1e-4


# (A, a tilt just below the critical tilt, a tilt just above it). The critical
# tilt is 1.539601 for A = 1 and 3.079201 for A = 2.
NEAR_CRITICAL_TILTS = [(1.0, 1.5395, 1.5397), (2.0, 3.0791, 3.0793)]


def test_wells_are_found_up_to_the_critical_tilt():
    """
    Just below the critical tilt both wells still exist. Each returned root
    must have zero slope and positive curvature: a minimum, not the barrier.
    """
    for barrier_height_value, tilt_below, _ in NEAR_CRITICAL_TILTS:
        for b in (tilt_below, -tilt_below):
            for root in find_well_positions(A=barrier_height_value, b=b):
                # V''(x) = 12*A*x**2 - 4*A
                curvature = barrier_height_value * (12.0 * root**2 - 4.0)

                assert abs(potential_derivative(root, A=barrier_height_value, b=b)) < 1e-9
                assert curvature > 0.0


def test_well_positions_refuse_a_single_well_tilt():
    """At or above the critical tilt one well is gone, so there is no pair to return."""
    for barrier_height_value, _, tilt_above in NEAR_CRITICAL_TILTS:
        for b in (tilt_above, -tilt_above):
            with pytest.raises(ValueError, match="critical tilt"):
                find_well_positions(A=barrier_height_value, b=b)


def test_tilted_energy_difference_matches_small_tilt_expansion():
    """For small b, V(x_plus) - V(x_minus) = 2*b + O(b**3)."""
    small_tilt = 0.05

    delta_V = energy_difference_of_minima(A=1.0, b=small_tilt)
    expansion = 2.0 * small_tilt

    relative_error = abs(delta_V - expansion) / expansion
    assert relative_error < 1e-3


def test_tilted_population_ratio_matches_harmonic_limit():
    """
    At large beta each basin integral becomes a Gaussian, so the ratio
    tends to sqrt(V''(x_minus) / V''(x_plus)) * exp(-beta * deltaV).
    The square root is the well-width factor that exp(-beta * deltaV)
    alone misses.
    """
    beta = 40.0
    tilt = 0.1
    x_minus, x_plus = find_well_positions(A=1.0, b=tilt)

    # V''(x) = 12*A*x**2 - 4*A (independent of the tilt)
    curvature_minus = 12.0 * x_minus**2 - 4.0
    curvature_plus = 12.0 * x_plus**2 - 4.0
    width_factor = np.sqrt(curvature_minus / curvature_plus)
    harmonic_ratio = width_factor * np.exp(-beta * energy_difference_of_minima(A=1.0, b=tilt))

    exact_ratio = boltzmann_population_ratio(beta=beta, A=1.0, b=tilt)
    assert abs(exact_ratio / harmonic_ratio - 1.0) < 5e-3


def test_tilted_population_favors_lower_well():
    """A positive tilt raises the x=+1 well, so it must hold less population."""
    ratio = boltzmann_population_ratio(beta=5.0, A=1.0, b=0.1)

    assert ratio < 1.0


def test_eyring_kramers_rate_matches_hand_calculation():
    """
    Hand-computed check at beta=5, A=1:
    rate = sqrt(8 * 4) / (2*pi) * exp(-5).
    """
    rate = eyring_kramers_rate_0d(beta=5.0, A=1.0)

    expected_rate = np.sqrt(32.0) / (2.0 * np.pi) * np.exp(-5.0)
    assert abs(rate - expected_rate) < 1e-12


def test_exact_rate_is_grid_converged():
    """Halving the cell size must change lambda_2 by less than 1e-4 (relative)."""
    for beta in (3.0, 7.0, 12.0):
        coarse_rate = exact_relaxation_rate_0d(beta=beta, n_cells=2000)
        fine_rate = exact_relaxation_rate_0d(beta=beta, n_cells=4000)

        assert abs(coarse_rate / fine_rate - 1.0) < 1e-4


def _mean_first_passage_time(beta):
    """
    Exact mean time to go from x=-1 to x=+1 (independent of the
    eigenvalue method):
    T = beta * integral_{-1}^{1} exp(beta*V(y)) integral_{-inf}^{y} exp(-beta*V(z)) dz dy
    """
    def inner_integral(y):
        return quad(lambda z: np.exp(-beta * potential(z)), -4.0, y, limit=200)[0]

    def outer_integrand(y):
        return np.exp(beta * potential(y)) * inner_integral(y)

    outer_integral = quad(outer_integrand, -1.0, 1.0, limit=200)[0]
    return beta * outer_integral


def test_exact_rate_matches_mean_first_passage_time():
    """
    For well-separated wells, lambda_2 = 2 / T, where T is the mean
    first-passage time from one well to the other. At beta=7 the two
    independent methods must agree within 1%.
    """
    beta = 7.0

    eigenvalue_rate = exact_relaxation_rate_0d(beta=beta)
    passage_time_rate = 2.0 / _mean_first_passage_time(beta)

    assert abs(eigenvalue_rate / passage_time_rate - 1.0) < 1e-2


def test_exact_rate_approaches_eyring_kramers_from_below():
    """
    Eyring-Kramers is the large-beta limit of the exact rate. The ratio
    exact / (2 * Eyring-Kramers) must stay below 1 and rise toward 1 as
    beta grows.
    """
    beta_values = (3.0, 7.0, 14.0)

    ratios = [exact_relaxation_rate_0d(beta=beta) / (2.0 * eyring_kramers_rate_0d(beta=beta))
              for beta in beta_values]

    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > 0.95


def test_euler_maruyama_bias_halves_with_time_step():
    """
    Euler-Maruyama has first-order weak error, so the chain's rate bias
    relative to the exact rate must shrink about linearly with dt: halving
    dt should roughly halve the bias.
    """
    beta = 5.0
    exact_rate = exact_relaxation_rate_0d(beta=beta)

    bias_at_dt = euler_maruyama_relaxation_rate_0d(beta=beta, dt=0.01) / exact_rate - 1.0
    bias_at_half_dt = euler_maruyama_relaxation_rate_0d(beta=beta, dt=0.005) / exact_rate - 1.0

    assert 0.005 < bias_at_dt < 0.02
    assert 0.4 < bias_at_half_dt / bias_at_dt < 0.6


def test_exact_rate_refuses_beta_beyond_precision_limit():
    """Above MAX_BETA_FOR_EXACT_RATE the eigenvalue is lost in round-off."""
    too_large_beta = MAX_BETA_FOR_EXACT_RATE + 1.0

    try:
        exact_relaxation_rate_0d(beta=too_large_beta)
    except ValueError:
        return
    raise AssertionError("expected ValueError for beta above the precision limit")
