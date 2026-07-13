"""
Known-answer tests for physics/known_answers.py.

These check the analytical Phase 1 (0-D) ground truth against values we
can work out by hand: the exact number of macrostates, the exact barrier
height, the exact symmetric-case free-energy/population results, a
finite-difference cross-check on the root-finding (in the spirit of
Module 1.1's test_derivative_matches_finite_difference), a perturbative
cross-check of the tilted free-energy difference, and a hand-computed
value of the 1-DOF Eyring-Kramers rate.
"""

import numpy as np

from physics.potential import potential, potential_derivative
from physics.known_answers import (
    expected_number_of_states,
    barrier_height,
    eyring_kramers_rate_0d,
    find_well_positions,
    free_energy_difference,
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


def test_symmetric_free_energy_difference_is_exactly_zero():
    """
    For the symmetric potential (b=0), V(x_plus)=V(x_minus)=0 exactly, so
    deltaF must be exactly zero (up to solver tolerance).
    """
    delta_F = free_energy_difference(A=1.0, b=0.0)

    assert abs(delta_F) < TOLERANCE


def test_symmetric_boltzmann_ratio_is_exactly_one():
    """With deltaF=0, the population ratio must be exactly 1 (equal wells)."""
    ratio = boltzmann_population_ratio(beta=5.0, A=1.0, b=0.0)

    assert abs(ratio - 1.0) < TOLERANCE


def test_root_positions_satisfy_derivative_via_finite_difference():
    """
    Cross-check the root-found well positions the same way Module 1.1
    cross-checked potential_derivative(): a true minimum of V must have
    zero slope AND that zero slope must agree with an independent
    central finite-difference estimate of dV/dx at that exact point, not
    just with potential_derivative()'s own formula (which is what
    find_well_positions() used to locate the root in the first place --
    this test intentionally uses a DIFFERENT, numerical method to avoid
    just re-checking the same formula against itself).
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


def test_tilted_free_energy_difference_matches_perturbative_2b():
    """
    For modest tilt b, perturbation theory predicts deltaF = 2*b + O(b**3)
    (the O(b**2) well-shift correction cancels exactly -- see
    PROJECT_STATE.md Sec 9 for the derivation). This cross-checks the
    root-finding-based free_energy_difference() against that independent
    closed-form approximation.
    """
    barrier_height_value = 1.0
    small_tilt = 0.05

    delta_F = free_energy_difference(A=barrier_height_value, b=small_tilt)
    perturbative_estimate = 2.0 * small_tilt

    relative_error = abs(delta_F - perturbative_estimate) / perturbative_estimate
    assert relative_error < 1e-3


def test_eyring_kramers_rate_matches_hand_calculation():
    """
    Hand-computed check at beta=5, A=1.0:
    rate = sqrt(8*1 * 4*1) / (2*pi) * exp(-5*1) = sqrt(32)/(2*pi) * exp(-5).
    """
    beta = 5.0
    barrier_height_value = 1.0

    rate = eyring_kramers_rate_0d(beta=beta, A=barrier_height_value)

    expected_rate = np.sqrt(32.0) / (2.0 * np.pi) * np.exp(-5.0)
    assert abs(rate - expected_rate) < 1e-12


def test_eyring_kramers_rate_decreases_with_beta():
    """
    Increasing beta (lowering temperature) must strictly decrease the
    escape rate -- a basic monotonicity sanity check on the formula.
    """
    rate_at_low_beta = eyring_kramers_rate_0d(beta=4.0, A=1.0)
    rate_at_high_beta = eyring_kramers_rate_0d(beta=8.0, A=1.0)

    assert rate_at_high_beta < rate_at_low_beta


def test_eyring_kramers_slope_in_beta_is_exactly_minus_A():
    """
    log(rate) = log(prefactor) - beta*A: the SLOPE in beta is exactly -A,
    independent of beta -- this is the ironclad part of the Eyring-Kramers
    check (the prefactor is only asymptotically accurate, see
    eyring_kramers_rate_0d()'s docstring), verified here across a range
    of beta values that don't include the beta used to derive the formula.
    """
    barrier_height_value = 1.0
    beta_values = np.array([3.0, 6.0, 9.0, 12.0])

    log_rates = np.log([eyring_kramers_rate_0d(beta=b, A=barrier_height_value)
                         for b in beta_values])

    slope = np.polyfit(beta_values, log_rates, deg=1)[0]

    assert abs(slope - (-barrier_height_value)) < 1e-10
