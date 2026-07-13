"""
Known-answer tests for physics/potential.py.

These check the double-well potential V(phi) = A*(phi**2-1)**2 + b*phi
against values we can work out by hand, so any bug in the formula or
its derivative is caught immediately. Most tests use the default b=0
(symmetric well); a few specifically check the b != 0 (tilted) case.
"""

import numpy as np

from physics.potential import potential, potential_derivative

# Tolerance for floating-point comparisons. 1e-10 is tight but safe here
# because these are simple polynomial evaluations, not iterative solves.
TOLERANCE = 1e-10


def test_potential_is_zero_at_minima():
    """V(phi) should be ~0 at phi = -1 and phi = +1 (the two wells)."""
    barrier_height = 1.0

    value_at_left_well = potential(-1.0, A=barrier_height)
    value_at_right_well = potential(1.0, A=barrier_height)

    assert abs(value_at_left_well) < TOLERANCE
    assert abs(value_at_right_well) < TOLERANCE


def test_potential_equals_A_at_barrier():
    """V(phi) should equal A exactly at phi = 0 (the barrier top)."""
    barrier_height = 1.0

    value_at_barrier = potential(0.0, A=barrier_height)

    assert abs(value_at_barrier - barrier_height) < TOLERANCE


def test_derivative_is_zero_at_stationary_points():
    """
    dV/dphi should be ~0 at phi = -1, 0, +1: the two minima and the
    barrier top are all stationary points of the potential.
    """
    stationary_points = np.array([-1.0, 0.0, 1.0])

    slopes = potential_derivative(stationary_points, A=1.0)

    assert np.allclose(slopes, 0.0, atol=TOLERANCE)


def test_potential_is_symmetric():
    """
    V(phi) should equal V(-phi) for any phi, since the potential only
    depends on phi through phi**2. A few arbitrary (non-special) points
    are enough to catch a sign bug that would break this symmetry.
    """
    arbitrary_phi_values = np.array([0.3, 0.7, 1.5, 2.2, -1.8])

    value_at_phi = potential(arbitrary_phi_values, A=1.0)
    value_at_minus_phi = potential(-arbitrary_phi_values, A=1.0)

    assert np.allclose(value_at_phi, value_at_minus_phi, atol=TOLERANCE)


def test_potential_is_non_negative_on_grid():
    """
    V(phi) = A*(phi**2-1)**2 is a square times a positive constant A, so
    it can never dip below zero anywhere. Sample a grid of phi values,
    including points far outside the wells, to check this holds broadly
    rather than just at a few hand-picked points.
    """
    phi_grid = np.linspace(-5.0, 5.0, 201)

    potential_values = potential(phi_grid, A=1.0)

    assert np.all(potential_values >= -TOLERANCE)


def test_derivative_matches_finite_difference():
    """
    Cross-check the analytical derivative potential_derivative() against
    a numerical central finite difference of potential() itself, at
    points away from the roots (-1, 0, +1) where a wrong coefficient
    (e.g. a missing or incorrect factor of 4A) would otherwise still
    slip through by accident (any linear function of the wrong slope
    still crosses zero at those same three roots).

    Central finite difference: (V(phi+eps) - V(phi-eps)) / (2*eps)
    approximates dV/dphi with error of order eps**2, so a small eps
    gives a very close match to the true derivative.
    """
    off_root_points = np.array([-2.3, -0.6, 0.4, 0.9, 1.7])
    barrier_height = 1.0
    step_size = 1e-5

    analytical_slopes = potential_derivative(off_root_points, A=barrier_height)

    potential_above = potential(off_root_points + step_size, A=barrier_height)
    potential_below = potential(off_root_points - step_size, A=barrier_height)
    numerical_slopes = (potential_above - potential_below) / (2 * step_size)

    assert np.allclose(analytical_slopes, numerical_slopes, atol=1e-6)


def test_derivative_matches_finite_difference_with_tilt():
    """
    Same finite-difference cross-check as test_derivative_matches_finite_difference,
    but with the tilt term b != 0, to make sure the "+ b" added to
    potential_derivative() actually corresponds to the "+ b*phi" added to
    potential() (e.g. would catch a sign error, or b added to the wrong
    one of the two functions).
    """
    off_root_points = np.array([-2.3, -0.6, 0.4, 0.9, 1.7])
    barrier_height = 1.0
    tilt = 0.1
    step_size = 1e-5

    analytical_slopes = potential_derivative(off_root_points, A=barrier_height, b=tilt)

    potential_above = potential(off_root_points + step_size, A=barrier_height, b=tilt)
    potential_below = potential(off_root_points - step_size, A=barrier_height, b=tilt)
    numerical_slopes = (potential_above - potential_below) / (2 * step_size)

    assert np.allclose(analytical_slopes, numerical_slopes, atol=1e-6)


def test_tilt_breaks_the_symmetry():
    """
    With b != 0, V(phi) should NOT equal V(-phi) in general -- unlike the
    b=0 case (test_potential_is_symmetric), the tilt is specifically
    meant to make the two wells unequal.
    """
    arbitrary_phi = 0.7
    tilt = 0.1

    value_at_phi = potential(arbitrary_phi, A=1.0, b=tilt)
    value_at_minus_phi = potential(-arbitrary_phi, A=1.0, b=tilt)

    assert abs(value_at_phi - value_at_minus_phi) > TOLERANCE
