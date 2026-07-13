"""
Analytical ground-truth answers for the Phase 1 0-D double-well benchmark,
dx = -V'(x) dt + sqrt(2/beta) dW. These are the checks the MSM pipeline
must reproduce -- see PROJECT_STATE.md Sec 3/9 and CLAUDE.md's PHYSICS
GROUND TRUTH section. No fitting, no simulation: every function here is
either a pure closed-form expression, or an exact numerical root-find of
an analytical equation (scipy.optimize.brentq locating the true root of
potential_derivative, not approximating anything from data).

IMPORTANT -- rate law used here vs the literature reference:
The rate law below is the textbook ONE-DEGREE-OF-FREEDOM Eyring-Kramers
formula, NOT the field-theoretic Eq. 13 of Rolland, Bouchet & Simonnet
(arXiv:1507.05577). Their Eq. 13's prefactor involves a ratio of
infinite-dimensional Hessian determinants (products over all the field's
eigenmodes) and a saddle eigenvalue that decays as exp(-L/sqrt(2)) with
the domain size L -- none of that applies here, since a single point has
no spatial modes and no L. The two formulas are NOT in conflict: dropping
Eq. 13's infinite products down to a single scalar "Hessian" (exactly what
happens when there is only one degree of freedom) reduces it exactly to
the classical 1-DOF Kramers rate implemented directly below (this is also
why their own Sec 3.2.3 uses the 1-DOF case as its reference point for
what a "clean" rate law looks like). We implement the simple, closed-form
1-DOF version directly here, rather than importing field-theory machinery
that doesn't belong in a 0-D module.
"""

import numpy as np
from scipy.optimize import brentq

from physics.potential import potential, potential_derivative


def expected_number_of_states():
    """
    The double well has exactly two stable minima, so the MSM built on
    top of it must recover exactly two dominant macrostates.
    """
    return 2


def barrier_height(A=1.0):
    """
    For the untilted (b=0) double well V(x) = A*(x**2-1)**2, the barrier
    height (V at the barrier x=0, minus V at either well, which are both
    exactly 0) is exactly A.
    """
    return A


def eyring_kramers_rate_0d(beta, A=1.0):
    """
    Exact closed-form, one-degree-of-freedom Eyring-Kramers escape rate
    for the SYMMETRIC (b=0) double well V(x) = A*(x**2-1)**2, in the
    small-noise (large-beta) asymptotic limit:

        rate = sqrt(V''(well) * |V''(saddle)|) / (2*pi) * exp(-beta*deltaV)

    Derivation of the three ingredients, using V''(x) = 12*A*x**2 - 4*A
    (the second derivative of A*(x**2-1)**2, see physics/potential.py):
    - curvature at each well (x=+-1): V''(+-1) = 12*A - 4*A = 8*A.
    - curvature at the barrier (x=0): V''(0) = -4*A, so |V''(0)| = 4*A
      (the barrier is a local MAXIMUM, hence the negative curvature; we
      take its absolute value, as Kramers theory requires).
    - deltaV = V(0) - V(+-1) = A - 0 = A.

    No root-finding is needed here (unlike free_energy_difference() below)
    because for the SYMMETRIC well these three quantities are known
    exactly without solving anything: the wells sit at exactly x=+-1 and
    the barrier at exactly x=0.

    Asymptotic caveat: this formula is exact only as beta -> infinity. At
    finite beta (e.g. this project's benchmark beta=5), expect the
    PREFACTOR to be accurate only to within ~15-20%, tightening as beta
    grows (see tests/test_known_answers.py and PROJECT_STATE.md Sec 9,
    where this is checked empirically against simulate_0d.py). The
    EXPONENT -- the slope of log(rate) vs beta, which is exactly -A --
    holds regardless of beta and is the ironclad part of this check; the
    absolute prefactor is the softer, asymptotic part.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    A : float, optional
        Barrier height of the double well. Default 1.0.

    Returns
    -------
    float
        The predicted escape rate (probability per unit time of crossing
        from one well to the other).
    """
    curvature_at_well = 8.0 * A
    curvature_at_barrier = 4.0 * A
    delta_V = A

    prefactor = np.sqrt(curvature_at_well * curvature_at_barrier) / (2.0 * np.pi)
    rate = prefactor * np.exp(-beta * delta_V)
    return rate


def find_well_positions(A=1.0, b=0.0):
    """
    Numerically locate the two well positions (minima of V) by root-
    finding V'(x) = 4*A*x*(x**2-1) + b = 0 with scipy.optimize.brentq,
    bracketing around x=-1 (left well) and x=+1 (right well).

    This is exact (up to floating-point/solver tolerance), not a fit:
    brentq finds the true root of the analytical derivative formula from
    physics/potential.py, it does not approximate anything from
    simulated data. For b=0 this returns (-1.0, 1.0) exactly. For b != 0
    (below the spinodal tilt where the double well disappears entirely,
    b < ~1.54*A), both wells shift by a small, equal amount -- see
    PROJECT_STATE.md Sec 9 for the perturbative derivation.

    Parameters
    ----------
    A : float, optional
        Barrier height. Default 1.0.
    b : float, optional
        Tilt strength. Default 0.0.

    Returns
    -------
    tuple of float
        (x_minus, x_plus): the left (near -1) and right (near +1) well
        positions.
    """
    def derivative_at(x):
        return potential_derivative(x, A=A, b=b)

    x_minus = brentq(derivative_at, -1.5, -0.5)
    x_plus = brentq(derivative_at, 0.5, 1.5)
    return x_minus, x_plus


def free_energy_difference(A=1.0, b=0.0):
    """
    Exact free-energy difference between the two wells,
    deltaF = V(x_plus) - V(x_minus), evaluated at the numerically
    root-found well positions (see find_well_positions()). For the
    symmetric potential (b=0), this is exactly zero by the x -> -x
    symmetry of V.

    Parameters
    ----------
    A : float, optional
        Barrier height. Default 1.0.
    b : float, optional
        Tilt strength. Default 0.0.

    Returns
    -------
    float
        V(x_plus) - V(x_minus). Positive means the x=+1-side well sits
        HIGHER (and is therefore less populated) than the x=-1-side well.
    """
    x_minus, x_plus = find_well_positions(A=A, b=b)
    delta_F = potential(x_plus, A=A, b=b) - potential(x_minus, A=A, b=b)
    return delta_F


def boltzmann_population_ratio(beta, A=1.0, b=0.0):
    """
    Exact equilibrium population ratio between the two wells,
    P(x_plus) / P(x_minus) = exp(-beta * deltaF), from the Boltzmann
    distribution P(x) ~ exp(-beta*V(x)) applied to the two well minima.
    For the symmetric potential (b=0), deltaF=0 so this is exactly 1
    (equal populations); for b != 0 the wells are unequally populated.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    A : float, optional
        Barrier height. Default 1.0.
    b : float, optional
        Tilt strength. Default 0.0.

    Returns
    -------
    float
        P(x_plus) / P(x_minus).
    """
    delta_F = free_energy_difference(A=A, b=b)
    ratio = np.exp(-beta * delta_F)
    return ratio
