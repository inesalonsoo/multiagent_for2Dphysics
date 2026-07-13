"""
The double-well potential and its derivative, with an optional linear
tilt that breaks the symmetry between the two wells.

    V(phi) = A * (phi**2 - 1)**2 + b * phi

- With b = 0 (the default), this is the classic symmetric double-well
  (Mexican hat cross-section): two minima at phi = -1 and phi = +1,
  both with V = 0, separated by a barrier of height A at phi = 0.
- With b != 0, the extra b*phi term tilts the landscape: it lowers V on
  the phi < 0 side and raises it on the phi > 0 side (for b > 0), so the
  two minima are no longer equally deep. This is deliberate: a purely
  symmetric double well has NO bulk free-energy difference between the
  two wells, so a droplet of one phase nucleated inside the other has no
  thermodynamic driving force to grow -- switching is governed only by
  surface tension/curvature (much more strongly suppressed than a tilted
  system, see PROJECT_STATE.md Sec. 9). Adding a modest tilt b gives the
  two wells different depths, exactly the situation that makes moire
  stacking domains (which this project's benchmark ultimately targets)
  physically interesting, so this also folds the Phase-4 "moire tilt"
  idea into the core physics rather than treating it as a separate demo.
- The tilt shifts the well POSITIONS slightly too (they are no longer
  exactly at phi = +-1) -- see physics/known_answers.py for the exact
  (numerically root-found) well positions and the free-energy difference
  between them, which is NOT simply b (though b turns out to be an
  extremely good approximation to 2*b for modest tilts -- again see
  known_answers.py).

The derivative dV/dphi is the force (well, minus the force) that drives
the deterministic part of the Allen-Cahn dynamics: phi is pushed
"downhill" on this landscape, i.e. in the direction of -dV/dphi.
"""

import numpy as np


def potential(phi, A=1.0, b=0.0):
    """
    Evaluate the (optionally tilted) double-well potential
    V(phi) = A * (phi**2 - 1)**2 + b * phi.

    Parameters
    ----------
    phi : float or np.ndarray
        The order parameter value(s) at which to evaluate the potential.
        Can be a single number or an array of field values.
    A : float, optional
        The barrier height of the untilted double well. With b = 0,
        V = 0 at the minima (phi = +-1) and V = A at the barrier
        (phi = 0). Default is 1.0.
    b : float, optional
        Tilt strength. b = 0 (default) recovers the symmetric double
        well. b != 0 breaks the phi -> -phi symmetry, making one well
        deeper than the other -- see the module docstring.

    Returns
    -------
    float or np.ndarray
        The potential energy at each input phi, same shape as phi.
    """
    # (phi**2 - 1) is zero exactly at phi = +-1, so squaring it gives
    # the two minima of the UNTILTED potential. At phi = 0 this bracket
    # equals -1, so the square is 1 and V = A there, giving the barrier
    # height. The b*phi term is added on top and is what breaks the
    # phi -> -phi symmetry when b != 0.
    well_shape = (phi**2 - 1)**2
    return A * well_shape + b * phi


def potential_derivative(phi, A=1.0, b=0.0):
    """
    Evaluate the derivative dV/dphi = 4*A*phi*(phi**2 - 1) + b.

    This is obtained by the chain rule from
    V(phi) = A * (phi**2 - 1)**2 + b*phi:
        dV/dphi = A * 2 * (phi**2 - 1) * d/dphi(phi**2 - 1) + b
                = A * 2 * (phi**2 - 1) * (2 * phi) + b
                = 4 * A * phi * (phi**2 - 1) + b

    Parameters
    ----------
    phi : float or np.ndarray
        The order parameter value(s) at which to evaluate the derivative.
    A : float, optional
        The barrier height of the untilted double well, same meaning as
        in potential(). Default 1.0.
    b : float, optional
        Tilt strength, same meaning as in potential(). Default 0.0.

    Returns
    -------
    float or np.ndarray
        The slope of the potential at each input phi, same shape as phi.
        With b = 0 this is zero exactly at phi = -1, 0, +1. With b != 0
        the three stationary points shift slightly away from these
        values -- see physics/known_answers.py for their exact locations.
    """
    # Chain-rule derivative of A * (phi**2 - 1)**2, plus the constant
    # slope b contributed by the linear tilt term (d/dphi of b*phi = b).
    slope = 4 * A * phi * (phi**2 - 1) + b
    return slope
