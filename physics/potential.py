"""
The double-well landscape and its slope, with an optional tilt.

    V(phi) = A * (phi**2 - 1)**2 + b * phi

- With b = 0 (the default) the landscape is symmetric: two valleys at
  phi = -1 and phi = +1, both at V = 0, separated by a hill of height A at
  phi = 0.
- A tilt b != 0 lowers one valley and raises the other (for b > 0 the
  phi > 0 valley sits higher), so the system prefers one of them. Phase 4
  uses this as a simple stand-in for symmetry breaking.
- The tilt also shifts the valleys slightly away from phi = +-1.
  physics/known_answers.py gives their exact positions and the energy
  difference between them (about 2*b for small tilts).

The slope dV/dphi is minus the force: the dynamics push phi downhill,
in the direction of -dV/dphi.
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
        deeper than the other (see the module docstring).

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
        values (physics/known_answers.py finds them exactly).
    """
    # Chain-rule derivative of A * (phi**2 - 1)**2, plus the constant
    # slope b contributed by the linear tilt term (d/dphi of b*phi = b).
    slope = 4 * A * phi * (phi**2 - 1) + b
    return slope
