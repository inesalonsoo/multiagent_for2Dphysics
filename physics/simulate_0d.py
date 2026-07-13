"""
The 0-D benchmark engine: a single Brownian particle moving in the same
(optionally tilted) double well used throughout this project, following

    dx = -V'(x) dt + sqrt(2/beta) dW

This is the "one degree of freedom" reference system that Rolland, Bouchet
& Simonnet (arXiv:1507.05577) use as the benchmark whose scaling laws every
extended-field result is checked against (their Sec. 2.1/3.2.1) -- there is
no spatial coupling, so this is exactly our 2D equation's gamma=1 case with
the Laplacian term dropped. Unlike the 2D field (physics/simulate.py), BOTH
the Eyring-Kramers rate and the Boltzmann population ratio between the two
wells are exact, closed-form, and directly observable from a trajectory --
see physics/known_answers.py for both formulas. This makes 0-D the clean,
always-checkable Phase 1 engine; the 2D field is Phase 4's deployment.
See PROJECT_STATE.md Sec 9 for the full reasoning behind this pivot.
"""

import numpy as np

from physics.potential import potential_derivative


def run_trajectory_0d(n_steps, seed, dt=0.01, beta=5.0, A=1.0, b=0.0, x0=1.0):
    """
    Integrate dx = -V'(x) dt + sqrt(2/beta) dW via Euler-Maruyama, where
    V(x) = A*(x**2-1)**2 + b*x is the same (optionally tilted) double well
    used throughout this project (see physics.potential).

    Named terms:
    - x(t): the scalar particle position (the 0-D analogue of phi in the
      2D field).
    - V'(x): potential_derivative(x, A, b) from physics/potential.py --
      pushes x toward the nearest well. Reusing this function (rather than
      re-deriving the formula here) keeps the 0-D and 2D engines from ever
      falling out of sync with each other.
    - beta: inverse temperature. Default 5.0, matching the Phase 1 baseline
      in PROJECT_STATE.md Sec 4.
    - sqrt(2/beta): the noise prefactor. This is the gamma=1 special case
      of the 2D field's sqrt(2*gamma/beta) -- there is no spatial mobility
      to speak of for a single point, so gamma is implicitly 1.

    Parameters
    ----------
    n_steps : int
        Number of output frames. Total simulated time is n_steps * dt.
    seed : int
        Seed for the random number generator driving the thermal noise,
        so a run is exactly reproducible.
    dt : float, optional
        Fixed Euler-Maruyama time step. Default 0.01. There is no CFL-style
        stability bound here (no spatial diffusion to destabilize) -- the
        only requirement is resolving the fastest relaxation timescale,
        1/V''(well) = 1/(8*A) = 0.125 for A=1, which dt=0.01 resolves with
        a comfortable margin (12-13 steps per relaxation time).
    beta : float, optional
        Inverse temperature, see above. Default 5.0.
    A : float, optional
        Barrier height of the untilted double well. Default 1.0.
    b : float, optional
        Tilt strength (see physics/potential.py). Default 0.0 (symmetric).
    x0 : float, optional
        Starting position. Default 1.0, i.e. starting in the x=+1 well.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps,): x(t) at simulated times dt, 2*dt, ...,
        n_steps*dt. The initial condition at t=0 is not included, matching
        the convention used by physics.simulate.run_trajectory.
    """
    rng = np.random.default_rng(seed)
    # Precompute every random draw at once (vectorized) instead of drawing
    # one Gaussian per step in the loop below -- much faster than repeated
    # small calls to the random number generator.
    noise_prefactor = np.sqrt(2.0 / beta)
    random_draws = noise_prefactor * np.sqrt(dt) * rng.standard_normal(n_steps)

    trajectory = np.empty(n_steps)
    x = x0
    for step in range(n_steps):
        drift = -potential_derivative(x, A=A, b=b) * dt
        x = x + drift + random_draws[step]
        trajectory[step] = x

    return trajectory
