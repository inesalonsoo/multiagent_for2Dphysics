"""
The 2D field version of the double well (Phase 4, not yet used for
results): every point of a 2D grid holds a value phi that feels the
double-well landscape, is coupled to its neighbors, and is shaken by
thermal noise. Integrated with py-pde.

Numerics (checked against the installed py-pde 0.57.0):
- Noise is added through the `noise=` argument of pde.PDE, which takes
  the VARIANCE of the noise, not its amplitude (see noise_variance in
  run_trajectory()).
- No py-pde solver allows adaptive time steps with noise, so a fixed step
  is used (default dt=0.005; see _check_cfl_condition()).
- The solver is explicit Euler-Maruyama ("euler"). Milstein would only
  differ for noise whose strength depends on phi; ours is constant.
"""

import numpy as np
import pde

from physics.potential import potential_derivative

# Physics parameters are the human's decisions (CLAUDE.md, hard boundary 2),
# so they are not function arguments. Under review for Phase 4: see
# PROJECT_STATE.md (the recorded plan is L = 2.5, not 10).
GRID_SHAPE = (32, 32)
DOMAIN_SIZE = (10.0, 10.0)


def _build_grid():
    """
    Build the 32x32 CartesianGrid covering a 10x10 physical domain with
    periodic boundary conditions in both directions.
    """
    grid = pde.CartesianGrid(
        bounds=[[0, DOMAIN_SIZE[0]], [0, DOMAIN_SIZE[1]]],
        shape=GRID_SHAPE,
        periodic=True,
    )
    return grid


def _check_cfl_condition(dt, gamma, dx):
    """
    Enforce the CFL (Courant-Friedrichs-Lewy) stability bound for the
    explicit-Euler treatment of the diffusive term gamma*laplacian(phi):

        dt < dx**2 / (4*gamma)

    Forward-Euler finite differencing of a 2D diffusion term is only
    numerically stable below this step size; above it, errors amplify
    every step and the field blows up (values diverge to +-inf) instead
    of diffusing smoothly. We check this explicitly at setup, with a
    real `raise` (not a bare `assert`, which Python can strip out when
    run with `-O`) so that a future change to dt, gamma, or the grid
    resolution cannot silently produce garbage output.

    Parameters
    ----------
    dt : float
        The proposed fixed integration time step.
    gamma : float
        Diffusion/mobility coefficient multiplying the Laplacian term.
    dx : float
        Grid spacing (physical size of one cell edge).
    """
    stability_bound = dx**2 / (4.0 * gamma)
    if not dt < stability_bound:
        raise ValueError(
            f"CFL condition violated: dt={dt} must be strictly less than "
            f"dx**2/(4*gamma) = {dx}**2/(4*{gamma}) = {stability_bound:.6g} "
            f"for the explicit Euler-Maruyama integrator to stay stable. "
            f"Reduce dt, reduce gamma, or coarsen the grid resolution."
        )


def _build_equation(gamma, A, noise_variance, seed, b=0.0, include_potential=True):
    """
    Build the py-pde PDE object encoding
    d(phi)/dt = gamma*laplacian(phi) - dV/dphi(phi) + noise,
    with the nonlinear dV/dphi term supplied directly by
    physics.potential.potential_derivative (via py-pde's user_funcs), so
    the two modules can never fall out of sync with each other.

    include_potential=False drops the -dV/dphi(phi) term entirely,
    leaving pure diffusion + noise. This isn't used for physics runs; it
    exists so tests/test_simulate.py can validate py-pde's noise scaling
    in isolation, without the potential's restoring force masking a
    noise-amplitude bug.
    """
    if include_potential:
        rhs_expression = "gamma_coef * laplace(phi) - dVdphi(phi)"
        # user_funcs lets the py-pde expression call our own Python
        # function directly, instead of duplicating the
        # 4*A*phi*(phi^2-1) + b formula as a second copy written out in
        # the string above.
        user_funcs = {"dVdphi": lambda phi_values: potential_derivative(phi_values, A=A, b=b)}
    else:
        rhs_expression = "gamma_coef * laplace(phi)"
        user_funcs = {}

    equation = pde.PDE(
        rhs={"phi": rhs_expression},
        user_funcs=user_funcs,
        consts={"gamma_coef": gamma},
        bc="periodic",
        noise=noise_variance,
        noise_interpretation="ito",
        rng=np.random.default_rng(seed),
    )
    return equation


def _make_storage(storage_path):
    """
    Create py-pde storage for a trajectory: an in-memory MemoryStorage
    (default), or a disk-backed FileStorage streaming to storage_path
    when given, so a long run doesn't need every frame held in RAM.
    """
    if storage_path is None:
        return pde.MemoryStorage()
    return pde.FileStorage(storage_path)


def _extract_trajectory_array(storage):
    """
    Convert py-pde storage into a plain numpy array, dropping the t=0
    initial-condition frame that storage.tracker(dt) always records
    first, so the number of returned frames matches n_steps exactly.
    """
    all_frames = np.stack([np.asarray(frame) for frame in storage.data])
    return all_frames[1:]


def run_trajectory(n_steps, seed, dt=0.005, gamma=1.0, beta=5.0, A=1.0, b=0.0,
                    initial_phi=1.0, storage_path=None, include_potential=True):
    """
    Integrate the stochastic Allen-Cahn equation

        d(phi)/dt = gamma * laplacian(phi) - dV/dphi(phi)
                    + sqrt(2*gamma/beta) * eta(r, t)

    where dV/dphi comes from the (optionally tilted) potential
    V(phi) = A*(phi**2-1)**2 + b*phi (physics/potential.py).

    on the 32x32, 10x10, periodic grid defined by GRID_SHAPE/DOMAIN_SIZE,
    using py-pde's fixed-step explicit Euler-Maruyama solver
    (solver="euler"). The default dt=0.005 satisfies the CFL stability
    bound dt < dx**2/(4*gamma) with a comfortable margin (dx=0.3125 here,
    so the bound is ~0.0244); _check_cfl_condition() enforces this for
    any dt actually passed in, see its docstring.

    Named terms of the equation:
    - phi(r, t): the scalar order parameter field on the 2D grid.
    - gamma: diffusion/mobility coefficient; how strongly neighboring
      grid points are coupled together. Default 1.0.
    - laplacian(phi): spatial Laplacian (periodic boundaries), computed
      by py-pde; it smooths phi toward its local neighborhood average.
    - dV/dphi(phi): derivative of the (optionally tilted) double-well
      potential V(phi) = A*(phi**2-1)**2 + b*phi, from
      physics.potential.potential_derivative. It pushes phi toward the
      nearest minimum (at phi = +-1 exactly only when b=0).
    - A: barrier height of the untilted double well. Default 1.0.
    - b: tilt strength (see physics/potential.py's module docstring).
      Default 0.0 (symmetric well). b != 0 makes one minimum lower than
      the other by about 2*b (physics/known_answers.py).
    - beta: inverse temperature, beta = 1/(k_B T). Default 5.0. For a
      single point the barrier is A, but for the whole field it grows
      with the area (about L**2 * A for a uniform flip), so switching
      is far rarer than beta * A suggests.
    - eta(r, t): Gaussian white noise, delta-correlated in space and time
      (thermal fluctuations).
    - sqrt(2*gamma/beta): the noise prefactor. Because gamma multiplies
      only the Laplacian, the prefactor consistent with the
      fluctuation-dissipation theorem is sqrt(2/beta); the two agree only
      at gamma=1, the only value used so far.

    Parameters
    ----------
    n_steps : int
        Number of output frames. The field is integrated for a total
        simulated time of n_steps * dt.
    seed : int
        Seed for the random number generator driving the thermal noise,
        so a run is exactly reproducible.
    dt : float, optional
        Fixed integration time step, and also the spacing (in simulated
        time) between returned frames. Default 0.005 (see CFL note
        above); must satisfy dt < dx**2/(4*gamma) or run_trajectory
        raises ValueError before integrating.
    gamma : float, optional
        Diffusion/mobility coefficient, see above. Default 1.0.
    beta : float, optional
        Inverse temperature, see above. Default 5.0.
    A : float, optional
        Barrier height of the untilted double well, see above. Default 1.0.
    b : float, optional
        Tilt strength, see above. Default 0.0 (symmetric well).
    initial_phi : float, optional
        Uniform starting value of phi across the whole grid. Default 1.0,
        i.e. the field starts sitting in the phi=+1 well.
    storage_path : str, optional
        If given, snapshots are streamed to this HDF5 file on disk via
        py-pde's FileStorage instead of being kept fully in memory during
        the run. Either way, the function still returns the complete
        trajectory as an in-memory numpy array.
    include_potential : bool, optional
        If False, drops the -dV/dphi(phi) term, leaving pure diffusion +
        noise. Default True. Only meant for the noise-amplitude sanity
        check in tests/test_simulate.py (see _build_equation()).

    Returns
    -------
    np.ndarray
        Array of shape (n_steps, 32, 32): phi(x, y) at simulated times
        dt, 2*dt, ..., n_steps*dt. The initial condition at t=0 is not
        included in the returned array.
    """
    grid = _build_grid()
    # Grid spacing is uniform and identical along both axes here (square
    # domain, square grid), so a single dx is enough for the CFL check.
    _check_cfl_condition(dt=dt, gamma=gamma, dx=grid.discretization[0])

    initial_state = pde.ScalarField(grid, data=initial_phi)

    # The physical noise PREFACTOR (multiplying eta in the SDE above) is
    # sqrt(2*gamma/beta) (equal to the fluctuation-dissipation value
    # sqrt(2/beta) at the default gamma=1; see run_trajectory's docstring).
    # py-pde's `noise=` argument expects the VARIANCE of the noise, not
    # this prefactor/amplitude, so we square it here: variance =
    # (sqrt(2*gamma/beta))**2 = 2*gamma/beta.
    noise_variance = 2.0 * gamma / beta

    equation = _build_equation(
        gamma=gamma, A=A, noise_variance=noise_variance, seed=seed, b=b,
        include_potential=include_potential,
    )
    storage = _make_storage(storage_path)

    total_time = n_steps * dt
    equation.solve(
        initial_state,
        t_range=total_time,
        dt=dt,
        solver="euler",
        tracker=storage.tracker(dt),
        # numba (py-pde's default backend) cannot JIT-compile a call into
        # our own Python potential_derivative() function; the numpy
        # backend runs it as ordinary (uncompiled) Python instead.
        backend="numpy",
    )

    trajectory = _extract_trajectory_array(storage)
    if storage_path is not None:
        storage.close()

    return trajectory
