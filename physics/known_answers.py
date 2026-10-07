"""
Ground-truth answers for the 0-D double well dx = -V'(x) dt + sqrt(2/beta) dW,
with V(x) = A*(x**2-1)**2 + b*x. The MSM pipeline is checked against these.

Nothing here is fitted to simulation data. Each function is a closed-form
expression or an exact numerical solution of the analytical problem (a root
find, a quadrature, or an eigenvalue of the discretized generator).

The reference rate is exact_relaxation_rate_0d(). The Eyring-Kramers formula
is kept as its large-beta limit, not as the oracle: at beta = 3-7, twice it
is 7-11% above the exact relaxation rate.
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from physics.potential import potential, potential_derivative

# Above this beta, the relaxation rate (~exp(-beta*A)) becomes comparable to
# the eigensolver's round-off, so exact_relaxation_rate_0d() refuses to run.
MAX_BETA_FOR_EXACT_RATE = 15.0


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
    Large-beta (small-noise) limit of the one-way escape rate for the
    symmetric (b=0) double well V(x) = A*(x**2-1)**2:

        rate = sqrt(V''(well) * |V''(saddle)|) / (2*pi) * exp(-beta*deltaV)

    The ingredients follow from V''(x) = 12*A*x**2 - 4*A:
    - curvature at each well (x=+-1): V''(+-1) = 8*A.
    - curvature at the barrier (x=0): V''(0) = -4*A, so |V''(0)| = 4*A.
    - barrier height: deltaV = V(0) - V(+-1) = A.

    This is an asymptotic formula, not an exact one. Its relative error is
    of order 1/beta: the exact one-way rate (lambda_2 / 2) is 0.90, 0.92
    and 0.94 times this value at beta = 3, 5 and 7. Use exact_relaxation_rate_0d() as the reference;
    this function is kept as the beta -> infinity cross-check.

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


def exact_relaxation_rate_0d(beta, A=1.0, n_cells=4000):
    """
    Exact relaxation rate lambda_2 of the symmetric (b=0) double well at
    finite beta. This is the reference value the MSM pipeline is checked
    against.

    lambda_2 is the smallest nonzero eigenvalue of the Fokker-Planck
    generator L f = -V'(x) f' + (1/beta) f''. An MSM measures exactly this
    quantity: its slowest implied timescale is t_2 = 1/lambda_2. For two
    wells, lambda_2 = k_AB + k_BA, so it tends to 2 * eyring_kramers_rate_0d()
    as beta -> infinity.

    Method: a finite-volume (SQRA) discretization on a uniform grid. The
    discretized generator is reversible, so it can be symmetrized into a
    tridiagonal matrix and its two smallest eigenvalues computed directly.
    With 4000 cells, lambda_2 is converged to 1e-5 to 1e-4 (relative),
    worse at higher beta.
    Valid for 1 <= beta*A and beta <= MAX_BETA_FOR_EXACT_RATE.
    """
    if beta > MAX_BETA_FOR_EXACT_RATE:
        raise ValueError(f"beta={beta} exceeds MAX_BETA_FOR_EXACT_RATE={MAX_BETA_FOR_EXACT_RATE}")

    # exp(-beta*V) is negligible beyond |x| = 2.6, where V = 33*A
    grid = np.linspace(-2.6, 2.6, n_cells)
    spacing = grid[1] - grid[0]
    diffusion = 1.0 / beta
    v = potential(grid, A=A)

    # SQRA hopping rate between neighbouring cells i -> j:
    # (D / h^2) * exp(-beta * (V_j - V_i) / 2)
    rate_right = diffusion / spacing**2 * np.exp(-beta * (v[1:] - v[:-1]) / 2.0)
    rate_left = diffusion / spacing**2 * np.exp(-beta * (v[:-1] - v[1:]) / 2.0)

    # Diagonal of -L: the total rate of leaving each cell
    leaving_rate = np.zeros(n_cells)
    leaving_rate[:-1] += rate_right
    leaving_rate[1:] += rate_left

    # Symmetrizing with the Boltzmann weights replaces each pair of opposite
    # hopping rates by their geometric mean, which is D / h^2 for every pair
    coupling = np.full(n_cells - 1, diffusion / spacing**2)

    # Eigenvalues of -L in ascending order: 0 (equilibrium), then lambda_2
    two_smallest = eigh_tridiagonal(leaving_rate, -coupling, select="i",
                                    select_range=(0, 1), eigvals_only=True)
    return two_smallest[1]


def euler_maruyama_relaxation_rate_0d(beta, dt, A=1.0, n_points=800):
    """
    Exact relaxation rate of the discrete chain that physics/simulate_0d.py
    actually runs, x' = x - V'(x)*dt + sqrt(2*dt/beta) * N(0, 1).

    A finite time step biases the rate: at dt = 0.01 this chain relaxes
    about 1.2% faster than exact_relaxation_rate_0d() for beta = 3-7. A
    converged MSM built from simulated trajectories should match this value.

    Method: discretize the Gaussian transition kernel on a grid, take the
    second-largest eigenvalue mu_2 of the resulting stochastic matrix, and
    convert it to a rate, lambda_2 = -ln(mu_2) / dt. 800 points converge the
    rate to about 1e-5 (relative).
    """
    grid = np.linspace(-2.6, 2.6, n_points)
    mean_next = grid - potential_derivative(grid, A=A) * dt
    variance = 2.0 * dt / beta

    # Row i: Gaussian jump probabilities from grid[i], renormalized on the grid
    kernel = np.exp(-(grid[None, :] - mean_next[:, None]) ** 2 / (2.0 * variance))
    kernel /= kernel.sum(axis=1, keepdims=True)

    # Largest eigenvalue is 1 (equilibrium); the next one sets the relaxation
    eigenvalues = np.sort(np.linalg.eigvals(kernel).real)[::-1]
    return -np.log(eigenvalues[1]) / dt


def find_well_positions(A=1.0, b=0.0):
    """
    Numerically locate the two well positions (minima of V) by root-
    finding V'(x) = 4*A*x*(x**2-1) + b = 0 with scipy.optimize.brentq,
    bracketing around x=-1 (left well) and x=+1 (right well).

    This is exact up to solver tolerance. For b=0 it returns (-1.0, 1.0).
    For 0 < |b| < 8*A/(3*sqrt(3)) (about 1.54*A, beyond which one well
    disappears), both wells shift by nearly the same small amount (equal
    to first order in b).

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


def energy_difference_of_minima(A=1.0, b=0.0):
    """
    Potential energy difference between the two minima, V(x_plus) - V(x_minus).
    It is exactly 0 for b=0 (by the x -> -x symmetry) and 2*b + O(b**3) for
    small b. Positive means the x=+1 well sits higher.

    This is not a free-energy difference: basin populations also depend on
    the well widths. Use boltzmann_population_ratio() for populations.
    """
    x_minus, x_plus = find_well_positions(A=A, b=b)
    delta_V = potential(x_plus, A=A, b=b) - potential(x_minus, A=A, b=b)
    return delta_V


def boltzmann_population_ratio(beta, A=1.0, b=0.0):
    """
    Exact equilibrium population ratio of the two basins, P(+) / P(-).

    Each basin's population is the integral of the Boltzmann weight
    exp(-beta*V(x)) over that basin, split at the barrier top. This includes
    the different well widths, which exp(-beta*deltaV) of the minima alone
    misses (by 4-10% at b = 0.1-0.2). For b=0 the ratio is 1 by symmetry.
    """
    x_minus, x_plus = find_well_positions(A=A, b=b)

    # The barrier top is the root of V' between the two wells
    barrier_top = brentq(lambda x: potential_derivative(x, A=A, b=b),
                         x_minus + 0.01, x_plus - 0.01)

    # Shift by the lowest minimum so the integrand stays O(1)
    lowest_energy = min(potential(x_minus, A=A, b=b), potential(x_plus, A=A, b=b))

    def boltzmann_weight(x):
        return np.exp(-beta * (potential(x, A=A, b=b) - lowest_energy))

    # 2 units beyond each well, V has grown by ~60*A, so the weight is negligible
    population_minus = quad(boltzmann_weight, x_minus - 2.0, barrier_top, points=[x_minus])[0]
    population_plus = quad(boltzmann_weight, barrier_top, x_plus + 2.0, points=[x_plus])[0]
    return population_plus / population_minus
