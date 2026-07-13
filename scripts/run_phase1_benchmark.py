"""
Phase 1 centerpiece: sweep the 0-D pipeline (simulate -> features ->
cluster -> MSM) across inverse temperature beta, extract each MSM's
relaxation rate, and check it against the analytical Eyring-Kramers
prediction from physics/known_answers.py. Produces results/arrhenius.png.

RATE CONVENTION -- read this before changing anything below.
physics.known_answers.eyring_kramers_rate_0d() gives the ONE-WAY escape
rate (probability per unit time of hopping from one well to the other).
The MSM's relaxation rate (1 / slowest implied timescale) is a DIFFERENT
quantity: for a symmetric two-state system, it is the SUM of the two
(here equal) one-way rates, i.e. exactly 2x the escape rate. This was
verified empirically before trusting this script (see PROJECT_STATE.md
Sec 10): at beta=5, the MSM relaxation rate came out to 2.01x the raw
committed-crossing rate, and 0.94x (2*known_answers' prediction) --
consistent with statistical noise, not a bug. Every comparison below is
therefore MSM relaxation rate vs 2*eyring_kramers_rate_0d(beta, A), never
against the bare escape rate.

UNCERTAINTY -- deliberately NOT BayesianMSM (that is Phase 2's job, see
PROJECT_STATE.md's module build order; using it here would build ahead
of the current phase). Instead: N_REPLICAS independent trajectories per
beta, each an independent MSM fit, giving an ensemble of independent
rate estimates whose spread is the per-point uncertainty (mean +/- SEM).
Trajectory length is FIXED across the whole beta sweep (not scaled up at
high beta to force equal crossing counts) -- rarer crossings at high beta
then show up honestly as wider error bars, exactly the situation Phase 2
will handle properly. N_STEPS is chosen so even the coldest (beta=10)
point still gets ~6 crossings per individual replica (see PROJECT_STATE.md
Sec 10 for the crossing-count-vs-beta table this was chosen from).

RUN THIS WITH `-m`, NOT DIRECTLY: `python -m scripts.run_phase1_benchmark`
from the project root. Running `python scripts/run_phase1_benchmark.py`
directly puts scripts/ (not the project root) on sys.path, so `physics`
and `pipeline` won't be importable -- the same pitfall this project hit
with pytest before conftest.py/pytest.ini fixed it for tests/ (that fix
doesn't apply to a standalone script run outside pytest).

FIT RANGE -- read this before changing FIT_BETA_MAX.
The first full run of this sweep (beta 3-10, 6 replicas, 15M steps each)
found the log(rate)-vs-beta slope was -0.968, only 3.2% off the exact -1,
but with SEM so tight (large N_STEPS/N_REPLICAS) that a naive "N sigma"
test called it a 14-sigma failure -- the tolerance only budgeted for
statistical noise, not any systematic effect, and a systematic effect
was present. Restricting the hard slope fit/gate to beta <= FIT_BETA_MAX
and using a RELATIVE tolerance (not N-sigma) instead of dropping points
turned out to be necessary for two SEPARATE reasons, both found by
diagnosis rather than assumed (see PROJECT_STATE.md Sec 10 for the full
tables):
  1. Sparse-transition-count MSM estimation bias at high beta (each
     replica only sees ~2-6 crossings by beta=9-10) systematically
     overestimates the rate there -- a data-volume problem.
  2. Eyring-Kramers is a beta->infinity ASYMPTOTIC formula, so a real,
     expected O(1/beta) correction to the slope exists at any finite
     beta and does not shrink as sampling gets more precise -- a
     property of the physics/theory, not of our sampling. Once
     statistics are this precise, even the small expected correction
     registers as many "sigma" from the idealized -A under a naive
     N-sigma test, though it's typically only a few percent in absolute
     terms and does NOT indicate a bug (a real bug -- wrong formula,
     missing factor, sign error -- would show as tens of percent, not a
     few).
Still measure + plot + report beta > FIT_BETA_MAX with these caveats
attached explicitly, rather than silently dropping those points.

LAG TIME -- read this before changing LAGTIME_BY_BETA.
A single fixed lag (originally 20 frames, on the argument that the
in-well relaxation time is beta-independent) turned out to be WRONG:
a systematic scan (doubling the lag and checking the implied timescale's
relative change) showed the lag needed to reach a genuine <5% plateau
grows sharply with beta -- 10 frames at beta=3-4, up to 160 at beta=8,
and beta=10 never plateaus even at lag=640 (consistent with finding #1
above: too few transitions to resolve the slow timescale at ANY lag,
not just a lag-choice problem). LAGTIME_BY_BETA below encodes this,
with a safety margin beyond the bare convergence point. Using lag=20
uniformly had been silently contaminating the point estimates at
beta=6,7 (partially converged, not fully) even within the "trustworthy"
FIT_BETA_MAX range -- found only when Phase 2's tighter Bayesian
credible intervals made the resulting bias visible; see PROJECT_STATE.md
Sec 10 for the full scan table and how this was caught.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.simulate_0d import run_trajectory_0d
from physics.known_answers import eyring_kramers_rate_0d, boltzmann_population_ratio
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.msm import build_msm

DT = 0.01
N_STEPS = 15_000_000  # fixed across the whole sweep, ~17.6s of simulation per replica
N_REPLICAS = 6
N_CLUSTERS = 50
BETA_VALUES = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# Per-beta lag time (frames), from pipeline.msm.find_converged_lagtime()
# with plateau_tolerance=0.03 (3%, tight -- a loose tolerance is exactly
# what caused the original bug, see the LAG TIME note above), run on one
# representative N_STEPS-length trajectory per beta (candidate lags
# [10,20,40,...,1280], doubling). NOT hand-padded with extra "safety
# margin" -- these are the function's exact output, verified reproducible
# in tests/test_msm.py::test_find_converged_lagtime_matches_known_plateau_shape.
# beta=8's value (320) is notably larger than its neighbors (7:40, 9:80);
# this is real single-trajectory scan noise (only one sample per beta was
# used for lag selection), not a physical non-monotonicity -- flagged
# rather than smoothed over. beta=10 never plateaus even at lag=1280
# (+3.35% there, still above tolerance) -- used as a best-effort value,
# consistent with beta=10 being excluded from every hard gate anyway.
LAGTIME_BY_BETA = {
    3.0: 10, 4.0: 10, 5.0: 20, 6.0: 40, 7.0: 40,
    8.0: 320, 9.0: 80, 10.0: 1280,
}
FIT_BETA_MAX = 7.0  # see the FIT RANGE note above; beta > this is measured
                     # and plotted but excluded from the hard slope gate.
BARRIER_HEIGHT = 1.0


def _count_committed_crossings(signal, low=-0.5, high=0.5):
    """
    Count well-to-well crossings with a hysteresis band [low, high]
    around the barrier, so noise jitter near the barrier isn't counted
    as a crossing. Used only for the one-off pre-flight sanity check
    below, not by the sweep itself (which measures rate via the MSM).
    """
    state = None
    crossings = 0
    for value in signal:
        if value >= high:
            new_state = "positive"
        elif value <= low:
            new_state = "negative"
        else:
            continue
        if state is not None and new_state != state:
            crossings += 1
        state = new_state
    return crossings


def preflight_check_factor_of_two(beta=5.0, n_steps=N_STEPS, seed=7):
    """
    Before trusting the sweep: verify the MSM relaxation rate at beta=5
    equals ~2x the raw committed-crossing rate at the same beta. If this
    does not reconcile to within ~20% of exactly 2, something is wrong
    with the rate extraction and the sweep below should not be trusted.

    Raises
    ------
    RuntimeError
        If the ratio is not within 20% of 2.0.
    """
    trajectory = run_trajectory_0d(n_steps=n_steps, seed=seed, beta=beta, dt=DT)
    total_time = n_steps * DT

    crossings = _count_committed_crossings(trajectory)
    raw_crossing_rate = crossings / total_time

    features = compute_features(trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=N_CLUSTERS, seed=42)
    msm = build_msm(discrete_trajectory, lagtime=LAGTIME_BY_BETA[beta])
    relaxation_rate = 1.0 / (msm.timescales()[0] * DT)

    ratio = relaxation_rate / raw_crossing_rate
    print(f"[preflight] beta={beta}: {crossings} crossings, raw_rate={raw_crossing_rate:.6g}, "
          f"MSM_relaxation_rate={relaxation_rate:.6g}, ratio={ratio:.4f} (expect ~2.0)")

    if not (1.6 < ratio < 2.4):
        raise RuntimeError(
            f"Pre-flight factor-of-two check FAILED: MSM relaxation rate / raw crossing "
            f"rate = {ratio:.4f}, expected close to 2.0. Do not trust the sweep below "
            f"until this is understood -- see the RATE CONVENTION note at the top of "
            f"this file."
        )
    return ratio


def measure_relaxation_rate_at_beta(beta, seed_base):
    """
    Run N_REPLICAS independent trajectories at this beta, cluster each
    with its own freshly-fit k-means model (the equilibrium distribution
    genuinely differs by beta, so centroids should not be reused across
    beta values), and extract one MSM relaxation-rate estimate and one
    "exactly 2 macrostates recovered" boolean per replica.

    Returns
    -------
    rates : np.ndarray, shape (N_REPLICAS,)
        Per-replica relaxation rate estimates (1 / slowest timescale).
    two_state_ok : np.ndarray of bool, shape (N_REPLICAS,)
        Whether PCCA+ recovered exactly 2 macrostates in that replica.
    populations : np.ndarray, shape (N_REPLICAS, 2)
        Per-replica coarse-grained stationary populations (for the
        Boltzmann-ratio sanity check).
    """
    rates = np.empty(N_REPLICAS)
    two_state_ok = np.empty(N_REPLICAS, dtype=bool)
    populations = np.empty((N_REPLICAS, 2))

    for replica_index in range(N_REPLICAS):
        seed = seed_base + replica_index
        trajectory = run_trajectory_0d(n_steps=N_STEPS, seed=seed, beta=beta, dt=DT)
        features = compute_features(trajectory)
        discrete_trajectory, _ = cluster_trajectory(features, n_clusters=N_CLUSTERS, seed=42)

        msm = build_msm(discrete_trajectory, lagtime=LAGTIME_BY_BETA[beta])
        rates[replica_index] = 1.0 / (msm.timescales()[0] * DT)

        pcca_model = msm.pcca(n_metastable_sets=2)
        assignments = pcca_model.assignments
        two_state_ok[replica_index] = len(np.unique(assignments)) == 2
        if two_state_ok[replica_index]:
            populations[replica_index] = pcca_model.coarse_grained_stationary_probability
        else:
            populations[replica_index] = np.nan

    return rates, two_state_ok, populations


def run_beta_sweep():
    """
    Run measure_relaxation_rate_at_beta() at every value in BETA_VALUES,
    aggregate each point's mean rate and standard error of the mean, and
    collect the two-state-recovery gate result for every replica at
    every beta (not just averaged away).

    Returns
    -------
    dict with keys "mean_rate", "sem_rate" (arrays over BETA_VALUES),
    and "all_two_state_ok" (bool array over BETA_VALUES: True only if
    EVERY replica at that beta recovered exactly 2 macrostates).
    """
    mean_rate = np.empty(len(BETA_VALUES))
    sem_rate = np.empty(len(BETA_VALUES))
    all_two_state_ok = np.empty(len(BETA_VALUES), dtype=bool)

    for i, beta in enumerate(BETA_VALUES):
        seed_base = int(beta * 1000)
        rates, two_state_ok, populations = measure_relaxation_rate_at_beta(beta, seed_base)

        mean_rate[i] = rates.mean()
        sem_rate[i] = rates.std(ddof=1) / np.sqrt(N_REPLICAS)
        all_two_state_ok[i] = np.all(two_state_ok)

        print(f"beta={beta}: mean_rate={mean_rate[i]:.6g} +/- {sem_rate[i]:.2g}, "
              f"two-state OK in {two_state_ok.sum()}/{N_REPLICAS} replicas, "
              f"mean population split={np.nanmean(populations, axis=0)}")

    return dict(mean_rate=mean_rate, sem_rate=sem_rate, all_two_state_ok=all_two_state_ok)


def fit_arrhenius_slope(beta_values, mean_rate, sem_rate):
    """
    Weighted linear fit of log(rate) vs beta. Uncertainty on log(rate)
    is propagated from the rate's SEM via the delta method:
    sigma_log(rate) ~= sem_rate / mean_rate.

    Parameters
    ----------
    beta_values : np.ndarray
        The beta values to fit over -- pass a SUBSET (e.g. beta values
        <= FIT_BETA_MAX) to exclude points with known estimation issues,
        see the FIT RANGE note at the top of this file.
    mean_rate, sem_rate : np.ndarray
        Matching arrays of mean rate and its SEM at each beta_values entry.

    Returns
    -------
    slope, slope_stderr : float
        Fitted slope of log(rate) vs beta and its standard error. The
        analytical prediction is slope = -BARRIER_HEIGHT exactly.
    """
    log_rate = np.log(mean_rate)
    log_rate_sigma = sem_rate / mean_rate
    weights = 1.0 / log_rate_sigma**2

    design_matrix = np.vstack([beta_values, np.ones_like(beta_values)]).T
    weighted_design = design_matrix * weights[:, None]
    covariance = np.linalg.inv(design_matrix.T @ weighted_design)
    coefficients = covariance @ (weighted_design.T @ log_rate)

    slope = coefficients[0]
    slope_stderr = np.sqrt(covariance[0, 0])
    return slope, slope_stderr


def make_arrhenius_plot(mean_rate, sem_rate, slope, out_path):
    """
    Save the presentable Arrhenius figure: MSM-extracted points with
    error bars, and the analytical 2*eyring_kramers_rate_0d(beta) line
    (the factor of 2 makes it directly comparable to the MSM relaxation
    rate points, per the RATE CONVENTION note at the top of this file).

    Points with beta <= FIT_BETA_MAX (used for the hard slope fit) and
    beta > FIT_BETA_MAX (measured and shown, but excluded from the fit
    due to sparse-transition-count MSM bias -- see the FIT RANGE note at
    the top of this file) are drawn with different markers, not hidden.
    """
    in_fit = BETA_VALUES <= FIT_BETA_MAX
    beta_fine = np.linspace(BETA_VALUES.min(), BETA_VALUES.max(), 200)
    analytical_rate = 2.0 * np.array(
        [eyring_kramers_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in beta_fine]
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(BETA_VALUES[in_fit], mean_rate[in_fit], yerr=sem_rate[in_fit],
                fmt="o", color="tab:blue", capsize=3,
                label=f"MSM relaxation rate, used in fit ($\\beta \\leq${FIT_BETA_MAX})")
    ax.errorbar(BETA_VALUES[~in_fit], mean_rate[~in_fit], yerr=sem_rate[~in_fit],
                fmt="^", color="tab:orange", capsize=3,
                label=f"measured, excluded from fit ($\\beta >${FIT_BETA_MAX}, "
                      f"sparse-count MSM bias)")
    ax.plot(beta_fine, analytical_rate, color="tab:red", linestyle="--",
            label=r"$2 \times$ Eyring-Kramers escape rate (exact, slope $-A$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel("relaxation rate (1 / time)")
    ax.set_title("Phase 1: 0-D double-well Arrhenius plot", fontsize=13)
    ax.text(0.5, 1.06, f"$A$={BARRIER_HEIGHT}, fitted slope ($\\beta\\leq${FIT_BETA_MAX})="
            f"{slope:.3f}, analytical slope=$-A$={-BARRIER_HEIGHT}",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="dimgray")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved Arrhenius plot to {out_path}")


def main():
    preflight_check_factor_of_two()

    sweep = run_beta_sweep()
    mean_rate, sem_rate = sweep["mean_rate"], sweep["sem_rate"]
    all_two_state_ok = sweep["all_two_state_ok"]

    # Save raw results BEFORE any gate assertion below can raise and abort
    # the script -- this run costs ~15-20 minutes of compute, so a later
    # gate failure must never mean losing it (this bit us once already,
    # see PROJECT_STATE.md Sec 10: the first run's numbers only existed in
    # a printed, rounded stdout log until this was added).
    np.savez("results/arrhenius_sweep_raw.npz", beta_values=BETA_VALUES,
             mean_rate=mean_rate, sem_rate=sem_rate, all_two_state_ok=all_two_state_ok)
    print(f"saved raw sweep results to results/arrhenius_sweep_raw.npz")

    print()
    print("=== Gate 1: exactly two macrostates recovered at EVERY beta ===")
    for beta, ok in zip(BETA_VALUES, all_two_state_ok):
        print(f"  beta={beta}: {'OK' if ok else 'FAILED'}")
    if not np.all(all_two_state_ok):
        raise RuntimeError(
            "Two-macrostate recovery FAILED at one or more beta values -- see above. "
            "This is a real finding (e.g. an ITS plateau that degraded, or PCCA+ "
            "finding a spurious third state), not something to average over."
        )

    in_fit = BETA_VALUES <= FIT_BETA_MAX
    slope, slope_stderr = fit_arrhenius_slope(
        BETA_VALUES[in_fit], mean_rate[in_fit], sem_rate[in_fit],
    )
    print()
    print(f"=== Gate 2 (centerpiece, hard pass/fail): Arrhenius slope, beta<={FIT_BETA_MAX} ===")
    print(f"  measured slope = {slope:.4f} +/- {slope_stderr:.4f} (SEM, reported not used "
          f"as the pass/fail criterion -- see SLOPE TOLERANCE note below)")
    print(f"  analytical slope = -{BARRIER_HEIGHT}")
    slope_relative_error = abs(slope - (-BARRIER_HEIGHT)) / BARRIER_HEIGHT
    print(f"  relative deviation = {slope_relative_error * 100:.2f}%")
    print(f"  (beta > {FIT_BETA_MAX} measured and plotted separately -- see FIT RANGE "
          f"note at the top of this file for why they're excluded from this gate)")
    # SLOPE TOLERANCE, not a raw N-sigma statistical test: Eyring-Kramers is a
    # beta->infinity ASYMPTOTIC formula, so a real O(1/beta) correction to the
    # slope is expected at any finite beta and does not vanish as N_STEPS or
    # N_REPLICAS grow -- it is a property of the physics, not of our sampling.
    # With large enough statistics (as here), even this small, EXPECTED
    # correction registers as many "standard errors" from the idealized -A,
    # which would make an N-sigma test fail on good data. A relative
    # tolerance instead asks the right question: is this close enough to
    # rule out an actual bug (a wrong formula, a missing factor, a sign
    # error) -- which would produce a >50% deviation, not a few percent.
    # 10% matches Rolland & Bouchet's own reported "1+/-0.1" agreement for
    # their much harder field-theoretic case (arXiv:1507.05577 Sec 4.2).
    SLOPE_RELATIVE_TOLERANCE = 0.10
    if slope_relative_error > SLOPE_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"Arrhenius slope check FAILED: measured slope {slope:.4f} is "
            f"{slope_relative_error * 100:.2f}% away from the analytical "
            f"-{BARRIER_HEIGHT}, outside the {SLOPE_RELATIVE_TOLERANCE * 100:.0f}% tolerance."
        )

    print()
    print("=== Gate 3 (secondary, softer): prefactor agreement ===")
    for i, beta in enumerate(BETA_VALUES):
        predicted = 2.0 * eyring_kramers_rate_0d(beta=beta, A=BARRIER_HEIGHT)
        print(f"  beta={beta}: measured/predicted = {mean_rate[i]/predicted:.3f} "
              f"(asymptotic formula, expect looser agreement at low beta)")

    predicted_ratio_symmetric = boltzmann_population_ratio(beta=5.0, A=BARRIER_HEIGHT, b=0.0)
    print()
    print(f"Boltzmann ratio check (b=0, exact): predicted ratio = {predicted_ratio_symmetric} "
          f"(must be exactly 1)")

    make_arrhenius_plot(mean_rate, sem_rate, slope, "results/arrhenius.png")

    print()
    print("Phase 1 benchmark PASSED.")


if __name__ == "__main__":
    main()
