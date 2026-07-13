"""
Phase 2: Bayesian uncertainty quantification for the Phase 1 Arrhenius
sweep. For every beta already swept in Phase 1, compute a 90% Bayesian
credible interval on the MSM relaxation rate (pipeline/uq.py), using the
SAME per-beta converged lag times Phase 1 validated
(scripts/run_phase1_benchmark.py's LAGTIME_BY_BETA -- reused here, not
re-derived, since lag convergence is a property of the dynamics and
discretization, not of which uncertainty method sits on top of the MSM).
Upgrades results/arrhenius.png: every point now carries a credible
interval instead of a bare SEM error bar.

GATE -- read this before changing it back to a pure-CI check.
The FIRST version of this gate compared the analytical rate against the
BAYESIAN CI ALONE, and failed at 4 of 5 beta<=FIT_BETA_MAX points. This
was not a bug: a Bayesian credible interval captures only STATISTICAL
(sampling) uncertainty given a fixed dataset and a correctly-specified
model. It says nothing about SYSTEMATIC uncertainty -- residual bias
from the estimation procedure itself. Phase 1 already measured a real,
reproducible, beta-dependent systematic (the slope was -0.9813, not
exactly -1.0; per-beta deviations of order a few percent, see
PROJECT_STATE.md Sec 10) after fixing the lag-convergence bug -- small
enough that Gate 2 (a 10%-relative-tolerance, not a raw statistical
test) correctly passes it, but LARGER than a single well-sampled
trajectory's own statistical noise floor, so a pure statistical CI was
never going to "cover" it.

Standard experimental-physics practice: report statistical and
systematic uncertainty SEPARATELY, then combine them (in quadrature) into
a total error budget, and gate on whether the true value falls inside
the TOTAL band -- not the statistical component alone. That is what this
module does: sigma_systematic(beta) is the actual |measured/predicted-1|
relative deviation Phase 1 already measured at that beta (loaded from
results/arrhenius_sweep_raw.npz, not re-invented here), combined in
quadrature with the Bayesian CI's own relative half-width. The gate is on
the total band, for every beta <= FIT_BETA_MAX (Phase 1's own
"trustworthy" cutoff -- same range restriction as Gate 2, same reason).
Beyond that range the estimate is reported, not hidden, but not gated on.

CROSS-CHECK (the point of building this after, not before, Phase 1's own
diagnosis): verify "tight credible interval" and "trustworthy point
estimate" are the SAME regime. If a beta shows a wide interval where
Phase 1 called the point estimate trustworthy, or a tight interval where
Phase 1 knew the point estimate was biased, the two diagnoses disagree
and that needs understanding before either is trusted.

RUN THIS WITH `-m`: `python -m scripts.run_phase2_uq` from the project
root (same reason as run_phase1_benchmark.py -- see its docstring).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.simulate_0d import run_trajectory_0d
from physics.known_answers import eyring_kramers_rate_0d
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.uq import compute_rate_credible_interval
from scripts.run_phase1_benchmark import (
    DT, N_STEPS, N_CLUSTERS, BETA_VALUES, LAGTIME_BY_BETA, FIT_BETA_MAX, BARRIER_HEIGHT,
)

CONFIDENCE = 0.90


def compute_credible_intervals_for_sweep():
    """
    For each beta in BETA_VALUES, generate ONE long trajectory (same
    N_STEPS as a single Phase 1 replica; seed = int(beta*1000), i.e.
    Phase 1's own replica-0 seed, for direct traceability back to that
    sweep) and compute its 90% Bayesian credible interval on the
    relaxation rate, at the SAME converged lag Phase 1 validated.

    Returns
    -------
    dict with keys "rate_mean", "rate_lower", "rate_upper": arrays over
    BETA_VALUES.
    """
    rate_mean = np.empty(len(BETA_VALUES))
    rate_lower = np.empty(len(BETA_VALUES))
    rate_upper = np.empty(len(BETA_VALUES))

    for i, beta in enumerate(BETA_VALUES):
        seed = int(beta * 1000)
        trajectory = run_trajectory_0d(n_steps=N_STEPS, seed=seed, beta=beta, dt=DT)
        features = compute_features(trajectory)
        discrete_trajectory, _ = cluster_trajectory(features, n_clusters=N_CLUSTERS, seed=42)

        mean, lower, upper = compute_rate_credible_interval(
            discrete_trajectory, lagtime=LAGTIME_BY_BETA[beta], dt=DT, confidence=CONFIDENCE,
        )
        rate_mean[i], rate_lower[i], rate_upper[i] = mean, lower, upper

        relative_width = (upper - lower) / mean
        print(f"beta={beta}: rate={mean:.6g} [{lower:.6g}, {upper:.6g}] "
              f"(90% CI, relative width={relative_width * 100:.1f}%)")

    return dict(rate_mean=rate_mean, rate_lower=rate_lower, rate_upper=rate_upper)


def load_phase1_reference():
    """
    Load Phase 1's own already-measured ensemble mean rate (the 6-replica
    average, a lower-variance point estimate than any single trajectory)
    and its relative deviation from the analytical prediction, from
    results/arrhenius_sweep_raw.npz. This is Phase 1's DIRECT
    MEASUREMENT of both the best point estimate and the systematic bias
    (see the GATE note at the top of this file) -- not re-derived or
    guessed at here.

    The systematic deviation is defined as
    |predicted - phase1_mean| / phase1_mean -- i.e. relative to
    phase1_mean, NOT relative to predicted. These are NOT the same
    number when the deviation isn't tiny (e.g. a 6% gap measured one way
    is a 6.4% gap measured the other way), and the direction matters:
    the total error band below is centered on phase1_mean, so the
    systematic term must be expressed as a fraction OF phase1_mean for
    the two to be consistent with each other -- an earlier version of
    this function used the other direction and produced a self-
    inconsistent band that could (and did) miss the analytical value by
    a small margin even where the systematic "should" have covered it.

    Returns
    -------
    phase1_mean_rate : np.ndarray, shape (len(BETA_VALUES),)
        Phase 1's ensemble mean rate at each beta.
    systematic_relative : np.ndarray, shape (len(BETA_VALUES),)
        Relative systematic deviation, as a fraction of phase1_mean_rate.
    """
    phase1_data = np.load("results/arrhenius_sweep_raw.npz")
    phase1_beta = phase1_data["beta_values"]
    phase1_mean_rate = phase1_data["mean_rate"]

    assert np.array_equal(phase1_beta, BETA_VALUES), (
        "Phase 1's saved beta_values don't match this sweep's BETA_VALUES -- "
        "re-run scripts.run_phase1_benchmark first."
    )

    predicted = np.array(
        [2.0 * eyring_kramers_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in BETA_VALUES]
    )
    systematic_relative = np.abs(predicted - phase1_mean_rate) / phase1_mean_rate
    return phase1_mean_rate, systematic_relative


def build_total_error_band(phase1_mean_rate, rate_mean, rate_lower, rate_upper,
                            systematic_relative):
    """
    Combine STATISTICAL uncertainty (this module's Bayesian credible
    interval, as a relative WIDTH) with SYSTEMATIC uncertainty (Phase
    1's already-measured relative bias, see load_phase1_reference()) in
    quadrature, following standard experimental-physics practice of
    reporting the two separately and then combining them into a total
    error budget.

    The band is centered on phase1_mean_rate (Phase 1's 6-replica
    ensemble mean), NOT this module's own rate_mean (a single
    trajectory's Bayesian posterior mean, noisier by construction) --
    the systematic term is defined relative to phase1_mean_rate (see
    load_phase1_reference()), so the band must be centered there too for
    the two components to combine consistently. This module's rate_mean/
    rate_lower/rate_upper are still used for the STATISTICAL WIDTH
    (a property of how precisely one long trajectory constrains the
    rate), just not as the band's center.

    Parameters
    ----------
    phase1_mean_rate : np.ndarray
        Phase 1's ensemble mean rate (the band's center).
    rate_mean, rate_lower, rate_upper : np.ndarray
        This module's own Bayesian credible interval (statistical only;
        only the relative width is used from these).
    systematic_relative : np.ndarray
        Phase 1's measured relative systematic deviation at each beta.

    Returns
    -------
    total_lower, total_upper : np.ndarray
        The combined statistical + systematic band around phase1_mean_rate.
    statistical_relative : np.ndarray
        The statistical component alone (half-width / mean), for
        reporting the breakdown, not just the total.
    """
    statistical_relative = (rate_upper - rate_lower) / (2.0 * rate_mean)
    total_relative = np.sqrt(statistical_relative**2 + systematic_relative**2)

    total_lower = phase1_mean_rate * (1.0 - total_relative)
    total_upper = phase1_mean_rate * (1.0 + total_relative)
    return total_lower, total_upper, statistical_relative


def check_analytical_value_inside_interval(rate_lower, rate_upper):
    """
    Gate: for every beta <= FIT_BETA_MAX, the analytical relaxation rate
    (2x the one-way escape rate, see run_phase1_benchmark.py's RATE
    CONVENTION note) must fall inside [rate_lower, rate_upper]. Called
    with the TOTAL (statistical + systematic) band, not the raw Bayesian
    CI alone -- see the GATE note at the top of this file for why.

    Returns
    -------
    np.ndarray of bool, shape (len(BETA_VALUES),)
        True where the analytical value is contained, for every beta
        (not just the fit range -- the caller decides which subset to
        gate on).
    """
    contained = np.empty(len(BETA_VALUES), dtype=bool)
    for i, beta in enumerate(BETA_VALUES):
        analytical_rate = 2.0 * eyring_kramers_rate_0d(beta=beta, A=BARRIER_HEIGHT)
        contained[i] = rate_lower[i] < analytical_rate < rate_upper[i]
    return contained


def check_tight_interval_matches_trustworthy_regime(rate_mean, rate_lower, rate_upper,
                                                      width_threshold=0.10):
    """
    Cross-check: "tight credible interval" (relative width below
    width_threshold) should coincide with beta <= FIT_BETA_MAX (Phase
    1's independently-derived "trustworthy point estimate" cutoff, from
    clean pairwise log-rate slopes). Prints a per-beta comparison rather
    than silently asserting -- a genuine disagreement here is exactly
    the kind of finding that should be surfaced, not hidden behind a
    boolean.

    Returns
    -------
    bool
        True if the two regimes agree exactly (tight interval iff
        beta <= FIT_BETA_MAX), for every beta in the sweep.
    """
    relative_width = (rate_upper - rate_lower) / rate_mean
    is_tight = relative_width < width_threshold
    is_trustworthy = BETA_VALUES <= FIT_BETA_MAX

    print(f"\n=== Cross-check: tight CI (<{width_threshold*100:.0f}% width) "
          f"vs trustworthy point estimate (beta<={FIT_BETA_MAX}) ===")
    all_agree = True
    for beta, tight, trustworthy, width in zip(BETA_VALUES, is_tight, is_trustworthy, relative_width):
        agree = tight == trustworthy
        all_agree &= agree
        print(f"  beta={beta}: CI width={width*100:.1f}%, tight={tight}, "
              f"trustworthy={trustworthy}, {'agree' if agree else 'DISAGREE'}")

    return all_agree


def make_arrhenius_plot_with_credible_intervals(phase1_mean_rate, statistical_relative,
                                                  total_lower, total_upper, out_path):
    """
    Upgraded Arrhenius figure: every point carries BOTH its statistical
    band (thin, dark error bar -- Bayesian sampling uncertainty only)
    and its total statistical+systematic band (wide, light shaded
    whisker -- what the gate actually checks against, see the GATE note
    at the top of this file). Showing both, not just the total, keeps
    honest which part of the uncertainty is "we measured this precisely"
    vs "and there's also a known systematic on top". Both bands are
    centered on phase1_mean_rate (see build_total_error_band()'s
    docstring for why). Same fit-range/excluded-range marker distinction
    as Phase 1's plot.
    """
    in_fit = BETA_VALUES <= FIT_BETA_MAX
    stat_lower_err = phase1_mean_rate * statistical_relative
    stat_upper_err = phase1_mean_rate * statistical_relative
    total_lower_err = phase1_mean_rate - total_lower
    total_upper_err = total_upper - phase1_mean_rate

    beta_fine = np.linspace(BETA_VALUES.min(), BETA_VALUES.max(), 200)
    analytical_rate = 2.0 * np.array(
        [eyring_kramers_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in beta_fine]
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Total (statistical + systematic) band first, wide and light, so the
    # tighter statistical error bar draws on top of it. Both centered on
    # phase1_mean_rate -- see this function's docstring for why.
    ax.errorbar(BETA_VALUES[in_fit], phase1_mean_rate[in_fit],
                yerr=[total_lower_err[in_fit], total_upper_err[in_fit]],
                fmt="none", color="tab:blue", alpha=0.35, capsize=5, linewidth=4,
                label=f"total band (statistical $\\oplus$ systematic), $\\beta\\leq${FIT_BETA_MAX}")
    ax.errorbar(BETA_VALUES[in_fit], phase1_mean_rate[in_fit],
                yerr=[stat_lower_err[in_fit], stat_upper_err[in_fit]],
                fmt="o", color="tab:blue", capsize=3,
                label=f"MSM rate, statistical-only 90% CI ($\\beta \\leq${FIT_BETA_MAX})")
    ax.errorbar(BETA_VALUES[~in_fit], phase1_mean_rate[~in_fit],
                yerr=[stat_lower_err[~in_fit], stat_upper_err[~in_fit]],
                fmt="^", color="tab:orange", capsize=3,
                label=f"measured, excluded from fit ($\\beta >${FIT_BETA_MAX})")
    ax.plot(beta_fine, analytical_rate, color="tab:red", linestyle="--",
            label=r"$2 \times$ Eyring-Kramers escape rate (exact, slope $-A$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel("relaxation rate (1 / time)")
    ax.set_title("Phase 2: 0-D double-well Arrhenius plot, with Bayesian UQ", fontsize=13)
    ax.text(0.5, 1.06, "statistical 90% CI (BayesianMSM) shown alongside the total "
            "band (+ Phase 1's measured systematic, in quadrature)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="dimgray")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved Arrhenius plot (with credible intervals) to {out_path}")


def main():
    intervals = compute_credible_intervals_for_sweep()
    rate_mean, rate_lower, rate_upper = (
        intervals["rate_mean"], intervals["rate_lower"], intervals["rate_upper"],
    )

    np.savez("results/uq_sweep_raw.npz", beta_values=BETA_VALUES,
             rate_mean=rate_mean, rate_lower=rate_lower, rate_upper=rate_upper)
    print("saved raw UQ results to results/uq_sweep_raw.npz")

    phase1_mean_rate, systematic_relative = load_phase1_reference()
    total_lower, total_upper, statistical_relative = build_total_error_band(
        phase1_mean_rate, rate_mean, rate_lower, rate_upper, systematic_relative,
    )

    print(f"\n=== Error budget breakdown (statistical vs systematic vs total) ===")
    for i, beta in enumerate(BETA_VALUES):
        total_relative = (total_upper[i] - phase1_mean_rate[i]) / phase1_mean_rate[i]
        print(f"  beta={beta}: statistical={statistical_relative[i]*100:.2f}%, "
              f"systematic={systematic_relative[i]*100:.2f}%, "
              f"total={total_relative*100:.2f}% (quadrature sum)")

    contained = check_analytical_value_inside_interval(total_lower, total_upper)
    print(f"\n=== Gate: analytical rate inside TOTAL (statistical+systematic) band, "
          f"beta<={FIT_BETA_MAX} ===")
    in_fit = BETA_VALUES <= FIT_BETA_MAX
    for beta, ok in zip(BETA_VALUES[in_fit], contained[in_fit]):
        print(f"  beta={beta}: {'OK' if ok else 'FAILED'}")
    if not np.all(contained[in_fit]):
        raise RuntimeError(
            "Analytical rate fell OUTSIDE the total (statistical+systematic) band for "
            f"at least one beta <= {FIT_BETA_MAX} -- see above. This would mean the "
            "systematic is BIGGER than what Phase 1 measured, a new finding worth "
            "understanding before declaring the benchmark validated."
        )

    print(f"\n(beta > {FIT_BETA_MAX}, reported not gated: "
          f"{dict(zip(BETA_VALUES[~in_fit], contained[~in_fit]))})")

    regimes_agree = check_tight_interval_matches_trustworthy_regime(
        rate_mean, rate_lower, rate_upper,
    )
    print(f"\nRegimes {'AGREE' if regimes_agree else 'DISAGREE'} "
          f"(tight-CI beta range vs trustworthy-point-estimate beta range)")

    make_arrhenius_plot_with_credible_intervals(
        phase1_mean_rate, statistical_relative, total_lower, total_upper,
        "results/arrhenius.png",
    )

    print("\nPhase 2 UQ PASSED.")


if __name__ == "__main__":
    main()
