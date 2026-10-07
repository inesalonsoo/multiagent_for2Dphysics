"""
Phase 1 benchmark: sweep beta, measure the MSM relaxation rate from simulated
trajectories, and check it against the exact rate.

Per replica: simulate (physics/simulate_0d.py) -> features -> 50 k-means
microstates -> MSM at a self-consistent lag (pipeline/msm.py).

Reference value: euler_maruyama_relaxation_rate_0d(), the exact rate of the
discrete chain the simulation actually runs (dt = 0.01). It is about 1.2%
above the continuous-time exact rate. Eyring-Kramers is shown only as the
large-beta limit.

Gates, fixed before running:
1. Two metastable states in every resolved replica: t_2 / t_3 above
   MIN_TIMESCALE_SEPARATION and Chapman-Kolmogorov error below MAX_CK_ERROR.
2. At every resolved beta, the replica mean rate is consistent with the
   exact chain rate: their difference lies within the two-sided 99%
   Student t interval (N_REPLICAS - 1 degrees of freedom).
A replica is unresolved when its trajectory spans fewer than 20 slowest
timescales (pipeline.msm.choose_lagtime returns None).

Run from the project root: python -m scripts.run_phase1_benchmark (about
5 hours, mostly the Bayesian intervals at long lags).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from physics.simulate_0d import run_trajectory_0d
from physics.known_answers import (
    eyring_kramers_rate_0d, exact_relaxation_rate_0d, euler_maruyama_relaxation_rate_0d,
)
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.msm import (
    MAX_CK_ERROR, MIN_TIMESCALE_SEPARATION, build_msm, choose_lagtime,
    chapman_kolmogorov_error, timescale_separation,
)
from pipeline.uq import compute_rate_credible_interval

DT = 0.01
N_STEPS = 15_000_000
N_REPLICAS = 6
N_CLUSTERS = 50
BETA_VALUES = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
BARRIER_HEIGHT = 1.0
CONFIDENCE = 0.90  # Bayesian credible intervals, analyzed in Phase 2
RAW_RESULTS_PATH = "results/arrhenius_sweep_raw.npz"

REPLICA_FIELDS = ("rate", "lagtime", "separation", "ck_error",
                  "ci_lower", "ci_upper", "population_imbalance")


def analyze_replica(beta, seed):
    """
    Simulate one trajectory at this beta and measure it. Returns a dict
    with one value per REPLICA_FIELDS entry, all NaN if the slowest
    timescale cannot be resolved.
    """
    trajectory = run_trajectory_0d(n_steps=N_STEPS, seed=seed, beta=beta, dt=DT, A=BARRIER_HEIGHT)
    features = compute_features(trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=N_CLUSTERS, seed=42)

    lagtime = choose_lagtime(discrete_trajectory)
    if lagtime is None:
        return {field: np.nan for field in REPLICA_FIELDS}

    msm = build_msm(discrete_trajectory, lagtime)
    populations = msm.pcca(n_metastable_sets=2).coarse_grained_stationary_probability
    _, ci_lower, ci_upper = compute_rate_credible_interval(
        discrete_trajectory, lagtime=lagtime, dt=DT, confidence=CONFIDENCE,
    )

    # Rate in physical time: 1 / (slowest timescale in frames * time per frame)
    rate = 1.0 / (msm.timescales()[0] * DT)
    return dict(
        rate=rate, lagtime=lagtime, separation=timescale_separation(msm),
        ck_error=chapman_kolmogorov_error(discrete_trajectory, lagtime),
        ci_lower=ci_lower, ci_upper=ci_upper,
        population_imbalance=abs(populations[0] - populations[1]),
    )


def run_beta_sweep():
    """
    Run analyze_replica() for every beta and replica. Returns a dict of
    arrays of shape (len(BETA_VALUES), N_REPLICAS), one per REPLICA_FIELDS.
    Seeds are int(beta * 1000) + replica index.
    """
    sweep = {field: np.full((len(BETA_VALUES), N_REPLICAS), np.nan) for field in REPLICA_FIELDS}

    for beta_index, beta in enumerate(BETA_VALUES):
        for replica_index in range(N_REPLICAS):
            replica = analyze_replica(beta, seed=int(beta * 1000) + replica_index)
            for field in REPLICA_FIELDS:
                sweep[field][beta_index, replica_index] = replica[field]

        n_resolved = np.sum(~np.isnan(sweep["rate"][beta_index]))
        print(f"beta={beta}: {n_resolved}/{N_REPLICAS} replicas resolved, "
              f"median lag={np.nanmedian(sweep['lagtime'][beta_index]):.0f} frames")

    return sweep


def summarize_beta(rates, beta):
    """
    Compare one beta's replica rates with the exact chain rate. Returns a
    dict with the mean, its standard error, the exact rates, and whether
    the mean agrees at 99% confidence. Unresolved replicas (NaN) are
    dropped; with fewer than 2 resolved replicas the beta is not gated.
    """
    resolved_rates = rates[~np.isnan(rates)]
    n_resolved = len(resolved_rates)
    chain_rate = euler_maruyama_relaxation_rate_0d(beta=beta, dt=DT, A=BARRIER_HEIGHT)
    summary = dict(n_resolved=n_resolved, chain_rate=chain_rate,
                   exact_rate=exact_relaxation_rate_0d(beta=beta, A=BARRIER_HEIGHT))
    if n_resolved < 2:
        return dict(summary, mean=np.nan, sem=np.nan, agrees=None)

    mean = resolved_rates.mean()
    sem = resolved_rates.std(ddof=1) / np.sqrt(n_resolved)
    # Two-sided 99% critical value of Student's t
    t_critical = stats.t.ppf(0.995, df=n_resolved - 1)
    agrees = abs(mean - chain_rate) <= t_critical * sem
    return dict(summary, mean=mean, sem=sem, agrees=agrees)


def make_arrhenius_plot(summaries, out_path):
    """
    Two panels sharing the beta axis. Top: measured rates (mean +- SEM) with
    the exact rate and the Eyring-Kramers limit, log scale. Bottom: percent
    deviation of each mean from the exact chain rate, linear scale.
    """
    resolved = [s for s in summaries if s["agrees"] is not None]
    betas = np.array([s["beta"] for s in resolved])
    means = np.array([s["mean"] for s in resolved])
    sems = np.array([s["sem"] for s in resolved])
    chain_rates = np.array([s["chain_rate"] for s in resolved])

    beta_fine = np.linspace(BETA_VALUES.min(), BETA_VALUES.max(), 100)
    exact_curve = [exact_relaxation_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in beta_fine]
    kramers_curve = [2.0 * eyring_kramers_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in beta_fine]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                            height_ratios=[2.2, 1.3])
    ax_top.errorbar(betas, means, yerr=sems, fmt="o", color="tab:blue", capsize=3,
                    label="MSM relaxation rate (mean $\\pm$ SEM, 6 replicas)")
    ax_top.plot(beta_fine, exact_curve, color="black", label="exact rate $\\lambda_2$")
    ax_top.plot(beta_fine, kramers_curve, color="tab:red", linestyle="--",
                label="$2\\times$ Eyring-Kramers (large-$\\beta$ limit)")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("relaxation rate (1 / time)")
    ax_top.set_title("0-D double well: measured vs exact relaxation rate", fontsize=13)
    ax_top.legend(fontsize=8.5)

    ax_bottom.axhline(0.0, color="dimgray", linestyle=":")
    ax_bottom.errorbar(betas, (means / chain_rates - 1.0) * 100.0,
                       yerr=sems / chain_rates * 100.0, fmt="o", color="tab:blue", capsize=3)
    ax_bottom.set_xlabel(r"inverse temperature $\beta$")
    ax_bottom.set_ylabel(f"deviation from exact\nchain rate, dt={DT} (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def check_gates(sweep, summaries):
    """Print every gate result and raise RuntimeError if any gate fails."""
    resolved = ~np.isnan(sweep["rate"])
    states_ok = ((sweep["separation"] > MIN_TIMESCALE_SEPARATION)
                 & (sweep["ck_error"] < MAX_CK_ERROR))[resolved]
    print(f"\nGate 1, two metastable states: {states_ok.sum()}/{states_ok.size} resolved replicas "
          f"(min t2/t3={np.nanmin(sweep['separation']):.1f}, "
          f"max CK error={np.nanmax(sweep['ck_error']):.4f})")

    print("\nGate 2, rate vs exact chain rate (99% confidence):")
    for s in summaries:
        if s["agrees"] is None:
            print(f"  beta={s['beta']}: unresolved ({s['n_resolved']}/{N_REPLICAS} replicas), not gated")
            continue
        print(f"  beta={s['beta']}: measured/chain={s['mean'] / s['chain_rate']:.4f} "
              f"+- {s['sem'] / s['chain_rate']:.4f}, exact/EK="
              f"{s['exact_rate'] / (2.0 * eyring_kramers_rate_0d(beta=s['beta'])):.3f}, "
              f"{'PASS' if s['agrees'] else 'FAIL'}")

    rates_ok = [s["agrees"] for s in summaries if s["agrees"] is not None]
    if not np.all(states_ok) or not np.all(rates_ok):
        raise RuntimeError("Phase 1 gate failed, see above. Report it; do not loosen the gate.")


def main():
    sweep = run_beta_sweep()
    np.savez(RAW_RESULTS_PATH, beta_values=BETA_VALUES, dt=DT, **sweep)
    print(f"saved {RAW_RESULTS_PATH}")

    summaries = [dict(summarize_beta(sweep["rate"][i], beta), beta=beta)
                 for i, beta in enumerate(BETA_VALUES)]
    make_arrhenius_plot(summaries, "results/arrhenius.png")
    check_gates(sweep, summaries)

    imbalance = sweep["population_imbalance"]
    print(f"\nSymmetric populations: mean |p1 - p2| = {np.nanmean(imbalance):.3f} "
          f"(expected 0 up to sampling noise)")
    print("\nPhase 1 benchmark PASSED.")


if __name__ == "__main__":
    main()
