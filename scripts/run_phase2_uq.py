"""
Phase 2: are the MSM's Bayesian credible intervals honest uncertainties?

Uses Phase 1's saved per-replica results (results/arrhenius_sweep_raw.npz);
no new simulation. For every resolved beta it reports:
- coverage: how many replicas' 90% Bayesian credible intervals contain the
  exact chain rate. Honest 90% intervals contain it about 90% of the time.
- width: the standard deviation implied by the Bayesian interval, compared
  with the actual spread of the replica rates.

The Bayesian posterior uses deeptime's "effective" transition counts, which
here are only about 20% below the overlapping sliding counts although those
are correlated over a whole lag. It therefore acts as if it had far more
independent data than it does; this report measures by how much.
The replica spread, not the Bayesian width, is the uncertainty used by the
Phase 1 gate and the Phase 3 Validator.

Run from the project root: python -m scripts.run_phase2_uq (seconds).
"""

import numpy as np
from scipy import stats

from physics.known_answers import euler_maruyama_relaxation_rate_0d
from scripts.run_phase1_benchmark import CONFIDENCE, RAW_RESULTS_PATH, BARRIER_HEIGHT


def summarize_interval_honesty(rates, ci_lower, ci_upper, chain_rate):
    """
    For one beta's resolved replicas, return
    (n_covering, n_resolved, bayesian_sigma, replica_sigma), with both
    sigmas relative to the mean rate.
    """
    resolved = ~np.isnan(rates)
    rates, ci_lower, ci_upper = rates[resolved], ci_lower[resolved], ci_upper[resolved]

    n_covering = int(np.sum((ci_lower <= chain_rate) & (chain_rate <= ci_upper)))

    # A central interval at this confidence spans +-z standard deviations
    z_score = stats.norm.ppf(0.5 + CONFIDENCE / 2.0)
    bayesian_sigma = np.mean((ci_upper - ci_lower) / (2.0 * z_score)) / rates.mean()
    replica_sigma = rates.std(ddof=1) / rates.mean()
    return n_covering, len(rates), bayesian_sigma, replica_sigma


def main():
    results = np.load(RAW_RESULTS_PATH)
    dt = float(results["dt"])

    print(f"beta | {CONFIDENCE:.0%} intervals containing exact rate | "
          f"Bayesian sigma | replica sigma | ratio")
    total_covering, total_resolved = 0, 0
    for i, beta in enumerate(results["beta_values"]):
        if np.sum(~np.isnan(results["rate"][i])) < 2:
            print(f"{beta:4.0f} | unresolved")
            continue
        chain_rate = euler_maruyama_relaxation_rate_0d(beta=beta, dt=dt, A=BARRIER_HEIGHT)
        n_covering, n_resolved, bayesian_sigma, replica_sigma = summarize_interval_honesty(
            results["rate"][i], results["ci_lower"][i], results["ci_upper"][i], chain_rate,
        )
        total_covering += n_covering
        total_resolved += n_resolved
        print(f"{beta:4.0f} | {n_covering}/{n_resolved} | {bayesian_sigma:.2%} | "
              f"{replica_sigma:.2%} | {replica_sigma / bayesian_sigma:.1f}x")

    print(f"\nPooled coverage: {total_covering}/{total_resolved} "
          f"= {total_covering / total_resolved:.0%} (honest intervals: about {CONFIDENCE:.0%})")


if __name__ == "__main__":
    main()
