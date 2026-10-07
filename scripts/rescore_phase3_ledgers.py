"""
Re-score the configs recorded in the Phase 3 convergence study under the
corrected Validator gates, without calling any LLM.

The study's runs were judged against 2 * Eyring-Kramers, which is about 8%
above the exact rate at beta=5. This script re-runs the deterministic
pipeline (agents/tools.py) on the same reference trajectory for every
recorded config, checks that it reproduces the recorded rate, and applies
the current physics checks (agents/validator.py). The agents' search paths
themselves are not re-run: they were steered by the old feedback.

Output: results/phase3_rescore.json, results/phase3_rescore.png, a printed table.
Run from the project root: python -m scripts.rescore_phase3_ledgers (~10 min).
"""

import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.loop import DT, REFERENCE_BETA, build_reference_context
from agents.schemas import PipelineConfig
from agents.tools import run_msm_pipeline
from agents.validator import _compute_physics_checks, reference_rate

LEDGER_PATTERN = "results/phase3_convergence_study/run_*_ledger.json"
OUTPUT_PATH = "results/phase3_rescore.json"
PLOT_PATH = "results/phase3_rescore.png"


def load_recorded_configs():
    """
    Return {config tuple: {"old_verdicts": [...], "recorded_rate": float}}
    for every config in the study's ledgers. Repeated configs are merged.
    """
    recorded = {}
    for path in sorted(glob.glob(LEDGER_PATTERN)):
        with open(path, encoding="utf-8") as ledger_file:
            run = json.load(ledger_file)
        for entry in run["entries"]:
            config = entry["proposal"]["config"]
            key = (config["n_clusters"], config["cluster_seed"], config["msm_lagtime"])
            record = recorded.setdefault(key, {"old_verdicts": [],
                                               "recorded_rate": entry["result"]["relaxation_rate_mean"]})
            record["old_verdicts"].append(entry["decision"]["verdict"])
    return recorded


def rescore(key, record, trajectory, rate_tolerance):
    """Re-run one config and apply the current physics checks."""
    n_clusters, cluster_seed, msm_lagtime = key
    config = PipelineConfig(n_clusters=n_clusters, cluster_seed=cluster_seed, msm_lagtime=msm_lagtime)
    result = run_msm_pipeline(config, trajectory, DT)

    two_states, rate_ok = _compute_physics_checks(result, REFERENCE_BETA, rate_tolerance)
    return dict(
        n_clusters=n_clusters, cluster_seed=cluster_seed, msm_lagtime=msm_lagtime,
        old_verdicts=record["old_verdicts"],
        reproduces_recorded_rate=abs(result.relaxation_rate_mean / record["recorded_rate"] - 1.0) < 1e-9,
        rate_over_exact=result.relaxation_rate_mean / reference_rate(REFERENCE_BETA),
        timescale_separation=result.timescale_separation,
        new_verdict="ACCEPT" if (two_states and rate_ok) else "REJECT",
    )


def make_rescore_plot(rows, rate_tolerance, out_path):
    """
    Measured rate / exact chain rate against lag time, with the Validator's
    band shaded. Filled markers: accepted by the current gates. Crosses:
    accepted by the original (Eyring-Kramers) gate.
    """
    lags = np.array([row["msm_lagtime"] for row in rows])
    ratios = np.array([row["rate_over_exact"] for row in rows])
    accepted_now = np.array([row["new_verdict"] == "ACCEPT" for row in rows])
    accepted_before = np.array(["ACCEPT" in row["old_verdicts"] for row in rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhspan(1.0 - rate_tolerance, 1.0 + rate_tolerance, color="tab:blue", alpha=0.15,
               label=f"Validator band (+-{rate_tolerance:.1%})")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.scatter(lags[accepted_now], ratios[accepted_now], color="tab:blue", zorder=3,
               label="accepted (current gates)")
    if np.any(~accepted_now):
        ax.scatter(lags[~accepted_now], ratios[~accepted_now], facecolors="none",
                   edgecolors="tab:blue", zorder=3, label="rejected (current gates)")
    ax.scatter(lags[accepted_before], ratios[accepted_before], marker="x", color="tab:red",
               s=70, zorder=4, label="accepted by original gate")
    ax.set_xscale("log")
    ax.set_xlabel("MSM lag time (frames)")
    ax.set_ylabel("measured rate / exact chain rate")
    ax.set_title("Phase 3 configs re-scored (beta=5, same trajectory)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    trajectory, _, rate_tolerance = build_reference_context(seed=7)
    print(f"tolerance: +-{rate_tolerance:.2%} around the exact chain rate "
          f"{reference_rate(REFERENCE_BETA):.6g}")

    recorded = load_recorded_configs()
    rows = [rescore(key, record, trajectory, rate_tolerance)
            for key, record in sorted(recorded.items(), key=lambda item: item[0][2])]

    print("\nlag | clusters | rate/exact | t2/t3 | old verdict(s) | new verdict")
    for row in rows:
        print(f"{row['msm_lagtime']:4d} | {row['n_clusters']:3d} | {row['rate_over_exact']:.4f} | "
              f"{row['timescale_separation']:.1f} | {'/'.join(row['old_verdicts'])} | {row['new_verdict']}")

    all_reproduced = all(row["reproduces_recorded_rate"] for row in rows)
    print(f"\nAll recorded rates reproduced exactly: {all_reproduced}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(dict(rate_tolerance=rate_tolerance, rows=rows), output_file, indent=2)
    print(f"saved {OUTPUT_PATH}")
    make_rescore_plot(rows, rate_tolerance, PLOT_PATH)


if __name__ == "__main__":
    main()
