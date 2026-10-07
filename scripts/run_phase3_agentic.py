"""
Phase 3 study: run the real agentic loop (agents/loop.py) N_REPETITIONS
times on one fixed reference trajectory at REFERENCE_BETA, saving every
run's ledger.

What it shows:
1. Whether independent runs search differently (check_paths_differ()).
2. Which configs the Validator accepts or rejects, and how the Optimizer
   reacts to rejections (in the ledgers).
Accepted rates lie inside the Validator's band by construction (that is the
accept rule), so the plot shows where they fall within it; it is not an
independent test.

The recorded study (4 runs, 20 iterations) was judged against the old
Eyring-Kramers reference; scripts/rescore_phase3_ledgers.py re-scores its
configs under the current gates.

Makes real API calls (requires ANTHROPIC_API_KEY); never run from the test
suite, which covers the loop machinery with fake agents.

Output: results/phase3_convergence_study/run_NN_ledger.json (saved before
any analysis) and results/phase3_convergence_study.png.
Run from the project root: python -m scripts.run_phase3_agentic
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.loop import build_reference_context, run_one_real_loop
from agents.schemas import AgenticRun
from agents.validator import REFERENCE_BETA, reference_rate

N_REPETITIONS = 4
# note. The minimum that demonstrates the four qualitative properties the claim needs, not a
# stats-gathering run; run_01 is already reused from a real dry run under this same design.
LEDGER_DIR = Path("results/phase3_convergence_study")


def run_all_repetitions(n_repetitions=N_REPETITIONS, ledger_dir=None,
                         trajectory=None, search_bounds=None, rate_tolerance=None,
                         optimizer_agent=None, validator_agent=None):
    """
    Run n_repetitions independent real agentic loops on ONE fixed
    reference trajectory (built once here, reused for every repetition --
    see agents.loop.build_reference_context()'s docstring for why this
    isolates the LLM's own reasoning as the sole source of run-to-run
    variation). Persists each run's ledger to disk immediately after it
    completes, before any downstream analysis.

    RESUMABLE: if run_NN_ledger.json already exists (and is non-empty --
    a zero-byte file means a previous attempt crashed mid-write, e.g. the
    UnicodeEncodeError this function used to hit before encoding="utf-8"
    was added, PROJECT_STATE.md Sec 9) it is loaded and reused instead of
    spending another real API call redoing that repetition. This matters
    because each repetition is real, billed API usage -- a script crash
    on repetition 4 must not force re-paying for repetitions 1-3. This
    property is verified deterministically in tests/
    test_run_phase3_agentic.py with fakes, not left to be discovered under
    a real crash a second time.

    ledger_dir/trajectory/search_bounds/rate_tolerance/optimizer_agent/
    validator_agent all default to the real, production values (building
    a fresh 15M-step trajectory via build_reference_context() and real
    Agents via run_one_real_loop()'s own defaults) -- tests override them
    with a temp directory, a small trajectory, and FunctionModel fakes.
    """
    ledger_dir = ledger_dir or LEDGER_DIR
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if trajectory is None or search_bounds is None or rate_tolerance is None:
        trajectory, search_bounds, rate_tolerance = build_reference_context()

    runs = []
    for run_id in range(1, n_repetitions + 1):
        ledger_path = ledger_dir / f"run_{run_id:02d}_ledger.json"
        if ledger_path.exists() and ledger_path.stat().st_size > 0:
            run = AgenticRun.model_validate_json(ledger_path.read_text(encoding="utf-8"))
            print(f"=== run {run_id}/{n_repetitions} (already completed, loaded from {ledger_path}) ===")
        else:
            print(f"=== run {run_id}/{n_repetitions} ===")
            run = run_one_real_loop(trajectory, search_bounds, rate_tolerance,
                                     optimizer_agent=optimizer_agent, validator_agent=validator_agent)
            ledger_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            print(f"  {len(run.entries)} iterations, stop_reason={run.stop_reason} -- saved to {ledger_path}")
        runs.append(run)

    return runs, rate_tolerance


def summarize_convergence(runs):
    """Report the convergence rate honestly: a run that hit the iteration
    cap without approval is NOT a converged run, and this function never
    conflates the two (agents/orchestrator.py's stop_reason is what makes
    this distinction possible at all)."""
    converged = [run for run in runs if run.stop_reason == "validator_accepted"]
    exhausted = [run for run in runs if run.stop_reason == "iteration_cap_reached"]

    print(f"\n=== Convergence rate: {len(converged)}/{len(runs)} runs converged within the cap ===")
    for run in exhausted:
        print(f"  EXHAUSTED after {len(run.entries)} iterations without approval")

    return converged, exhausted


def check_paths_differ(runs):
    """
    Print every run's proposed-config sequence -- the direct evidence for
    "the search explores." No pass/fail assertion: there is no known
    answer for "how much should paths differ," this is a demonstrated,
    human-inspectable property, not a gate.
    """
    print("\n=== Search paths per run (demonstrates non-determinism) ===")
    for i, run in enumerate(runs, start=1):
        steps = [
            f"(n_clusters={entry.proposal.config.n_clusters}, "
            f"msm_lagtime={entry.proposal.config.msm_lagtime}, "
            f"seed={entry.proposal.config.cluster_seed})"
            for entry in run.entries
        ]
        print(f"  run {i}: {len(run.entries)} iterations -- " + " -> ".join(steps))


def check_accepted_rates_inside_band(converged_runs, rate_tolerance):
    """
    Report where each converged run's accepted rate lies within the
    Validator's band around the exact chain rate. Inside is guaranteed by
    the accept rule; an OUTSIDE line would mean the loop's bookkeeping broke.
    """
    analytical_rate = reference_rate(REFERENCE_BETA)
    lower = analytical_rate * (1.0 - rate_tolerance)
    upper = analytical_rate * (1.0 + rate_tolerance)

    print(f"\n=== Accepted rates vs. the Validator's band "
          f"[{lower:.6g}, {upper:.6g}] (exact chain rate={analytical_rate:.6g}) ===")
    accepted_rates = []
    all_inside = True
    for i, run in enumerate(converged_runs, start=1):
        accepted_rate = run.entries[-1].result.relaxation_rate_mean
        inside = lower < accepted_rate < upper
        all_inside = all_inside and inside
        accepted_rates.append(accepted_rate)
        print(f"  run {i}: accepted_rate={accepted_rate:.6g} -- {'INSIDE' if inside else 'OUTSIDE'} the band")

    return accepted_rates, lower, upper, analytical_rate, all_inside


def make_comparison_plot(accepted_rates, lower, upper, analytical_rate, out_path):
    """Plot each converged run's accepted rate against the Validator's band."""
    fig, ax = plt.subplots(figsize=(7, 5))
    run_indices = np.arange(1, len(accepted_rates) + 1)
    ax.axhspan(lower, upper, color="tab:blue", alpha=0.15, label="Validator tolerance band")
    ax.axhline(analytical_rate, color="tab:red", linestyle="--", label="exact chain rate")
    ax.scatter(run_indices, accepted_rates, color="tab:blue", zorder=3, label="accepted rate per run")
    ax.set_xlabel("converged run index")
    ax.set_ylabel("accepted relaxation rate (1/time)")
    ax.set_title("Phase 3: accepted rates per run")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved comparison plot to {out_path}")


def main():
    runs, rate_tolerance = run_all_repetitions()
    converged, exhausted = summarize_convergence(runs)
    check_paths_differ(runs)

    if not converged:
        print("\nNo runs converged -- cannot check accepted-rate consistency. This is itself "
              "a real finding about the loop's reliability at these settings, not hidden.")
        return

    accepted_rates, lower, upper, analytical_rate, all_inside = check_accepted_rates_inside_band(
        converged, rate_tolerance
    )
    make_comparison_plot(accepted_rates, lower, upper, analytical_rate,
                          "results/phase3_convergence_study.png")

    print(f"\n{'All' if all_inside else 'NOT all'} converged runs' accepted rates fall inside "
          f"the Validator's band.")
    print("Phase 3 convergence-robustness study complete.")


if __name__ == "__main__":
    main()
