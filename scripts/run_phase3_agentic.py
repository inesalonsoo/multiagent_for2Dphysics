"""
Phase 3 convergence-robustness study (PROJECT_STATE.md Sec 9/10, the
PI-flagged next step once the three-agent loop existed). Runs the REAL
agentic loop (agents/loop.py, agents/orchestrator.py) N_REPETITIONS times
on ONE fixed reference trajectory at REFERENCE_BETA (agents/validator.py
-- Phase 1/2's clean beta<=7 range, where the rate gate has real teeth.
A high-beta demo would prove nothing: Phase 2 already showed the total
error band widens with sparse transition counts until almost anything
passes -- see PROJECT_STATE.md Sec 9).

WHAT THIS STUDY PROVES -- two things in tension, on purpose:
1. The search is genuinely non-deterministic: repeated runs' ledgers must
   visibly differ -- different configs proposed, different iteration
   counts, different reasoning. If they came back identical, either the
   LLM isn't actually varying or the logging is too coarse to see it --
   either way that would be a finding, not a footnote, and is reported as
   such by check_paths_differ() below.
2. Despite divergent paths, every CONVERGED run's accepted config's
   measured rate falls inside Phase 2's own total (statistical (+)
   systematic) error band around the analytical rate -- the outcome is
   bounded even though the path isn't. This is the empirical,
   DEMONSTRATED (not asserted) proof that agents/validator.py's gate does
   real constraining work: identical outcomes every time would mean the
   gate isn't being tested by real variation; outcomes scattered beyond
   the UQ band would mean the gate is too permissive or the UQ too tight.

MAKES REAL API CALLS -- requires ANTHROPIC_API_KEY. Never run from the
test suite: tests/test_orchestrator.py and tests/test_loop.py already
cover all the deterministic routing/ledger-faithfulness/wiring machinery
this script exercises, with fakes. Only the agents' reasoning quality is
demonstrated here, not unit-tested -- consistent with agents/optimizer.py
and agents/validator.py's own documented demonstrated-not-proven stance.

HONESTY, IN ADVANCE: this script reports whatever it actually observes.
A convergence rate below N_REPETITIONS/N_REPETITIONS is a real,
reportable property of the loop's reliability at these settings, not a
bug to hide (summarize_convergence() below never conflates an exhausted
run with a converged one -- see agents/orchestrator.py's stop_reason).
A converged run whose accepted rate falls OUTSIDE the UQ band is a real
finding about the gate or the UQ (check_converged_rates_inside_uq_band()
reports it plainly, does not filter it out). Go in willing to find either.

OUTPUT: every run's ledger persisted to
results/phase3_convergence_study/run_NN_ledger.json (before any analysis
that could fail and lose the data -- the lesson Phase 1 already paid for,
PROJECT_STATE.md Sec 9), plus results/phase3_convergence_study.png -- the
agentic-layer analogue of results/arrhenius.png: bounded outcome despite
varied path, shown visually.

[2026-07-12 REDESIGN + HONEST SCOPING, PROJECT_STATE.md Sec 9.] The first
version of this study (N_REPETITIONS=8) ran under a SearchBounds that
handed the Optimizer the converged msm_lagtime directly -- every one of 8
real runs proposed the byte-identical config on iteration 1, so the
Validator's gate was never exercised against a genuinely wrong config.
Fixed in agents/optimizer.py: SearchBounds now states only the physical
reasoning that bounds a sensible lag, not the solved value, so a real
search actually has to happen. N_REPETITIONS dropped from 8 to 4:
budget is real, and what the claim needs is not statistical weight but
four QUALITATIVE properties -- (a) proposals genuinely diverge across
runs, (b) at least one run hits a config the Validator rejects on real
physics grounds, (c) the Optimizer reacts to that rejection and moves,
(d) every accepted config lands inside the UQ band. A single real dry
run under the redesigned prompt already demonstrated (b), (c), and (d)
in one pass (6 iterations: 5 genuine physics rejections, each with the
Optimizer visibly reasoning over its own accumulating history, converging
on a DIFFERENT config than the old anchored answer, measured rate inside
the UQ band) -- that run is reused here as run_01 (its trajectory is
byte-identical to what build_reference_context(seed=7) below produces,
so nothing about reusing it is inconsistent with a fresh run). The
remaining repetitions exist mainly to confirm (a): that independent runs
genuinely propose different search paths, not just that one run can
search internally.

RUN THIS WITH `-m`: `python -m scripts.run_phase3_agentic` from the
project root (same reason as run_phase1_benchmark.py -- see its
docstring).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.loop import build_reference_context, run_one_real_loop
from agents.schemas import AgenticRun
from agents.validator import REFERENCE_BETA
from physics.known_answers import eyring_kramers_rate_0d

N_REPETITIONS = 4  # [2026-07-12] revised down from 8 -- see module docstring's "HONEST SCOPING"
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


def check_converged_rates_inside_uq_band(converged_runs, rate_tolerance):
    """
    The other half of the claim: every converged run's accepted config's
    measured rate must fall inside Phase 2's own total error band around
    the analytical rate. Reports each one; a run OUTSIDE the band is
    printed as such, not hidden -- that would be a real finding about the
    gate or the UQ, not something to paper over here.
    """
    analytical_rate = 2.0 * eyring_kramers_rate_0d(beta=REFERENCE_BETA)
    lower = analytical_rate * (1.0 - rate_tolerance)
    upper = analytical_rate * (1.0 + rate_tolerance)

    print(f"\n=== Accepted rates vs. Phase 2 total error band "
          f"[{lower:.6g}, {upper:.6g}] (analytical={analytical_rate:.6g}) ===")
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
    """The agentic-layer analogue of results/arrhenius.png: bounded
    outcome despite varied path, shown visually -- the stronger Koppens
    artifact, since a single clean run could be luck."""
    fig, ax = plt.subplots(figsize=(7, 5))
    run_indices = np.arange(1, len(accepted_rates) + 1)
    ax.axhspan(lower, upper, color="tab:blue", alpha=0.15, label="Phase 2 total error band")
    ax.axhline(analytical_rate, color="tab:red", linestyle="--", label="analytical rate")
    ax.scatter(run_indices, accepted_rates, color="tab:blue", zorder=3, label="accepted rate per run")
    ax.set_xlabel("converged run index")
    ax.set_ylabel("accepted relaxation rate (1/time)")
    ax.set_title("Phase 3: accepted physics agrees despite divergent search paths")
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

    accepted_rates, lower, upper, analytical_rate, all_inside = check_converged_rates_inside_uq_band(
        converged, rate_tolerance
    )
    make_comparison_plot(accepted_rates, lower, upper, analytical_rate,
                          "results/phase3_convergence_study.png")

    print(f"\n{'All' if all_inside else 'NOT all'} converged runs' accepted rates fall inside "
          f"the Phase 2 total error band.")
    print("Phase 3 convergence-robustness study complete.")


if __name__ == "__main__":
    main()
