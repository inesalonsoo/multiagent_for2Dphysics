"""
Known-answer tests for scripts/run_phase2_uq.py's TOTAL (statistical +
systematic) error-budget arithmetic, and its held-out replica-split gate.

**[2026-08-03] The second test below changed what it asserts -- see
PROJECT_STATE.md Sec 9 for the full derivation.**
report_error_budget_decomposition() (formerly named
check_analytical_value_inside_interval() and formerly a hard-gating
check) is NOT a physics test: once the total band is centered on
phase1_mean_rate and systematic_relative is defined relative to that
SAME phase1_mean_rate, containment of the analytical rate is an
algebraic identity, not a fact about the data -- it cannot fail. The
REAL, falsifiable claim -- does an honestly-built statistical+systematic
band, built from data that never saw the tested value, actually contain
that value -- now lives in
test_held_out_replica_mean_falls_inside_band_built_from_other_half,
using check_held_out_replica_mean_inside_band() against the odd/even
replica split (see run_phase1_benchmark.py::run_beta_sweep's "all_rates"
and scripts/run_phase2_uq.py's GATE note for why this split is
necessary for falsifiability where testing the analytical value is not).

Two layers, for two different reasons:
1. test_total_error_band_is_centered_on_phase1_mean -- a fast, hermetic,
   synthetic-data test of build_total_error_band()'s arithmetic alone. It
   exists to guard the exact bug class this session just fixed: an
   earlier version centered the total band on THIS module's own single-
   trajectory rate_mean instead of Phase 1's ensemble mean, which are
   close but not equal, and that small mismatch alone caused the gate to
   fail at beta=4 and beta=7. No trajectory simulation, no disk I/O --
   just the combination math.
2. test_held_out_replica_mean_falls_inside_band_built_from_other_half
   -- an integration test against the REAL, already-computed outputs of
   scripts/run_phase1_benchmark.py and scripts/run_phase2_uq.py (cached in
   results/*.npz). This is the actual physical claim under test: does a
   band built ENTIRELY from the odd-indexed replicas really contain the
   even-indexed replicas' mean -- data the band never saw -- at every
   beta <= FIT_BETA_MAX. Skipped (not failed) if those files are missing,
   since generating them from scratch takes on the order of tens of
   minutes (full Phase 1 sweep) -- see scripts/run_phase1_benchmark.py and
   scripts/run_phase2_uq.py docstrings to regenerate them.
"""

import os

import numpy as np
import pytest

from scripts.run_phase2_uq import (
    build_total_error_band,
    check_held_out_replica_mean_inside_band,
    load_phase1_reference,
    load_phase1_replica_split,
)
from physics.known_answers import eyring_kramers_rate_0d
from scripts.run_phase1_benchmark import BETA_VALUES, FIT_BETA_MAX, BARRIER_HEIGHT


def test_total_error_band_is_centered_on_phase1_mean():
    """
    Synthetic regression test for the centering bug: construct a case
    where this module's own rate_mean deliberately DIFFERS from
    phase1_mean_rate (mimicking single-trajectory sampling noise against
    a more precise ensemble mean), and confirm the returned total_lower/
    total_upper straddle phase1_mean_rate, not rate_mean.
    """
    phase1_mean_rate = np.array([1.0])
    rate_mean = np.array([1.05])       # deliberately offset from phase1_mean_rate
    rate_lower = np.array([1.03])
    rate_upper = np.array([1.07])
    systematic_relative = np.array([0.05])

    total_lower, total_upper, statistical_relative = build_total_error_band(
        phase1_mean_rate, rate_mean, rate_lower, rate_upper, systematic_relative,
    )

    band_center = (total_lower[0] + total_upper[0]) / 2.0
    assert band_center == pytest.approx(phase1_mean_rate[0]), (
        "total band must be centered on phase1_mean_rate, not this module's "
        "own rate_mean -- this is the exact bug that made the gate fail at "
        "beta=4 and beta=7 before the fix"
    )

    # The systematic term is defined as |predicted - phase1_mean| / phase1_mean
    # (see load_phase1_reference()'s docstring), so a band exactly as wide as
    # systematic_relative must reach the phase1_mean * (1 +/- systematic_relative)
    # points by construction, independent of where rate_mean happens to sit.
    assert total_upper[0] >= phase1_mean_rate[0] * (1.0 + systematic_relative[0])
    assert total_lower[0] <= phase1_mean_rate[0] * (1.0 - systematic_relative[0])


def _held_out_gate_inputs_available():
    """
    True only if BOTH cached .npz files exist AND arrhenius_sweep_raw.npz
    was written by a version of run_phase1_benchmark.py new enough to
    include the "all_rates" key (added this session -- an older cached
    file from before that change exists on disk but lacks it, which
    should skip cleanly, not raise a KeyError).
    """
    if not (os.path.exists("results/arrhenius_sweep_raw.npz")
            and os.path.exists("results/uq_sweep_raw.npz")):
        return False
    with np.load("results/arrhenius_sweep_raw.npz") as cached_data:
        return "all_rates" in cached_data.files


@pytest.mark.skipif(
    not _held_out_gate_inputs_available(),
    reason="requires cached results/arrhenius_sweep_raw.npz (with the "
           "all_rates key -- re-run scripts.run_phase1_benchmark if your "
           "cached copy predates this session) and results/uq_sweep_raw.npz",
)
def test_held_out_replica_mean_falls_inside_band_built_from_other_half():
    """
    The physically-correct, genuinely falsifiable known-answer check (see
    this file's module docstring and scripts/run_phase2_uq.py's GATE note
    for why testing the analytical value here is NOT this, anymore):
    using Phase 1's REAL per-replica rates, split into odd/even halves,
    does a total (statistical + systematic) band built ENTIRELY from the
    odd-indexed replicas contain the even-indexed replicas' mean -- real
    data that band never saw -- at every beta <= FIT_BETA_MAX?
    """
    uq_data = np.load("results/uq_sweep_raw.npz")
    rate_mean = uq_data["rate_mean"]
    rate_lower = uq_data["rate_lower"]
    rate_upper = uq_data["rate_upper"]
    assert np.array_equal(uq_data["beta_values"], BETA_VALUES), (
        "cached uq_sweep_raw.npz was computed for a different BETA_VALUES -- "
        "re-run scripts.run_phase2_uq"
    )

    odd_mean_rate, even_mean_rate = load_phase1_replica_split()
    predicted_rate = np.array(
        [2.0 * eyring_kramers_rate_0d(beta=b, A=BARRIER_HEIGHT) for b in BETA_VALUES]
    )
    systematic_relative_odd = np.abs(predicted_rate - odd_mean_rate) / odd_mean_rate
    held_out_lower, held_out_upper, _ = build_total_error_band(
        odd_mean_rate, rate_mean, rate_lower, rate_upper, systematic_relative_odd,
    )

    contained = check_held_out_replica_mean_inside_band(
        even_mean_rate, held_out_lower, held_out_upper,
    )
    in_fit = BETA_VALUES <= FIT_BETA_MAX

    assert np.all(contained[in_fit]), (
        f"even-replica mean fell outside a band built entirely from the "
        f"odd-replica data for at least one beta <= {FIT_BETA_MAX}: "
        f"{dict(zip(BETA_VALUES[in_fit], contained[in_fit]))}"
    )
