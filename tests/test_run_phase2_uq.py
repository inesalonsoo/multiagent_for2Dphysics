"""
Known-answer tests for scripts/run_phase2_uq.py's TOTAL (statistical +
systematic) error-budget gate.

This is where the physically-correct containment claim -- "the analytical
relaxation rate falls inside a properly combined statistical + systematic
band" -- actually gets asserted (see tests/test_uq.py's module docstring
for why that assertion does NOT live there: pipeline/uq.py only computes
the statistical component, and the systematic term is characterized from
Phase 1's ensemble, outside that module).

Two layers, for two different reasons:
1. test_total_error_band_is_centered_on_phase1_mean -- a fast, hermetic,
   synthetic-data test of build_total_error_band()'s arithmetic alone. It
   exists to guard the exact bug class this session just fixed: an
   earlier version centered the total band on THIS module's own single-
   trajectory rate_mean instead of Phase 1's ensemble mean, which are
   close but not equal, and that small mismatch alone caused the gate to
   fail at beta=4 and beta=7. No trajectory simulation, no disk I/O --
   just the combination math.
2. test_total_band_contains_analytical_rate_for_real_phase1_and_phase2_data
   -- an integration test against the REAL, already-computed outputs of
   scripts/run_phase1_benchmark.py and scripts/run_phase2_uq.py (cached in
   results/*.npz). This is the actual physical claim under test: does the
   real total band, built from real measured statistical and systematic
   components, really contain the real analytical rate at every
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
    check_analytical_value_inside_interval,
    load_phase1_reference,
)
from scripts.run_phase1_benchmark import BETA_VALUES, FIT_BETA_MAX


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


@pytest.mark.skipif(
    not (os.path.exists("results/arrhenius_sweep_raw.npz")
         and os.path.exists("results/uq_sweep_raw.npz")),
    reason="requires cached results/arrhenius_sweep_raw.npz and "
           "results/uq_sweep_raw.npz -- run scripts.run_phase1_benchmark "
           "then scripts.run_phase2_uq to generate them",
)
def test_total_band_contains_analytical_rate_for_real_phase1_and_phase2_data():
    """
    The physically-correct known-answer check, at the level where all its
    inputs actually live: using Phase 1's REAL measured systematic bias
    and Phase 2's REAL measured statistical credible intervals, does the
    combined total band contain the analytical relaxation rate at every
    beta <= FIT_BETA_MAX (Phase 1's own trustworthy-fit-range cutoff)?
    """
    uq_data = np.load("results/uq_sweep_raw.npz")
    rate_mean = uq_data["rate_mean"]
    rate_lower = uq_data["rate_lower"]
    rate_upper = uq_data["rate_upper"]
    assert np.array_equal(uq_data["beta_values"], BETA_VALUES), (
        "cached uq_sweep_raw.npz was computed for a different BETA_VALUES -- "
        "re-run scripts.run_phase2_uq"
    )

    phase1_mean_rate, systematic_relative = load_phase1_reference()
    total_lower, total_upper, _ = build_total_error_band(
        phase1_mean_rate, rate_mean, rate_lower, rate_upper, systematic_relative,
    )

    contained = check_analytical_value_inside_interval(total_lower, total_upper)
    in_fit = BETA_VALUES <= FIT_BETA_MAX

    assert np.all(contained[in_fit]), (
        f"analytical rate fell outside the total (statistical+systematic) "
        f"band for at least one beta <= {FIT_BETA_MAX}: "
        f"{dict(zip(BETA_VALUES[in_fit], contained[in_fit]))}"
    )
