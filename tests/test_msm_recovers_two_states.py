"""
Fast, small-scale regression test for scripts/run_phase1_benchmark.py's
logic. This is NOT the full centerpiece sweep (that lives in the script
itself, run directly to produce results/arrhenius.png with 6 replicas x
15M steps x 8 beta values, taking ~15-20 minutes) -- it reuses the exact
same functions with much smaller settings, just to catch a broken
pipeline quickly in the normal fast test suite. A slope within a
generous tolerance here is a smoke test, not a substitute for actually
running the script and inspecting results/arrhenius.png.
"""

import numpy as np

import scripts.run_phase1_benchmark as run_phase1_benchmark


def test_preflight_factor_of_two_check_passes():
    """
    The MSM relaxation rate must be close to 2x the raw crossing rate at
    a modest scale too, not just in the full-scale run recorded in
    PROJECT_STATE.md.
    """
    ratio = run_phase1_benchmark.preflight_check_factor_of_two(
        beta=5.0, n_steps=500_000, seed=1,
    )

    assert 1.5 < ratio < 2.5


def test_small_scale_sweep_recovers_two_states_and_correct_slope():
    """
    A fast, small-scale version of the full sweep: fewer steps, fewer
    replicas, fewer beta values. Checks the same two gates as the real
    script (two-state recovery at every beta; slope close to -A), with a
    looser slope tolerance appropriate for the much smaller sample sizes
    used here.
    """
    original_n_steps = run_phase1_benchmark.N_STEPS
    original_n_replicas = run_phase1_benchmark.N_REPLICAS
    original_beta_values = run_phase1_benchmark.BETA_VALUES
    try:
        run_phase1_benchmark.N_STEPS = 400_000
        run_phase1_benchmark.N_REPLICAS = 3
        run_phase1_benchmark.BETA_VALUES = np.array([3.0, 5.0, 7.0])

        sweep = run_phase1_benchmark.run_beta_sweep()

        assert np.all(sweep["all_two_state_ok"]), (
            "Two-macrostate recovery failed at one or more beta values in the "
            "small-scale smoke sweep -- this indicates a real pipeline problem, "
            "not just small-sample noise (macrostate COUNT should be robust "
            "even with few replicas)."
        )

        slope, slope_stderr = run_phase1_benchmark.fit_arrhenius_slope(
            run_phase1_benchmark.BETA_VALUES, sweep["mean_rate"], sweep["sem_rate"],
        )
        analytical_slope = -run_phase1_benchmark.BARRIER_HEIGHT
        # Relative tolerance, not an N-sigma statistical test -- see the
        # SLOPE TOLERANCE note in scripts/run_phase1_benchmark.py's main()
        # for why: Eyring-Kramers is a beta->infinity asymptotic formula, so
        # a real O(1/beta) correction is expected at any finite beta and an
        # N-sigma test can fail on good data once statistics are tight
        # enough (this bit the full-scale sweep once already, see
        # PROJECT_STATE.md). 20% here (vs. the real script's 10%) accounts
        # for this test's much smaller, noisier sample (3 replicas, 400k
        # steps vs. 6 replicas, 15M steps).
        slope_relative_error = abs(slope - analytical_slope) / run_phase1_benchmark.BARRIER_HEIGHT

        assert slope_relative_error < 0.20, (
            f"measured slope {slope:.4f} +/- {slope_stderr:.4f} is "
            f"{slope_relative_error * 100:.2f}% from the analytical {analytical_slope}"
        )
    finally:
        run_phase1_benchmark.N_STEPS = original_n_steps
        run_phase1_benchmark.N_REPLICAS = original_n_replicas
        run_phase1_benchmark.BETA_VALUES = original_beta_values
