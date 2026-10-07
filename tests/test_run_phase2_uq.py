"""
Known-answer tests for scripts/run_phase2_uq.py, on synthetic replica data
where coverage and spreads are known by construction.
"""

import numpy as np
from scipy import stats

from scripts.run_phase1_benchmark import CONFIDENCE
from scripts.run_phase2_uq import summarize_interval_honesty

TRUE_RATE = 1.0
Z_SCORE = stats.norm.ppf(0.5 + CONFIDENCE / 2.0)
RATES = TRUE_RATE * np.array([0.98, 1.02, 0.99, 1.01, 1.00, 1.00])
REPLICA_SIGMA = RATES.std(ddof=1) / RATES.mean()  # about 1.4%


def test_honest_intervals_cover_and_match_spread():
    """
    Intervals whose width matches the replica scatter: every interval here
    contains the true rate, and the two sigmas agree.
    """
    half_width = Z_SCORE * REPLICA_SIGMA * TRUE_RATE

    n_covering, n_resolved, bayesian_sigma, replica_sigma = summarize_interval_honesty(
        RATES, RATES - half_width, RATES + half_width, TRUE_RATE,
    )

    assert (n_covering, n_resolved) == (6, 6)
    assert abs(replica_sigma / bayesian_sigma - 1.0) < 1e-9


def test_too_narrow_intervals_are_detected():
    """
    Intervals 4 times narrower than the scatter: only the two replicas
    sitting on the true rate are covered, and the replica sigma is 4 times
    the Bayesian one.
    """
    half_width = Z_SCORE * REPLICA_SIGMA / 4.0 * TRUE_RATE

    n_covering, _, bayesian_sigma, replica_sigma = summarize_interval_honesty(
        RATES, RATES - half_width, RATES + half_width, TRUE_RATE,
    )

    assert n_covering == 2
    assert abs(replica_sigma / bayesian_sigma - 4.0) < 1e-9


def test_unresolved_replicas_are_ignored():
    """NaN replicas (unresolved) must not count toward coverage or spread."""
    rates = np.array([0.99, 1.01, np.nan, 1.00])
    half_width = Z_SCORE * 0.02

    _, n_resolved, _, _ = summarize_interval_honesty(
        rates, rates - half_width, rates + half_width, TRUE_RATE,
    )

    assert n_resolved == 3
