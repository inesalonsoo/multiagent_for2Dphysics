"""
Fast regression test for scripts/run_phase1_benchmark.py, on shorter
trajectories than the real sweep. Checks that one replica passes both
Phase 1 gates, that an unresolvable replica is reported as such, and that
the rate gate accepts and rejects the right synthetic means.
"""

import numpy as np

import scripts.run_phase1_benchmark as phase1
from physics.known_answers import euler_maruyama_relaxation_rate_0d


def _with_n_steps(n_steps, function, *args):
    """Run function(*args) with phase1.N_STEPS temporarily set to n_steps."""
    original_n_steps = phase1.N_STEPS
    phase1.N_STEPS = n_steps
    try:
        return function(*args)
    finally:
        phase1.N_STEPS = original_n_steps


def test_resolved_replica_passes_both_gates():
    """
    At beta=3, 1.5M steps see about 600 barrier crossings (about 4%
    statistical error): the seeded replica must show two metastable states
    and a rate within 6% of the exact chain rate.
    """
    replica = _with_n_steps(1_500_000, phase1.analyze_replica, 3.0, 11)
    chain_rate = euler_maruyama_relaxation_rate_0d(beta=3.0, dt=phase1.DT)

    assert replica["separation"] > phase1.MIN_TIMESCALE_SEPARATION
    assert replica["ck_error"] < phase1.MAX_CK_ERROR
    assert replica["ci_lower"] < replica["rate"] < replica["ci_upper"]
    assert abs(replica["rate"] / chain_rate - 1.0) < 0.06


def test_unresolvable_replica_is_reported_not_measured():
    """At beta=9, 200k steps cannot resolve t_2 (about 480k frames)."""
    replica = _with_n_steps(200_000, phase1.analyze_replica, 9.0, 3)

    assert all(np.isnan(value) for value in replica.values())


def test_rate_gate_accepts_agreement_and_rejects_bias():
    """
    Synthetic replica rates scattered 1% around the exact chain rate must
    pass; the same scatter shifted 5% high must fail; a single resolved
    replica is not gated.
    """
    beta = 5.0
    chain_rate = euler_maruyama_relaxation_rate_0d(beta=beta, dt=phase1.DT)
    scatter = np.array([-0.01, 0.005, 0.01, -0.005, 0.0, 0.002])

    unbiased = phase1.summarize_beta(chain_rate * (1.0 + scatter), beta)
    biased = phase1.summarize_beta(chain_rate * (1.05 + scatter), beta)
    single = phase1.summarize_beta(np.array([chain_rate, np.nan]), beta)

    assert unbiased["agrees"]
    assert not biased["agrees"]
    assert single["agrees"] is None
