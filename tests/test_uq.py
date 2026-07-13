"""
Statistical-correctness tests for pipeline/uq.py.

Uses a real 0-D trajectory at the project's baseline beta=5.0 and checks
properties the Bayesian credible interval machinery itself must satisfy
(well-orderedness, shrinking with more data, widening with confidence).

Scope note: this module intentionally does NOT check the credible interval
against the analytical rate from physics/known_answers.py. A single-trajectory
statistical CI is not expected to always contain the true value once a real,
separately-characterized systematic bias exists (see the note on
test_credible_interval_scales_down_with_more_data below, and
PROJECT_STATE.md Sec 9). That accuracy check lives at the Phase 2 level
(tests/test_run_phase2_uq.py), where the systematic term is available.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.uq import compute_rate_credible_interval

DT = 0.01
LAGTIME = 20


def _sample_discrete_trajectory(n_steps=1_500_000, seed=7, beta=5.0):
    trajectory = run_trajectory_0d(n_steps=n_steps, seed=seed, beta=beta, dt=DT)
    features = compute_features(trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=50, seed=42)
    return discrete_trajectory


def test_credible_interval_is_well_ordered_and_contains_the_mean():
    """lower <= mean <= upper must hold for a sane credible interval."""
    discrete_trajectory = _sample_discrete_trajectory()

    rate_mean, rate_lower, rate_upper = compute_rate_credible_interval(
        discrete_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )

    assert rate_lower < rate_mean < rate_upper


def test_credible_interval_scales_down_with_more_data():
    """
    A statistical-correctness check for the CI machinery itself: doubling
    the trajectory length should shrink (or at worst not grow) the interval,
    since it is purely a sampling-uncertainty statement about a fixed model.

    NOTE on scope: this module's known-answer test used to also assert that
    the analytical relaxation rate (2x physics.known_answers.eyring_kramers_rate_0d,
    see RATE CONVENTION note in scripts/run_phase1_benchmark.py) falls inside
    the 90% credible interval. That assertion has been REMOVED, not loosened
    -- its premise was physically wrong, not its tolerance too tight.

    A single-trajectory Bayesian credible interval only captures STATISTICAL
    (sampling) uncertainty. Phase 1's 6-replica ensemble already measured a
    real, ~2-9% SYSTEMATIC bias (sparse-transition-count effects at high beta,
    plus a genuine asymptotic 1/beta correction to the Eyring-Kramers prefactor
    -- see PROJECT_STATE.md Sec 9) that a single-replica statistical CI does not,
    and should not be expected to, cover. Demanding pure-CI containment here
    was demanding the wrong thing of this module.

    That accuracy question -- does the analytical value fall inside a properly
    combined statistical (+) systematic error budget -- is now asserted at the
    Phase 2 level (tests/test_run_phase2_uq.py), where the systematic term
    (characterized from Phase 1, outside this module) actually lives. This
    module's job is only to compute a correctly-behaved statistical interval,
    which the tests below verify.
    """
    beta = 5.0
    short_trajectory = _sample_discrete_trajectory(n_steps=750_000, seed=7, beta=beta)
    long_trajectory = _sample_discrete_trajectory(n_steps=1_500_000, seed=7, beta=beta)

    _, short_lower, short_upper = compute_rate_credible_interval(
        short_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )
    _, long_lower, long_upper = compute_rate_credible_interval(
        long_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )

    assert (long_upper - long_lower) <= (short_upper - short_lower)


def test_wider_confidence_gives_wider_interval():
    """A 95% interval must be at least as wide as a 90% interval on the same data."""
    discrete_trajectory = _sample_discrete_trajectory()

    _, lower_90, upper_90 = compute_rate_credible_interval(
        discrete_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.90,
    )
    _, lower_95, upper_95 = compute_rate_credible_interval(
        discrete_trajectory, lagtime=LAGTIME, dt=DT, confidence=0.95,
    )

    assert (upper_95 - lower_95) >= (upper_90 - lower_90)
