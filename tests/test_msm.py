"""
Known-answer tests for pipeline/msm.py.

Uses a real 0-D trajectory (physics.simulate_0d) long enough to show
several barrier crossings at the project's baseline beta=5.0, so these
checks are grounded in actual bistable dynamics, not synthetic data.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d
from physics.known_answers import expected_number_of_states, boltzmann_population_ratio
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory
from pipeline.msm import (
    build_msm, implied_timescales, recover_two_macrostates, find_converged_lagtime,
)

# Shared trajectory for all tests in this file. beta=5.0, dt=0.01,
# 1.5M steps -> total time 15000, giving ~85 COMMITTED crossings (checked
# with a +-0.5 hysteresis band, not a naive x=0 sign change, which
# overcounts barrier-grazing noise as "crossings"). This matters: an
# earlier, shorter 250k-step version of this trajectory had only 12
# committed crossings and showed a genuine (not a bug -- confirmed against
# the raw trajectory itself) 20/80 population split purely from finite-
# sample noise, since a handful of exponential-ish dwell times can easily
# sum unevenly. ~85 crossings brings the two-well population estimate
# within a few percent of the true 50/50 (see test below for the reasoning
# behind its tolerance).
_TRAJECTORY = run_trajectory_0d(n_steps=1_500_000, seed=7, beta=5.0, dt=0.01)
_FEATURES = compute_features(_TRAJECTORY)
_DISCRETE_TRAJECTORY, _ = cluster_trajectory(_FEATURES, n_clusters=50, seed=42)


def test_build_msm_returns_valid_model():
    """The fitted MSM's state count must not exceed the number of microstates."""
    msm = build_msm(_DISCRETE_TRAJECTORY, lagtime=10)

    assert msm.n_states <= 50
    assert msm.n_states > 1


def test_implied_timescales_are_positive_and_correct_length():
    """implied_timescales() must return one positive value per lag time."""
    lagtimes = [1, 5, 10, 20]

    timescales = implied_timescales(_DISCRETE_TRAJECTORY, lagtimes)

    assert timescales.shape == (len(lagtimes),)
    assert np.all(timescales > 0)


def test_implied_timescale_plateaus_at_larger_lag_times():
    """
    The slowest implied timescale should stabilize (stop changing much)
    once the lag time is long enough for the microstate dynamics to be
    Markovian -- the classic ITS plateau. Tolerance is TIGHT (3%, not a
    loose round number): a loose tolerance can call a curve that is
    still visibly climbing "converged" (see find_converged_lagtime's
    docstring and PROJECT_STATE.md Sec 10 for a case -- beta=7 at lag 20
    -- where a 25% tolerance would have wrongly accepted a +12.84% step).
    At this trajectory's beta=5, lag=20->40 is independently known (same
    PROJECT_STATE.md section) to change by +1.92%, comfortably under 3%.
    """
    lag_20_timescale = implied_timescales(_DISCRETE_TRAJECTORY, [20])[0]
    lag_40_timescale = implied_timescales(_DISCRETE_TRAJECTORY, [40])[0]

    relative_change = abs(lag_40_timescale - lag_20_timescale) / lag_20_timescale
    assert relative_change < 0.03


def test_find_converged_lagtime_matches_known_plateau_shape():
    """
    At beta=5 with a 3% tolerance, the ITS plateau is independently known
    (PROJECT_STATE.md Sec 10's convergence scan) to first satisfy the
    criterion at lag=20 (the 20->40 step is +1.92%, under 3%; the 10->20
    step is +5.33%, over). find_converged_lagtime() should reproduce this
    exactly on this trajectory, not just return "some" plausible lag.
    """
    candidate_lags = [10, 20, 40, 80]

    converged_lag, timescales = find_converged_lagtime(
        _DISCRETE_TRAJECTORY, candidate_lags, plateau_tolerance=0.03,
    )

    assert converged_lag == 20
    assert len(timescales) == len(candidate_lags)


def test_find_converged_lagtime_returns_none_when_data_is_insufficient():
    """
    A short, sparsely-sampled trajectory at a high beta (few crossings)
    should NOT show a genuine plateau within a modest candidate lag
    range -- find_converged_lagtime must honestly report this (None),
    not silently pick the largest lag and pretend it converged.
    """
    short_trajectory = run_trajectory_0d(n_steps=200_000, seed=3, beta=9.0, dt=0.01)
    features = compute_features(short_trajectory)
    discrete_trajectory, _ = cluster_trajectory(features, n_clusters=50, seed=42)

    converged_lag, _ = find_converged_lagtime(
        discrete_trajectory, [10, 20, 40, 80], plateau_tolerance=0.03,
    )

    assert converged_lag is None


def test_recovers_exactly_two_macrostates():
    """
    PCCA+ coarse-graining must partition the 50 microstates into exactly
    2 macrostates, matching physics.known_answers.expected_number_of_states().
    """
    _, pcca_model = recover_two_macrostates(_DISCRETE_TRAJECTORY, lagtime=20)

    unique_macrostates = np.unique(pcca_model.assignments)

    assert len(unique_macrostates) == expected_number_of_states()


def test_macrostate_populations_match_symmetric_boltzmann_ratio():
    """
    For the symmetric (b=0) potential used to generate this trajectory,
    physics.known_answers.boltzmann_population_ratio() predicts exactly
    equal populations (ratio 1). The MSM's recovered coarse-grained
    stationary populations should be close to 50/50, within the
    statistical noise of a finite trajectory: with ~85 committed
    crossings (~42 dwell periods per well), the relative fluctuation in
    cumulative dwell time is roughly 1/sqrt(42) =~ 15%, so a 0.15
    absolute tolerance on the population difference is generous but
    principled, not an arbitrary round number.
    """
    _, pcca_model = recover_two_macrostates(_DISCRETE_TRAJECTORY, lagtime=20)
    populations = pcca_model.coarse_grained_stationary_probability

    predicted_ratio = boltzmann_population_ratio(beta=5.0, A=1.0, b=0.0)

    assert abs(predicted_ratio - 1.0) < 1e-8  # sanity on the known answer itself
    assert abs(populations.sum() - 1.0) < 1e-6
    assert abs(populations[0] - populations[1]) < 0.15
