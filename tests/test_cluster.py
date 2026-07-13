"""
Known-answer tests for pipeline/cluster.py.
"""

import numpy as np

from physics.simulate_0d import run_trajectory_0d
from pipeline.features import compute_features
from pipeline.cluster import cluster_trajectory


def _sample_features(n_steps=2000, seed=0):
    trajectory = run_trajectory_0d(n_steps=n_steps, seed=seed)
    return compute_features(trajectory)


def test_discrete_trajectory_has_expected_shape_and_range():
    """
    discrete_trajectory must have one microstate label per frame, and
    every label must be a valid microstate index in [0, n_clusters).
    """
    features = _sample_features()
    n_clusters = 20

    discrete_trajectory, cluster_centers = cluster_trajectory(
        features, n_clusters=n_clusters, seed=42,
    )

    assert discrete_trajectory.shape == (features.shape[0],)
    assert discrete_trajectory.min() >= 0
    assert discrete_trajectory.max() < n_clusters
    assert cluster_centers.shape == (n_clusters, 1)


def test_same_seed_gives_identical_clustering():
    """Reusing the same seed must give identical microstate assignments."""
    features = _sample_features()

    first_run, _ = cluster_trajectory(features, n_clusters=20, seed=42)
    second_run, _ = cluster_trajectory(features, n_clusters=20, seed=42)

    assert np.array_equal(first_run, second_run)


def test_cluster_centers_span_the_data_range():
    """
    With enough microstates, the k-means centers should span roughly the
    same range as the underlying data -- not collapse to a narrow band
    (a sanity check that clustering is actually resolving structure,
    not just noise).
    """
    features = _sample_features(n_steps=5000, seed=1)

    _, cluster_centers = cluster_trajectory(features, n_clusters=50, seed=42)

    data_range = features.max() - features.min()
    centers_range = cluster_centers.max() - cluster_centers.min()

    assert centers_range > 0.8 * data_range
