"""
Known-answer tests for pipeline/features.py.
"""

import numpy as np

from pipeline.features import compute_features


def test_features_have_expected_shape():
    """A 1-D trajectory of length n must become an (n, 1) feature array."""
    trajectory = np.array([1.0, 0.5, -0.3, -1.0, 0.2])

    features = compute_features(trajectory)

    assert features.shape == (5, 1)


def test_features_preserve_values_unchanged():
    """compute_features must not alter the underlying values, only reshape."""
    trajectory = np.array([1.0, 0.5, -0.3, -1.0, 0.2])

    features = compute_features(trajectory)

    assert np.array_equal(features.flatten(), trajectory)
