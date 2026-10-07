"""
Turn a 0-D trajectory into the array shape the MSM tools expect.

In 0-D the particle's position x(t) is already the natural coordinate for
the hop between valleys, so no feature choice is needed: this is just a
reshape. (The 2D field will need a summary number per frame, such as the
spatial average of phi.)
"""

import numpy as np


def compute_features(trajectory):
    """
    Reshape a 1-D trajectory x(t) into the (n_frames, n_features) array
    shape that deeptime's clustering/MSM stages expect, with
    n_features=1 (the position itself).

    Parameters
    ----------
    trajectory : np.ndarray
        1-D array of shape (n_frames,), e.g. from
        physics.simulate_0d.run_trajectory_0d().

    Returns
    -------
    np.ndarray
        Array of shape (n_frames, 1): the same trajectory values,
        reshaped, with no other transformation.
    """
    features = trajectory.reshape(-1, 1)
    return features
