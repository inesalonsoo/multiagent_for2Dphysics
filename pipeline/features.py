"""
Turn a 0-D trajectory into an MSM-ready feature array.

In 0-D, the particle's position x(t) already IS the reaction coordinate:
there is no higher-dimensional field to project down, and no choice of
order parameter to argue about (the 2D field's Phase 4 pipeline has to
pick one -- the spatial-mean order parameter -- as a stand-in for a true
reaction coordinate it does not have direct access to). Here the state
variable and the reaction coordinate coincide exactly, so "feature
engineering" is just a reshape. This is exactly why the 0-D benchmark is
clean: there is no feature-choice ambiguity to introduce error before the
MSM stage even begins.
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
