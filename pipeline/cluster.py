"""
K-means clustering of the feature trajectory into microstates, using
deeptime.clustering.KMeans.

We deliberately use MANY microstates (n_clusters=50, PROJECT_STATE.md
Sec 4), not just 2. Clustering directly into 2 microstates would silently
hand-build the two-macrostate answer ourselves instead of letting the MSM
recover it from the transition dynamics between many finer microstates --
exactly the "accidentally hand-built a 2-state model" failure mode that
would make the downstream implied-timescale plot trivially true and
uninformative. With 50 microstates spanning the state space, a single
barrier-crossing event visibly passes through several distinct
microstates on its way from one well to the other (see the visual check
run alongside this module, described in PROJECT_STATE.md).
"""

import numpy as np
from deeptime.clustering import KMeans


def cluster_trajectory(features, n_clusters=50, seed=42, max_fit_frames=50_000):
    """
    Assign each frame of a feature trajectory to one of n_clusters
    microstates via k-means.

    For long trajectories, k-means FITS (finds centroids for) a random
    subsample of at most max_fit_frames frames, then TRANSFORMS (assigns
    a microstate to) every frame of the full trajectory. Fitting k-means
    on the full data scales badly (fitting on 1.5M points took ~74s
    against this project's 1-D data vs ~1s for a 50k-frame subsample,
    with visually identical resulting centroids -- see PROJECT_STATE.md).
    This does NOT lose statistics where it matters: the MSM is built from
    the discrete labels of the FULL trajectory, only the centroid-finding
    step is subsampled.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_frames, n_features), e.g. from
        pipeline.features.compute_features().
    n_clusters : int, optional
        Number of k-means microstates. Default 50 (PROJECT_STATE.md Sec 4).
    seed : int, optional
        Random seed for k-means initialization and for choosing the
        fitting subsample, so clustering is reproducible. Default 42
        (PROJECT_STATE.md Sec 4).
    max_fit_frames : int, optional
        Largest number of frames used to FIT the k-means centroids.
        Default 50,000. Trajectories shorter than this are used in full
        (no subsampling occurs).

    Returns
    -------
    discrete_trajectory : np.ndarray
        Array of shape (n_frames,), dtype int32: the microstate index
        assigned to each frame of the FULL trajectory.
    cluster_centers : np.ndarray
        Array of shape (n_clusters, n_features): the position of each
        microstate's centroid, sorted by k-means (not by position).
    """
    n_frames = features.shape[0]
    if n_frames > max_fit_frames:
        rng = np.random.default_rng(seed)
        fit_frame_indices = rng.choice(n_frames, size=max_fit_frames, replace=False)
        fitting_data = features[fit_frame_indices]
    else:
        fitting_data = features

    kmeans_estimator = KMeans(n_clusters=n_clusters, fixed_seed=seed)
    kmeans_model = kmeans_estimator.fit(fitting_data).fetch_model()

    discrete_trajectory = kmeans_model.transform(features)
    cluster_centers = kmeans_model.cluster_centers

    return discrete_trajectory, cluster_centers
