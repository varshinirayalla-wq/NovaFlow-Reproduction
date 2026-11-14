import numpy as np
from scipy.ndimage import gaussian_filter1d


def optimize_trajectory(targets, sigma=1.0):

    targets = np.asarray(
        targets,
        dtype=float
    )

    if (
        targets.ndim != 2
        or targets.shape[1] != 3
        or len(targets) < 2
    ):
        raise ValueError(
            "Targets must have shape (T,3)"
        )

    optimized = gaussian_filter1d(
        targets,
        sigma=sigma,
        axis=0,
        mode="nearest"
    )

    return optimized