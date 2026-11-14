import numpy as np


def generate_centroid_grasp(point_cloud):

    points = np.asarray(
        point_cloud,
        dtype=float
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
    ):
        raise ValueError(
            "Point cloud must have shape (N,3)"
        )

    center = points.mean(axis=0)

    grasp = {
        "position": center,
        "orientation": np.eye(3),
        "method": "centroid_baseline_not_GraspGen"
    }

    return grasp