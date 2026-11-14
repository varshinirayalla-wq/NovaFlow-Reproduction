import numpy as np


def execute_cartesian_demo(trajectory):

    trajectory = np.asarray(
        trajectory,
        dtype=float
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 3
    ):
        raise ValueError(
            "Trajectory must have shape (T,3)"
        )

    result = {
        "executed": True,
        "num_waypoints": len(trajectory),
        "start": trajectory[0].tolist(),
        "end": trajectory[-1].tolist(),
        "simulation_type": "kinematic_placeholder"
    }

    return result