import json
import numpy as np

from config import (
    create_directories,
    RESULTS_DIR
)

from simulation.oracle import (
    synthetic_rigid_motion
)

from models.pose_estimator import (
    kabsch_ransac,
    rotation_error_degrees
)

from models.depth_calibration import (
    calibrate_depth
)

from evaluation.metrics import (
    translation_error,
    wilson_interval
)


def main():

    create_directories()

    # Generate synthetic 3D points

    source, target, r_true, t_true = (
        synthetic_rigid_motion()
    )

    # Add outliers

    target_corrupt = target.copy()

    target_corrupt[:3] += 0.15

    # Estimate pose

    r_est, t_est, inliers = kabsch_ransac(

        source,

        target_corrupt,

        iterations=200,

        threshold=0.01,

        seed=42

    )

    # Calculate errors

    rotation_error = rotation_error_degrees(
        r_est,
        r_true
    )

    translation_err = translation_error(
        t_est,
        t_true
    )

    # Test depth calibration

    sensor_depth = np.full(
        (32, 32),
        0.8,
        dtype=np.float32
    )

    predicted_depths = [

        np.full(
            (32, 32),
            2.0,
            dtype=np.float32
        )

        for _ in range(4)

    ]

    calibrated, scale = calibrate_depth(
        sensor_depth,
        predicted_depths
    )

    # Prepare results

    result = {

        "mode": "synthetic_math_smoke_test",

        "rotation_error_degrees": rotation_error,

        "translation_error_m": translation_err,

        "ransac_inliers": int(inliers.sum()),

        "ransac_total_points": len(inliers),

        "depth_scale": scale,

        "calibrated_depth_median": float(
            np.median(calibrated[0])
        ),

        "example_wilson_interval": wilson_interval(
            80,
            100
        ),

        "warning": (
            "This is not a full NovaFlow reproduction."
        )

    }

    output = (
        RESULTS_DIR / "smoke_test_results.json"
    )

    output.write_text(
        json.dumps(result, indent=4)
    )

    print(json.dumps(result, indent=4))

    print("\nResults saved successfully!")


if __name__ == "__main__":

    main()