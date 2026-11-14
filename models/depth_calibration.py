import numpy as np


def calibrate_depth(sensor_depth, predicted_depths):

    sensor = np.asarray(
        sensor_depth,
        dtype=np.float32
    )

    preds = [
        np.asarray(d, dtype=np.float32)
        for d in predicted_depths
    ]

    if not preds:
        raise ValueError("predicted_depths is empty")

    sensor_valid = sensor[
        np.isfinite(sensor) & (sensor > 0)
    ]

    pred_valid = preds[0][
        np.isfinite(preds[0]) & (preds[0] > 0)
    ]

    if sensor_valid.size == 0 or pred_valid.size == 0:
        raise ValueError("No valid positive depth values")

    scale = (
        np.median(sensor_valid) /
        np.median(pred_valid)
    )

    calibrated = []

    for depth in preds:

        result = np.where(
            np.isfinite(depth) & (depth > 0),
            depth * scale,
            0
        )

        calibrated.append(
            result.astype(np.float32)
        )

    return calibrated, float(scale)