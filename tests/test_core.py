import numpy as np

from models.pose_estimator import (
    kabsch,
    transform_points
)

from models.depth_calibration import (
    calibrate_depth
)

from evaluation.metrics import (
    wilson_interval
)


def test_kabsch():

    rng = np.random.default_rng(3)

    points = rng.normal(
        size=(20, 3)
    )

    angle = 0.3

    rotation = np.array([

        [np.cos(angle), -np.sin(angle), 0],

        [np.sin(angle), np.cos(angle), 0],

        [0, 0, 1]

    ])

    translation = np.array([
        0.1,
        -0.2,
        0.05
    ])

    target = transform_points(
        points,
        rotation,
        translation
    )

    r, t = kabsch(
        points,
        target
    )

    assert np.allclose(
        r,
        rotation,
        atol=1e-6
    )

    assert np.allclose(
        t,
        translation,
        atol=1e-6
    )


def test_depth_calibration():

    sensor = np.full(
        (10, 10),
        0.6
    )

    predicted = [

        np.full((10, 10), 2.0),

        np.full((10, 10), 3.0)

    ]

    calibrated, scale = calibrate_depth(
        sensor,
        predicted
    )

    assert np.isclose(scale, 0.3)

    assert np.isclose(
        np.median(calibrated[0]),
        0.6
    )


def test_wilson_interval():

    low, high = wilson_interval(
        80,
        100
    )

    assert low < 0.8 < high