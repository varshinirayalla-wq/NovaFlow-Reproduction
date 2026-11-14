import numpy as np


def synthetic_rigid_motion(
    num_points=30,
    seed=42
):

    rng = np.random.default_rng(seed)

    source = rng.normal(
        size=(num_points, 3)
    ) * 0.04

    angle = np.deg2rad(30)

    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

    translation = np.array([
        0.08,
        0.02,
        0.01
    ])

    target = (
        source @ rotation.T
        + translation
    )

    target += rng.normal(
        scale=0.001,
        size=target.shape
    )

    return (
        source,
        target,
        rotation,
        translation
    )