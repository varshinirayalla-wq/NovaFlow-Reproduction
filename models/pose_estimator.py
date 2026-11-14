import numpy as np


def kabsch(source, target):

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError("Point shapes must match")

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")

    if len(source) < 3:
        raise ValueError("At least 3 points required")

    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)

    source_centered = source - centroid_source
    target_centered = target - centroid_target

    covariance = source_centered.T @ target_centered

    u, _, vt = np.linalg.svd(covariance)

    correction = np.eye(3)

    correction[2, 2] = np.linalg.det(vt.T @ u.T)

    rotation = vt.T @ correction @ u.T

    translation = (
        centroid_target -
        rotation @ centroid_source
    )

    return rotation, translation


def transform_points(points, rotation, translation):

    points = np.asarray(points, dtype=np.float64)

    return points @ rotation.T + translation


def kabsch_ransac(
    source,
    target,
    iterations=200,
    threshold=0.01,
    seed=42
):

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError("Point shapes must match")

    n = len(source)

    if n < 4:
        if n < 3:
            raise ValueError("At least 3 points required")

        r, t = kabsch(source, target)

        return r, t, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)

    best = np.zeros(n, dtype=bool)

    for _ in range(iterations):

        ids = rng.choice(
            n,
            size=4,
            replace=False
        )

        r, t = kabsch(
            source[ids],
            target[ids]
        )

        predicted = transform_points(
            source,
            r,
            t
        )

        errors = np.linalg.norm(
            predicted - target,
            axis=1
        )

        inliers = errors <= threshold

        if inliers.sum() > best.sum():
            best = inliers

    if best.sum() < 3:
        best[:] = True

    rotation, translation = kabsch(
        source[best],
        target[best]
    )

    return rotation, translation, best


def rotation_error_degrees(r_est, r_true):

    relative = r_est.T @ r_true

    value = np.clip(
        (np.trace(relative) - 1) / 2,
        -1,
        1
    )

    return float(
        np.degrees(np.arccos(value))
    )