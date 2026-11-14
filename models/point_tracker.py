import cv2
import numpy as np


def track_points_2d(
    frames,
    initial_points=None,
    max_corners=150
):

    if len(frames) < 2:
        raise ValueError("At least two frames required")

    gray = []

    for frame in frames:

        if frame.ndim == 3:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2GRAY
            )

        gray.append(frame)

    if initial_points is None:

        points = cv2.goodFeaturesToTrack(
            gray[0],
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=7
        )

    else:

        points = np.asarray(
            initial_points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

    if points is None:
        raise ValueError("No trackable points found")

    current = points.astype(np.float32)

    tracks = [
        current.reshape(-1, 2).copy()
    ]

    visibility = [
        np.ones(len(current), dtype=bool)
    ]

    for i in range(1, len(gray)):

        next_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                gray[i - 1],
                gray[i],
                current,
                None,
                winSize=(21, 21),
                maxLevel=3
            )
        )

        if next_points is None:

            status = np.zeros(
                (len(current), 1),
                dtype=np.uint8
            )

            next_points = current.copy()

        good = status.reshape(-1).astype(bool)

        current = next_points

        tracks.append(
            current.reshape(-1, 2).copy()
        )

        visibility.append(good)

    return (
        np.stack(tracks),
        np.stack(visibility)
    )