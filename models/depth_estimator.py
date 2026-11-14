from pathlib import Path
import numpy as np


def load_depth_sequence(depth_dir):

    paths = sorted(
        Path(depth_dir).glob("*.npy")
    )

    if not paths:
        raise FileNotFoundError(
            "No depth files found"
        )

    return [
        np.load(path).astype(np.float32)
        for path in paths
    ]


def estimate_depth(
    frames,
    backend="provided_depth",
    depth_maps=None,
    **kwargs
):

    if backend == "provided_depth":

        if depth_maps is None:
            raise ValueError("Provide depth_maps")

        if len(depth_maps) != len(frames):
            raise ValueError(
                "Depth and frame counts must match"
            )

        return [
            np.asarray(d, dtype=np.float32)
            for d in depth_maps
        ]

    if backend == "megasam":

        raise NotImplementedError(
            "Connect official MegaSaM inference and checkpoints."
        )

    raise ValueError("Unsupported depth backend")