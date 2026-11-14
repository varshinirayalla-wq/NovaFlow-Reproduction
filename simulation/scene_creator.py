from pathlib import Path
import json
import numpy as np
import cv2

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    NUM_FRAMES,
    RANDOM_SEED
)


def create_synthetic_scene(output_dir):

    out = Path(output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT

    frames = []
    depths = []

    x0 = width // 4
    y0 = height // 2

    for i in range(NUM_FRAMES):

        t = i / (NUM_FRAMES - 1)

        x = int(
            x0 + t * (width * 0.42)
        )

        frame = np.full(
            (height, width, 3),
            235,
            dtype=np.uint8
        )

        cv2.rectangle(
            frame,
            (0, height - 90),
            (width, height),
            (190, 190, 190),
            -1
        )

        cv2.circle(
            frame,
            (x, y0),
            34,
            (220, 40, 30),
            -1
        )

        cv2.putText(
            frame,
            "SYNTHETIC TEST SCENE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2
        )

        depth = np.full(
            (height, width),
            0.8,
            dtype=np.float32
        )

        cv2.circle(
            depth,
            (x, y0),
            34,
            0.65,
            -1
        )

        frames.append(frame)
        depths.append(depth)

    video_path = out / "synthetic_demo.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height)
    )

    for frame in frames:

        writer.write(
            cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )
        )

    writer.release()

    np.save(
        out / "initial_depth.npy",
        depths[0]
    )

    np.save(
        out / "depth_sequence.npy",
        np.stack(depths)
    )

    metadata = {
        "scene_id": "synthetic_demo_001",
        "task": "Move the red circular object to the right",
        "is_synthetic": True,
        "camera_intrinsics": {
            "fx": 600,
            "fy": 600,
            "cx": width / 2,
            "cy": height / 2
        },
        "num_frames": NUM_FRAMES,
        "note": "Synthetic software testing scene"
    }

    (
        out / "metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4)
    )

    return video_path, np.stack(depths), metadata