from pathlib import Path
import cv2


def load_video(video_path):

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames found")

    return frames


def generate_video(
    image_path,
    instruction,
    output_path=None,
    backend="input_video",
    input_video=None,
    **kwargs
):

    if backend == "input_video":

        if input_video is None:
            raise ValueError("Provide input_video")

        return load_video(input_video)

    if backend == "wan2.1":

        raise NotImplementedError(
            "Connect the official Wan2.1 model and checkpoint."
        )

    raise ValueError("Unsupported video backend")