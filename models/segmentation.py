def segment_object(
    frames,
    boxes,
    backend="sam2",
    **kwargs
):

    if backend != "sam2":
        raise ValueError("Unsupported backend")

    raise NotImplementedError(
        "Connect the official SAM 2 video predictor "
        "and checkpoint."
    )