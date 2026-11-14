def detect_object(
    frames,
    text_prompt,
    backend="grounding_dino",
    **kwargs
):

    if backend != "grounding_dino":
        raise ValueError("Unsupported backend")

    raise NotImplementedError(
        "Connect the official Grounding DINO "
        "repository, checkpoint and preprocessing."
    )