import numpy as np


def select_candidate(candidates, scores=None):

    if not candidates:
        raise ValueError("No candidates available")

    if scores is None:
        raise ValueError("Provide candidate scores")

    scores = np.asarray(scores, dtype=float)

    if len(scores) != len(candidates):
        raise ValueError("Candidate and score counts differ")

    if not np.isfinite(scores).any():
        raise ValueError("No valid scores")

    index = int(np.nanargmax(scores))

    return candidates[index], index