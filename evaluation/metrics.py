import numpy as np


def endpoint_error(predicted, ground_truth):

    p = np.asarray(predicted, dtype=float)
    g = np.asarray(ground_truth, dtype=float)

    if p.shape != g.shape:
        raise ValueError("Shapes must match")

    return float(
        np.mean(
            np.linalg.norm(p - g, axis=-1)
        )
    )


def translation_error(t_est, t_true):

    return float(
        np.linalg.norm(
            np.asarray(t_est) - np.asarray(t_true)
        )
    )


def success_rate(success_flags):

    values = np.asarray(
        success_flags,
        dtype=bool
    )

    if values.size == 0:
        return float("nan")

    return float(values.mean())


def wilson_interval(successes, total, z=1.96):

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid success counts")

    p = successes / total

    denominator = 1 + z*z/total

    center = (
        p + z*z/(2*total)
    ) / denominator

    margin = (
        z * np.sqrt(
            p*(1-p)/total
            + z*z/(4*total*total)
        )
    ) / denominator

    return (
        float(center - margin),
        float(center + margin)
    )