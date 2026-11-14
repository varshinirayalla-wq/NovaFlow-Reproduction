BASELINES = [

    "full_pipeline",

    "raft_2d_flow_baseline",

    "without_depth_calibration",

    "without_candidate_rejection",

    "without_goal_image",

    "oracle_flow_simulator_upper_bound"

]


def comparison_record(method, metrics):

    if method not in BASELINES:
        raise ValueError("Unknown baseline")

    return {
        "method": method,
        **metrics
    }