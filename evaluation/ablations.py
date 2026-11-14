ABLATIONS = [

    "full_pipeline",

    "remove_depth_calibration",

    "remove_vlm_candidate_selection",

    "remove_goal_image",

    "replace_3d_flow_with_2d_flow"

]


def build_ablation_plan():

    results = []

    for name in ABLATIONS:

        results.append({

            "experiment": name,

            "description": name.replace(
                "_",
                " "
            )

        })

    return results