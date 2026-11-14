from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset"
SIMULATED_DIR = DATASET_DIR / "simulated"
REAL_DIR = DATASET_DIR / "real"

OUTPUT_DIR = ROOT / "outputs"

VIDEO_DIR = OUTPUT_DIR / "generated_videos"
DEPTH_DIR = OUTPUT_DIR / "depth_maps"
TRACK_DIR = OUTPUT_DIR / "tracked_points"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"
RESULTS_DIR = OUTPUT_DIR / "results"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

NUM_FRAMES = 41

RANSAC_ITERATIONS = 200
RANSAC_INLIER_THRESHOLD_M = 0.01

TRACK_VISIBILITY_MIN = 0.5

RANDOM_SEED = 42


def create_directories():

    directories = [
        SIMULATED_DIR,
        REAL_DIR,
        VIDEO_DIR,
        DEPTH_DIR,
        TRACK_DIR,
        TRAJECTORY_DIR,
        RESULTS_DIR,
        ROOT / "models",
        ROOT / "simulation",
        ROOT / "evaluation",
        ROOT / "tests"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)