"""Where things live. Everything resolves from the repo root, whatever the current directory is."""
from pathlib import Path

CORE = Path(__file__).parent
REPO = CORE.parent
CONTROL_DIR = REPO / "control"           # runtime state and logs (gitignored)
CALIB_PATH = CORE / "perception" / "calib.json"
JOINT_OFFSETS_PATH = CORE / "robot" / "joint_offsets.json"
