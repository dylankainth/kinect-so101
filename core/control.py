"""Flag files shared by the stop button (core/tools/stop_gui.py) and the arm runner.

control/stop   - abort the current motion and hold position
control/park   - (after a stop) return slowly to the rest pose, then go limp
control/limp   - torque off now
control/heartbeat - touched by the runner while it owns the serial port
control/status - one line of runner status shown in the GUI
"""
import time
from .paths import CONTROL_DIR

DIR = CONTROL_DIR
DIR.mkdir(exist_ok=True)
STOP, PARK, LIMP = DIR / "stop", DIR / "park", DIR / "limp"
HEARTBEAT, STATUS = DIR / "heartbeat", DIR / "status"


def request(flag):
    flag.write_text(str(time.time()))


def is_set(flag):
    return flag.exists()


def clear(*flags):
    for f in flags or (STOP, PARK, LIMP):
        f.unlink(missing_ok=True)


def beat(status=None):
    HEARTBEAT.write_text(str(time.time()))
    if status is not None:
        STATUS.write_text(status)


def runner_alive(max_age=2.0):
    try:
        return time.time() - float(HEARTBEAT.read_text()) < max_age
    except (OSError, ValueError):
        return False
