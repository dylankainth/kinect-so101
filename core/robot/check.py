"""Read-only diagnostic. Never writes a register, never enables torque.

    .venv\\Scripts\\python -m core.robot.check [COM4] [--source file|motor]
"""
import logging
import sys

from .arm import JOINTS, SafeArm
from .mapping import real_to_sim

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
args = [a for a in sys.argv[1:] if a.upper().startswith("COM")]
port = args[0] if args else "COM4"
source = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "file"

with SafeArm(port=port, allow_writes=False, calibration_source=source) as arm:
    raw = arm.read_raw()
    pos = arm.read_positions()
    sim = real_to_sim(pos)
    st = arm.read_status(full=True)
    print(f"calibration source: {source}   (writes disabled)")
    print(f"{'joint':14s} {'raw':>5s} {'deg/pct':>9s} {'sim rad':>8s} {'soft lo..hi':>16s} {'temp':>5s} {'volt':>5s} {'load':>5s} {'cur':>4s} torque")
    for j in JOINTS:
        print(f"{j:14s} {raw[j]:5d} {pos[j]:9.2f} {sim[j]:8.3f} {arm.lo[j]:7.1f}..{arm.hi[j]:6.1f} "
              f"{st['temp'][j]:4d}C {st['volt'][j]:4.1f}V {st['load'][j]:5d} {st['current'][j]:4d} {st['torque'][j]}")
    print("gripper units: 0-100 (0 closed, 100 open); other joints degrees")
    if arm.mismatch:
        print("\nWARNING calibration file != calibration stored in motors (homing_offset, range_min, range_max):")
        for j, m in arm.mismatch.items():
            print(f"  {j:14s} file={m['file']}  motor={m['motor']}  frame shift applied={arm.corr[j]:+d} ticks")
    again = arm.read_status(full=True)["torque"]
    ok = all(v == 0 for v in again.values())
    print(f"\nTorque_Enable re-read: {again}  -> {'OK, torque still OFF' if ok else 'TORQUE IS ON!'}")
    if any(t >= 50 for t in st["temp"].values()):
        print("WARNING: a servo is >= 50C")
