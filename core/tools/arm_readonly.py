"""Read-only SO-101 check: ping servos and print joint positions. Never enables torque."""
import sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
bus = FeetechMotorsBus(
    port=port,
    motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(names)},
)
bus.connect(handshake=False)
try:
    print("found:", bus.broadcast_ping())
    for n in names:
        raw = bus.read("Present_Position", n, normalize=False)
        temp = bus.read("Present_Temperature", n, normalize=False)
        volt = bus.read("Present_Voltage", n, normalize=False)
        print(f"{n:14s} raw={raw:5d}  temp={temp}C  volt={volt/10:.1f}V")
finally:
    bus.disconnect(disable_torque=False)
