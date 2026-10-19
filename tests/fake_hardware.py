"""Stand-ins for SafeArm and the Kinect so run logic can be exercised with no hardware attached."""
import numpy as np

from core.robot.arm import JOINTS


class FakeSafeArm:
    """Same interface core.grasp uses. `grip_reads` is what settle_gripper() reports on successive closes;
    `fail_on_follow` raises on that call number (1-based) to imitate a safety abort."""
    instances = []
    grip_reads = [25.0]
    fail_on_follow = None

    def __init__(self, allow_writes=False, **_):
        self.allow_writes = allow_writes
        self.mismatch = []
        self.max_jump_deg = 15.0
        self.trace = []
        self.stop_requested = lambda: False
        self.pos = {"shoulder_pan": -4.6, "shoulder_lift": -97.0, "elbow_flex": 94.0, "wrist_flex": 80.0,
                    "wrist_roll": 0.0, "gripper": 30.0}
        self.log = []
        self.follows = 0
        self.torque = False
        FakeSafeArm.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.log.append("exit")

    def read_positions(self):
        return dict(self.pos)

    def enable_torque(self, tries=4):
        self.torque = True
        self.log.append("torque_on")

    def disable_torque(self):
        self.torque = False
        self.log.append("torque_off")

    def stop(self):
        self.log.append("stop")

    def clamp(self, f):
        return dict(f)

    def move_to(self, targets, max_speed_deg_s=30.0, dry_run=False, allow_big_jump=False):
        self.log.append("move_dry" if dry_run else "move")
        if not dry_run:
            self.pos = dict(targets)

    def follow(self, frames, rate_hz, **_):
        self.follows += 1
        if FakeSafeArm.fail_on_follow == self.follows or self.stop_requested():
            raise RuntimeError("fake safety abort")
        if frames:
            self.pos = dict(frames[-1])
        reads = FakeSafeArm.grip_reads
        if self.log.count("settle") and len(reads) > 1:   # the jaws report what they sense, not the command
            self.pos["gripper"] = reads[1]
        self.log.append(f"follow:{len(frames)}")

    def settle_gripper(self, target, **_):
        self.log.append("settle")
        reads = FakeSafeArm.grip_reads
        g = reads[min(self.log.count("settle") - 1, len(reads) - 1)]
        self.pos["gripper"] = g
        return g

    @classmethod
    def reset(cls, grip_reads=(25.0,), fail_on_follow=None):
        cls.instances = []
        cls.grip_reads = list(grip_reads)
        cls.fail_on_follow = fail_on_follow


class FakeKinect:
    intr = dict(fx=529.2, fy=525.6, cx=328.9, cy=267.5)

    def start(self):
        return self

    def stop(self):
        pass

    def wait_for_frames(self, timeout=5.0):
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.zeros((480, 640), np.uint16)
        return rgb, 0.0, depth, 0.0

    get_frames = wait_for_frames

    def depth_to_meters(self, d):
        return np.full(d.shape, np.nan, np.float32)
