"""Check core.perception.multi_block on synthetic depth images rendered through the real calibration.

    .venv\Scripts\python -m tests.test_multi_block
"""
import numpy as np

from core.perception.calibrate import load_calib
from core.perception.kinect import RGB_INTR
from core.perception.multi_block import find_blocks

S = 0.03
T = load_calib()
INTR = RGB_INTR


def render(boxes, noise=0.003, seed=0, extra=()):
    """boxes: (x, y, z_centre, yaw) cubes of side S. extra: (x, y, z_centre, yaw, sx, sy, sz) other boxes."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640]
    d_c = np.stack([(u - INTR["cx"]) / INTR["fx"], (v - INTR["cy"]) / INTR["fy"], np.ones_like(u, float)], -1)
    d = d_c @ T[:3, :3].T
    o = T[:3, 3]
    depth = -o[2] / d[..., 2]                      # table plane z = 0
    depth = np.where(depth > 0, depth, np.inf)
    allb = [(x, y, z, yaw, S, S, S) for x, y, z, yaw in boxes] + list(extra)
    for x, y, z, yaw, sx, sy, sz in allb:
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ol = R @ (o - np.array([x, y, z]))
        dl = d @ R.T
        hs = np.array([sx, sy, sz]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-hs - ol) / dl, (hs - ol) / dl
        tmin = np.minimum(t1, t2).max(-1)
        tmax = np.maximum(t1, t2).min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        depth = np.where(hit & (tmin < depth), tmin, depth)
    depth = depth + rng.normal(0, noise, depth.shape)
    return depth.astype(np.float32)


def near(obs, x, y):
    return min(obs, key=lambda b: np.hypot(*(b.xy - (x, y))))


def check(found, expect, tol=0.008, period=np.pi / 2):
    assert len(found) == len(expect), f"found {len(found)} blocks, expected {len(expect)}: {[b.xy.round(3) for b in found]}"
    for x, y, level, yaw in expect:
        b = near(found, x, y)
        err = np.hypot(*(b.xy - (x, y)))
        dyaw = abs((b.yaw - yaw + period / 2) % period - period / 2)
        assert err < tol, f"({x},{y}) off by {err*1000:.1f} mm"
        assert b.level == level, f"({x},{y}) level {b.level}, expected {level}"
        assert dyaw < np.radians(6), f"({x},{y}) yaw error {np.degrees(dyaw):.1f} deg"
        print(f"  block ({x:+.2f},{y:+.2f}) -> {b.xy.round(3)} err {err*1000:.1f} mm, level {b.level}, "
              f"top {b.top_z*100:.1f} cm, yaw err {np.degrees(dyaw):.1f} deg")


def main():
    print("three loose blocks, random yaws")
    blocks = [(0.18, 0.08, S / 2, 0.3), (0.24, -0.08, S / 2, -0.5), (0.20, 0.0, S / 2, 0.0)]
    check(find_blocks(render(blocks), INTR, T, S), [(x, y, 1, yaw) for x, y, _, yaw in blocks])

    print("a stack of two plus a loose block")
    blocks = [(0.20, 0.0, S / 2, 0.2), (0.20, 0.0, 1.5 * S, 0.25), (0.25, 0.06, S / 2, 0.0)]
    found = find_blocks(render(blocks), INTR, T, S)
    check(found, [(0.20, 0.0, 2, 0.25), (0.25, 0.06, 1, 0.0)])

    print("a stack of three")
    blocks = [(0.20, 0.0, S / 2 + k * S, 0.1) for k in range(3)]
    check(find_blocks(render(blocks), INTR, T, S), [(0.20, 0.0, 3, 0.1)])

    print("noise seeds")
    blocks = [(0.17, 0.08, S / 2, 0.7), (0.27, -0.12, S / 2, -0.2), (0.21, 0.01, S / 2, 0.1)]
    for seed in range(5):
        check_quiet = find_blocks(render(blocks, seed=seed), INTR, T, S)
        assert len(check_quiet) == 3, (seed, len(check_quiet))
    print("  5/5 seeds found all three")

    print("two touching blocks are rejected, not guessed at")
    rej = []
    found = find_blocks(render([(0.20, 0.0, S / 2, 0.0), (0.20, 0.031, S / 2, 0.0)]), INTR, T, S, rejected=rej)
    assert found == [] and len(rej) == 1, (found, rej)
    print("  rejected:", rej[0][1])

    print("excluded disc (where the parked arm sits) is ignored")
    found = find_blocks(render([(0.27, 0.05, S / 2, 0.0), (0.20, 0.0, S / 2, 0.0)]), INTR, T, S,
                        exclude=[(0.27, 0.05, 0.05)])
    assert len(found) == 1 and abs(found[0].xy[0] - 0.20) < 0.01
    print("elongated 8 x 5 x 3 cm boxes: yaw is the short axis, period 180 deg")
    def box(x, y, yaw, z=0.015):
        return (x, y, z, yaw, 0.05, 0.08, 0.03)          # short side along the box's x axis, long along y
    extra = [box(0.30, -0.18, 0.4), box(0.27, -0.04, -1.0), box(0.33, 0.07, 2.0)]
    found = find_blocks(render([], extra=extra), INTR, T, 0.03, footprint=(0.08, 0.05))
    check(found, [(0.30, -0.18, 1, 0.4), (0.27, -0.04, 1, -1.0 % np.pi), (0.33, 0.07, 1, 2.0 % np.pi)], period=np.pi)
    for b in found:
        assert abs(b.size[0] - 0.08) < 0.006 and abs(b.size[1] - 0.05) < 0.006, b.size
        assert abs(b.period - np.pi) < 1e-9

    print("a stack of two boxes")
    extra = [box(0.25, -0.05, 0.3), box(0.25, -0.05, 0.3, z=0.045)]
    found = find_blocks(render([], extra=extra), INTR, T, 0.03, footprint=(0.08, 0.05))
    check(found, [(0.25, -0.05, 2, 0.3)], period=np.pi)
    print("ALL OK")


if __name__ == "__main__":
    main()
