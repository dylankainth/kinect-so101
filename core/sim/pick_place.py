"""Scripted pick-and-place planner + MuJoCo executor for the SO-101.

plan_pick_place(block_xyz, target_xyz, block_yaw=0.0) -> list[dict]
    Dense joint-target trajectory sampled at PLAN_DT (50 Hz). Each waypoint is a dict with the exact keys
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    Units: all values RADIANS in the MJCF joint convention (see README.md).
      gripper: MJCF `gripper` jaw joint angle, rad. GRIPPER_OPEN = 0.8 rad (~46 deg, ~4.5 cm pad gap),
               GRIPPER_CLOSED = -0.10 rad (just under the fully-closed limit -0.17; jaws stall on the cube).
    block_xyz / target_xyz: CENTRE of the cube, metres, base frame (x fwd, y left, z up). Target z is the
    cube-centre height when resting on the table (defaults to the block's z).
"""
import numpy as np
import mujoco

from .kinematics import SO101Kinematics, JOINTS, SCENE

PLAN_DT = 0.02  # s, control period of the waypoint stream
GRIPPER_OPEN = 0.8
GRIPPER_CLOSED = -0.10
HOVER = 0.065         # m, pre-grasp/pre-place height of the grasp point above the cube-centre rest height
TRAVEL_Z_MIN = 0.08   # m, grasp-point height while carrying (vertical tool cannot reach much higher)
V_APPROACH = 0.05     # m/s descend/ascend speed
V_TRANSFER = 0.12     # m/s lateral speed
W_JOINT = 0.7         # rad/s peak-ish joint speed for joint-space moves
HOME_XYZ = (0.20, 0.0, 0.08)

_K = None


def _kin():
    global _K
    if _K is None:
        _K = SO101Kinematics()
    return _K


def _min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _wp(q5, g):
    return {n: float(v) for n, v in zip(JOINTS, list(q5) + [g])}


def home_pose():
    k = _kin()
    q, _ = k.solve(HOME_XYZ, tilts=(0.0, 0.3, 0.6))
    return _wp(np.append(q, 0.0), GRIPPER_OPEN)


def _joint_move(q_from, q_to, vmax=W_JOINT, min_t=0.5):
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    T = max(min_t, 1.875 * np.max(np.abs(q_to - q_from)) / vmax)  # min-jerk peak speed = 1.875*dist/T
    n = max(2, int(np.ceil(T / PLAN_DT)))
    return [q_from + (q_to - q_from) * _min_jerk((i + 1) / n) for i in range(n)]


def _cart_move(xyz_from, xyz_to, q_start, tilt, roll, speed):
    """Straight-line move of the grasp point with IK warm-started on the previous solution.
    Returns list of 5-joint arrays (incl. roll)."""
    k = _kin()
    xyz_from, xyz_to = np.asarray(xyz_from, float), np.asarray(xyz_to, float)
    dist = np.linalg.norm(xyz_to - xyz_from)
    T = max(0.3, 1.875 * dist / speed)
    n = max(2, int(np.ceil(T / PLAN_DT)))
    out, q = [], np.array(q_start, float)
    for i in range(n):
        p = xyz_from + (xyz_to - xyz_from) * _min_jerk((i + 1) / n)
        q4, pe, te = k.ik(p, tilt, q0=np.append(q[:4], 0.0), roll=roll)
        if pe > 2e-3:
            raise ValueError(f"unreachable waypoint {p.round(3)} (err {pe*1000:.1f} mm)")
        q = np.append(q4, roll)
        out.append(q.copy())
    return out


def plan_pick_place(block_xyz, target_xyz=None, block_yaw=0.0, tilt=None, yaw_period=np.pi / 2):
    """See module docstring. Raises ValueError if the block or target is unreachable."""
    k = _kin()
    block = np.asarray(block_xyz, float)
    target = np.asarray(target_xyz if target_xyz is not None else block, float).copy()
    if target_xyz is not None and len(target) == 3 and target[2] <= 0:
        target[2] = block[2]
    zr = block[2]  # cube-centre height at rest on table
    tgt_rest = np.array([target[0], target[1], zr])

    pre_g = block + [0, 0, HOVER]
    pre_t = tgt_rest + [0, 0, HOVER]
    zt = max(TRAVEL_Z_MIN, pre_g[2])
    pre_t[2] = pre_g[2] = zt
    place = tgt_rest + [0, 0, 0.004]  # release 4 mm above the table

    # choose the approach tilt that works at all key points
    tilts = (0.0, 0.15, 0.3, 0.45) if tilt is None else (tilt,)
    chosen = None
    for t in tilts:
        try:
            qs = [k.solve(p, tilts=(t,))[0] for p in (pre_g, block, pre_t, place)]
            chosen = t
            break
        except ValueError:
            continue
    if chosen is None:
        raise ValueError("block/target outside reachable workspace")
    # jaw roll aligning the closing axis with the cube faces (cube is 90-deg symmetric); the grasp point is
    # offset from the roll axis so re-solve the pick IK with the roll included, then keep roll constant.
    roll_pick = k.roll_for_yaw(qs[1], block_yaw, yaw_period)
    qs[1] = k.solve(block, tilts=(chosen,), roll=roll_pick)[0]
    qs[0] = k.solve(pre_g, tilts=(chosen,), roll=roll_pick)[0]
    roll_place = roll_pick

    traj = []  # list of (q5, gripper)

    def add(qs5, g):
        for q in qs5:
            traj.append((np.asarray(q, float), g))

    h = home_pose()
    hq = np.array([h[n] for n in JOINTS[:5]])
    # 1. home -> pre-grasp (joint space, with roll set for cube yaw)
    q_pre = np.append(qs[0], roll_pick)
    add(_joint_move(hq, q_pre), GRIPPER_OPEN)
    cur = q_pre
    add([cur] * 10, GRIPPER_OPEN)
    # 2. descend
    seg = _cart_move(pre_g, block, cur, chosen, roll_pick, V_APPROACH); add(seg, GRIPPER_OPEN); cur = seg[-1]
    add([cur] * 15, GRIPPER_OPEN)
    # 3. close (ramp the jaw command over 0.6 s, then hold 0.6 s)
    n = int(0.6 / PLAN_DT)
    for i in range(n):
        traj.append((cur, GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * _min_jerk((i + 1) / n)))
    add([cur] * 30, GRIPPER_CLOSED)
    # 4. lift
    seg = _cart_move(block, pre_g, cur, chosen, roll_pick, V_APPROACH); add(seg, GRIPPER_CLOSED); cur = seg[-1]
    # 5. transfer above the target
    seg = _cart_move(pre_g, pre_t, cur, chosen, roll_place, V_TRANSFER); add(seg, GRIPPER_CLOSED); cur = seg[-1]
    add([cur] * 10, GRIPPER_CLOSED)
    # 6. descend to place
    seg = _cart_move(pre_t, place, cur, chosen, roll_place, V_APPROACH); add(seg, GRIPPER_CLOSED); cur = seg[-1]
    add([cur] * 15, GRIPPER_CLOSED)
    # 7. open (ramp 0.5 s, hold 0.4 s)
    n = int(0.5 / PLAN_DT)
    for i in range(n):
        traj.append((cur, GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * _min_jerk((i + 1) / n)))
    add([cur] * 20, GRIPPER_OPEN)
    # 8. retreat straight up, 9. home
    seg = _cart_move(place, pre_t, cur, chosen, roll_place, V_APPROACH); add(seg, GRIPPER_OPEN); cur = seg[-1]
    add(_joint_move(cur, hq), GRIPPER_OPEN)
    add([hq] * 10, GRIPPER_OPEN)
    return [_wp(q, g) for q, g in traj]


# ----------------------------------------------------------------------------------------------
# Execution in simulation
# ----------------------------------------------------------------------------------------------
def set_cube_pose(model, data, xyz, yaw):
    j = model.joint("cube_joint")
    a = model.jnt_qposadr[j.id]
    data.qpos[a:a + 3] = xyz
    data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qvel[model.jnt_dofadr[j.id]:model.jnt_dofadr[j.id] + 6] = 0


def execute_plan(plan, block_xyz, target_xyz, block_yaw=0.0, latency_steps=0, model=None, data=None, viewer=None,
                 settle_time=1.0):
    """Run `plan` in sim with the cube at block_xyz. latency_steps = number of PLAN_DT control steps the
    command is delayed. Returns dict(success, cube_xyz, err_xy, on_table, max_cube_height)."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(SCENE)
    data = data if data is not None else mujoco.MjData(model)
    aid = [model.actuator(n).id for n in JOINTS]
    cube_geom = model.geom("cube_geom")
    half = float(cube_geom.size[2])
    h = plan[0]
    for n in JOINTS:
        data.qpos[model.joint(n).qposadr] = h[n]
    set_cube_pose(model, data, np.asarray(block_xyz, float), block_yaw)
    mujoco.mj_forward(model, data)
    nsub = int(round(PLAN_DT / model.opt.timestep))
    cub = model.body("cube").id
    maxz = 0.0

    def step(cmd):
        nonlocal maxz
        data.ctrl[aid] = [cmd[n] for n in JOINTS]
        for _ in range(nsub):
            mujoco.mj_step(model, data)
        maxz = max(maxz, data.xpos[cub][2])
        if viewer is not None:
            viewer.sync()

    for i in range(len(plan)):
        step(plan[max(0, i - latency_steps)])
    for _ in range(int(settle_time / PLAN_DT)):
        step(plan[-1])
    pos = data.xpos[cub].copy()
    err = float(np.linalg.norm(pos[:2] - np.asarray(target_xyz)[:2]))
    up = data.xmat[cub].reshape(3, 3)[2, 2]  # cube z-axis alignment (flat on table => ~1)
    on_table = bool(abs(pos[2] - half) < 0.004 and up > 0.97)
    return dict(success=bool(err < 0.03 and on_table), cube_xyz=pos, err_xy=err, on_table=on_table,
                max_cube_height=maxz)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="open interactive MuJoCo viewer")
    a = ap.parse_args()
    block, target = (0.20, 0.12, 0.015), (0.20, -0.12, 0.015)
    plan = plan_pick_place(block, target)
    print(f"{len(plan)} waypoints, {len(plan)*PLAN_DT:.1f} s")
    if a.view:
        import mujoco.viewer
        m = mujoco.MjModel.from_xml_path(SCENE)
        d = mujoco.MjData(m)
        with mujoco.viewer.launch_passive(m, d) as v:
            print(execute_plan(plan, block, target, model=m, data=d, viewer=v))
    else:
        print(execute_plan(plan, block, target))
