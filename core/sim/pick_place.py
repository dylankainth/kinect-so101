"""Scripted pick-and-place planner + MuJoCo executor for the SO-101.

plan_pick_place(block_xyz, target_xyz, block_yaw=0.0) -> list[dict]
    Dense joint-target trajectory sampled at PLAN_DT (50 Hz). Each waypoint is a dict with the exact keys
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    Units: all values RADIANS in the MJCF joint convention (see README.md).
      gripper: MJCF `gripper` jaw joint angle, rad. GRIPPER_OPEN = 0.8 rad (~46 deg, ~6 cm between the pad faces),
               GRIPPER_CLOSED = -0.10 rad (just under the fully-closed limit -0.17; jaws stall on the cube).
    block_xyz / target_xyz: CENTRE of the cube, metres, base frame (x fwd, y left, z up). Target z is the
    cube-centre height when resting on the table (defaults to the block's z). To put the block on top of
    something (a stack), pass place_z = the cube-centre height at the place spot.
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
    r0, r1 = roll if isinstance(roll, (tuple, list)) else (roll, roll)   # roll may ramp over the move
    T = max(0.3, 1.875 * dist / speed, 1.875 * abs(r1 - r0) / W_JOINT)
    n = max(2, int(np.ceil(T / PLAN_DT)))
    out, q = [], np.array(q_start, float)
    for i in range(n):
        s = _min_jerk((i + 1) / n)
        p = xyz_from + (xyz_to - xyz_from) * s
        r = r0 + (r1 - r0) * s
        q4, pe, te = k.ik(p, tilt, q0=np.append(q[:4], 0.0), roll=r)
        if pe > 2e-3:
            raise ValueError(f"unreachable waypoint {p.round(3)} (err {pe*1000:.1f} mm)")
        q = np.append(q4, r)
        out.append(q.copy())
    return out


def jaw_frame(q5, gripper):
    """Where the two pads sit along the jaws' closing axis for the pose q5 (5 arm joints, rad) with the gripper
    at `gripper` rad. Returns (u, s_fixed, s_moving, t): u = horizontal unit vector from the swinging pad to the
    fixed pad; s_* = pad centres along u measured from the grasp point (m); t = pad half thickness (m)."""
    k = _kin()
    d = mujoco.MjData(k.m)
    for n, v in zip(JOINTS, list(q5) + [gripper]):
        d.qpos[k.m.joint(n).qposadr] = v
    mujoco.mj_forward(k.m, d)
    gp = d.site_xpos[k.m.site("grasp_point").id]
    pf, pm = d.geom_xpos[k.m.geom("fixed_pad").id], d.geom_xpos[k.m.geom("moving_pad").id]
    u = (pf - pm)[:2]
    u = u / np.linalg.norm(u)
    return u, float((pf - gp)[:2] @ u), float((pm - gp)[:2] @ u), float(k.m.geom("fixed_pad").size[0])


def plan_pick_place(block_xyz, target_xyz=None, block_yaw=0.0, tilt=None, yaw_period=np.pi / 2, place_z=None,
                    gripper_open=None, held_width=None, place_hover=None, place_yaw=None, held_gripper=None,
                    release_clear=0.0):
    """See module docstring. Raises ValueError if the block or target is unreachable.

    gripper_open: jaw angle (rad) used wherever the plan says "open" (default GRIPPER_OPEN).
    held_width: the block's width between the jaws (m). The grasp point is not the middle of the open jaws (one
    jaw is fixed, the other swings), so a block wider than about 4 cm centred on the grasp point lands on the
    fixed pad. With held_width the planner moves the pick so the middle of the open jaws is on the block, and
    the place so the block, pressed against the fixed pad while carried, ends on the target. block_xyz and
    target_xyz then mean the block's centre at pick and at place. Default None keeps the old behaviour.
    place_hover: height (m) of the approach point above the place spot (default HOVER). Stacking high wants a
    small one: the arm can reach the place spot itself well before it can reach 6.5 cm above it.
    place_yaw: world direction the jaws should close along at the place (default: the same wrist roll as at the
    pick). Giving the stack's own orientation picks the roll for the place pose, ramps it during the transfer,
    and lines the placed block up with the one below.
    held_gripper: jaw angle (rad) commanded while the block is held (default GRIPPER_CLOSED, i.e. squeezing as
    hard as the jaws go). The real arm stalls on the block and eases off, so a rehearsal that wants the same
    grip force passes the angle the jaws stall at, minus a little.
    release_clear: distance (m) to slide the open gripper along the closing axis, away from the block, before
    lifting. The fixed pad touches the block when it is released, and lifting straight up can drag it along.
    Needs held_width (it is what finds the closing axis at the place pose)."""
    go = GRIPPER_OPEN if gripper_open is None else gripper_open
    gc = GRIPPER_CLOSED if held_gripper is None else held_gripper
    k = _kin()
    block = np.asarray(block_xyz, float)
    target = np.asarray(target_xyz if target_xyz is not None else block, float).copy()
    if target_xyz is not None and len(target) == 3 and target[2] <= 0:
        target[2] = block[2]
    zr = block[2]  # cube-centre height at rest on table
    tgt_rest = np.array([target[0], target[1], zr if place_z is None else place_z])

    pre_g = block + [0, 0, HOVER]
    pre_t = tgt_rest + [0, 0, HOVER if place_hover is None else place_hover]
    if place_z is None:
        pre_t[2] = pre_g[2] = max(TRAVEL_Z_MIN, pre_g[2])
    else:   # placing on a stack: keep the pick approach low (high points can be out of reach), rise on the way
        pre_g[2] = max(TRAVEL_Z_MIN, pre_g[2])
        pre_t[2] = max(TRAVEL_Z_MIN, pre_t[2])
    place = tgt_rest + [0, 0, 0.004]  # release 4 mm above the table

    def build(chosen, qs, block, pre_g, pre_t, place):
        """The whole trajectory for one approach tilt. Raises ValueError if a waypoint is out of reach."""
        # jaw roll aligning the closing axis with the cube faces (cube is 90-deg symmetric); the grasp point is
        # offset from the roll axis so re-solve the pick IK with the roll included, then keep roll constant.
        roll_pick = k.roll_for_yaw(qs[1], block_yaw, yaw_period)
        qs[1] = k.solve(block, tilts=(chosen,), roll=roll_pick)[0]
        roll_place = roll_pick if place_yaw is None else k.roll_for_yaw(qs[3], place_yaw, yaw_period)
        # the block is symmetric, so the roll is only fixed modulo yaw_period; take the one nearest the pick roll.
        # The block is held off the roll axis, so turning the wrist by ~180 deg instead of ~0 swings it by
        # centimetres (and takes longer): it would land that far off the target.
        roll_place += yaw_period * round((roll_pick - roll_place) / yaw_period)
        clear_dir = None
        if held_width is not None:
            u, sf, sm, tp = jaw_frame(np.append(qs[1], roll_pick), go)
            shift = np.r_[-(sf + sm) / 2 * u, 0.0]                      # open jaws' middle onto the block
            block, pre_g = block + shift, pre_g + shift
            qs[1] = k.solve(block, tilts=(chosen,), roll=roll_pick)[0]
            # at the place pose the pan differs, so the closing axis points elsewhere: find it there
            q_pl = k.solve(place, tilts=(chosen,), roll=roll_place)[0]
            u2, sf2, _, tp2 = jaw_frame(np.append(q_pl, roll_place), go)
            shift = np.r_[-(sf2 - tp2 - held_width / 2) * u2, 0.0]        # held block touches the fixed pad
            clear_dir = u2
            pre_t, place = pre_t + shift, place + shift
        qs[0] = k.solve(pre_g, tilts=(chosen,), roll=roll_pick)[0]

        traj = []  # list of (q5, gripper)

        def add(qs5, g):
            for q in qs5:
                traj.append((np.asarray(q, float), g))

        h = home_pose()
        hq = np.array([h[n] for n in JOINTS[:5]])
        # 1. home -> pre-grasp (joint space, with roll set for cube yaw)
        q_pre = np.append(qs[0], roll_pick)
        add(_joint_move(hq, q_pre), go)
        cur = q_pre
        add([cur] * 10, go)
        # 2. descend
        seg = _cart_move(pre_g, block, cur, chosen, roll_pick, V_APPROACH); add(seg, go); cur = seg[-1]
        add([cur] * 15, go)
        # 3. close (ramp the jaw command over 0.6 s, then hold 0.6 s)
        n = int(0.6 / PLAN_DT)
        for i in range(n):
            traj.append((cur, go + (gc - go) * _min_jerk((i + 1) / n)))
        add([cur] * 30, gc)
        # 4. lift
        seg = _cart_move(block, pre_g, cur, chosen, roll_pick, V_APPROACH); add(seg, gc); cur = seg[-1]
        # 5. transfer above the target
        seg = _cart_move(pre_g, pre_t, cur, chosen, (roll_pick, roll_place), V_TRANSFER); add(seg, gc); cur = seg[-1]
        add([cur] * 10, gc)
        # 6. descend to place
        seg = _cart_move(pre_t, place, cur, chosen, roll_place, V_APPROACH); add(seg, gc); cur = seg[-1]
        add([cur] * 15, gc)
        # 7. open (ramp 0.5 s, hold 0.4 s)
        n = int(0.5 / PLAN_DT)
        for i in range(n):
            traj.append((cur, gc + (go - gc) * _min_jerk((i + 1) / n)))
        add([cur] * 20, go)
        # 7b. slide the open gripper clear of the block (the fixed pad is touching it), 8. retreat straight up, 9. home
        lift_from, lift_to = place, pre_t
        if release_clear and clear_dir is not None:
            off = np.r_[clear_dir * release_clear, 0.0]
            seg = _cart_move(place, place + off, cur, chosen, roll_place, V_APPROACH); add(seg, go); cur = seg[-1]
            add([cur] * 5, go)
            lift_from, lift_to = place + off, pre_t + off
        seg = _cart_move(lift_from, lift_to, cur, chosen, roll_place, V_APPROACH); add(seg, go); cur = seg[-1]
        add(_joint_move(cur, hq), go)
        add([hq] * 10, go)
        return [_wp(q, g) for q, g in traj]

    # try the approach tilts in turn: the first one whose key points solve. When stacking (place_z / held_width)
    # the wrist roll can put a waypoint out of reach under a tilt whose key points solved without it, so then
    # a tilt whose trajectory fails is dropped for the next; the plain pick-and-place keeps its single choice.
    tilts = (0.0, 0.15, 0.3, 0.45) if tilt is None else (tilt,)
    retry = held_width is not None or place_z is not None
    failure = None
    for t in tilts:
        try:
            qs = [k.solve(p, tilts=(t,))[0] for p in (pre_g, block, pre_t, place)]
        except ValueError:
            continue
        try:
            return build(t, qs, block.copy(), pre_g.copy(), pre_t.copy(), place.copy())
        except ValueError as e:
            if not retry:
                raise
            failure = e
    raise failure or ValueError("block/target outside reachable workspace")


# ----------------------------------------------------------------------------------------------
# Execution in simulation
# ----------------------------------------------------------------------------------------------
def set_cube_pose(model, data, xyz, yaw):
    j = model.joint("cube_joint")
    a = model.jnt_qposadr[j.id]
    data.qpos[a:a + 3] = xyz
    data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qvel[model.jnt_dofadr[j.id]:model.jnt_dofadr[j.id] + 6] = 0


def scene_with_support(xy=None, top_z=0.0, half_xy=0.015, cube_half=None, yaw=0.0):
    """Sim model with a fixed box under (x, y) up to height top_z: stands in for the blocks already stacked.
    half_xy: the support's half footprint, a number or (hx, hy); yaw turns it about z. cube_half = (hx, hy, hz)
    resizes the cube being moved (default 3 cm cube); with xy=None there is no support."""
    spec = mujoco.MjSpec.from_file(SCENE)
    if cube_half is not None:
        spec.geom("cube_geom").size = list(cube_half)
    if xy is not None and top_z > 0:
        hx, hy = (half_xy, half_xy) if np.isscalar(half_xy) else half_xy
        h = top_z / 2
        spec.worldbody.add_geom(name="support", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, h],
                                pos=[xy[0], xy[1], h], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                                rgba=[0.2, 0.4, 0.9, 1], friction=[0.8, 0.01, 0.001])
    return spec.compile()


def execute_plan(plan, block_xyz, target_xyz, block_yaw=0.0, latency_steps=0, model=None, data=None, viewer=None,
                 settle_time=1.0, rest_z=None):
    """Run `plan` in sim with the cube at block_xyz. latency_steps = number of PLAN_DT control steps the
    command is delayed. rest_z = cube-centre height it should end at (default: resting on the table; for a
    stack, build the model with scene_with_support and pass the stacked height).
    Returns dict(success, cube_xyz, err_xy, on_table, max_cube_height)."""
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
    on_table = bool(abs(pos[2] - (half if rest_z is None else rest_z)) < 0.004 and up > 0.97)
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
