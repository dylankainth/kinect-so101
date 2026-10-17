"""Generate assets/so101/so101_sim.xml from the official so101_new_calib.xml.

Changes vs. upstream: (1) the convex-hull collision meshes of the two gripper jaws are disabled
(MuJoCo hulls would fill the gap between the jaws), (2) box fingertip pads with high friction
(TPU-tip stand-in) are added, (3) a `grasp_point` site is added between the pads.
"""
import re
from pathlib import Path

A = Path(__file__).parent / "assets" / "so101"
s = (A / "so101_new_calib.xml").read_text()
for mesh in ["wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"]:
    pat = re.compile(r'(<geom type="mesh" class="collision"[^>]*mesh="%s")' % mesh)
    assert pat.search(s)
    s = pat.sub(r'\1 contype="0" conaffinity="0"', s)

PAD = 'type="box" class="collision" size="0.0025 0.007 0.0175" friction="1.5 0.02 0.002" condim="4" rgba="0.1 0.1 0.1 1"'
fixed = f'''<geom name="fixed_pad" {PAD} pos="-0.011 0 -0.0825"/>
                <site name="grasp_point" group="3" pos="0.012 0 -0.088" size="0.003"/>
                '''
s = s.replace("<!-- Link moving_jaw_so101_v1 -->", fixed + "<!-- Link moving_jaw_so101_v1 -->", 1)
moving = f'''<geom name="moving_pad" {PAD} pos="-0.0092 -0.0591 0.0188" quat="0.707107 -0.707107 0 0"/>
                  <!-- Part moving_jaw_so101_v1 -->'''
s = s.replace("<!-- Part moving_jaw_so101_v1 -->", moving, 1)
s = s.replace('<mujoco model="so101_new_calib">', '<mujoco model="so101_sim">')
s = s.replace('meshdir="assets"', 'meshdir="assets/so101/assets"')  # resolved relative to sim/scene.xml
(A / "so101_sim.xml").write_text(s)
print("wrote so101_sim.xml")
