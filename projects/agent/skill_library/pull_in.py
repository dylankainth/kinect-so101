
def run(arm, x_from=0.335, x_to=0.25, y=0.115, z=0.024, speed=0.04):
    arm.jaws(0.0)                                      # closed jaws: a flat plate
    arm.move(x_from, y, 0.08, tip="fixed")             # fixed pad above the start, beyond the box
    arm.move(x_from, y, z, tip="fixed", speed=0.05)    # down behind the box
    arm.move(x_to, y, z, tip="fixed", speed=speed)     # pull toward the robot with the fixed pad's outer face
    arm.move(x_to, y, 0.08, tip="fixed")               # straight up
