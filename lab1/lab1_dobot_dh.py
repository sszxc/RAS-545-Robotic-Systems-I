import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.spatial.transform import Rotation


np.set_printoptions(precision=4, suppress=True)

class Dobot(DHRobot):
    def __init__(self):
        # ! q[3] = -(q[2] + q[1])  # mechinal constraint
        L1 = RevoluteDH(offset=0,        d=0,    a=0,      alpha=-np.pi/2)
        L2 = RevoluteDH(offset=-np.pi/2, d=0,    a=0.15,   alpha=0)
        L3 = RevoluteDH(offset=np.pi/2,  d=0,    a=0.15,   alpha=0)
        L4 = RevoluteDH(offset=0,        d=0,    a=0.09,   alpha=np.pi/2)
        L5 = RevoluteDH(offset=0,        d=0,    a=0,      alpha=0)

        super().__init__(
            [L1, L2, L3, L4, L5],
            name="Dobot"
        )


if __name__ == "__main__":
    # create robot instance
    my_robot = Dobot()
    print(my_robot)

    # test different joint angles
    # target_q = [0, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # 各关节角度
    # target_q = [0, 0, 0, 0, 0]
    q = [24.0200, 40.2303, 56.4400, 0.0000, -8.1800]
    q[3] = -(q[2] + q[1])
    target_q = [x/180*np.pi for x in q]

    print("\nRobot position at joint angles", target_q, ":")
    print(my_robot.fkine(target_q))

    rot = Rotation.from_matrix(np.array(my_robot.fkine(target_q))[:3, :3 ])
    print(f"End effector orientation (roll, pitch, yaw):{rot.as_euler('xyz', degrees=True)}\n")

    # define start and end joint position
    q0 = [0, 0, 0, 0, 0]
    print(f"Start generating trajectory from\n{q0}\nto\n{target_q}...\n")

    # generate trajectory
    qt = rtb.jtraj(q0, target_q, 50)

    # visualize motion
    output_filename = "dobot_motion.gif"
    my_robot.plot(qt.q, backend='pyplot', movie=output_filename)
    print(f"Trajectory generated as {output_filename}.")
    # plt.show()
