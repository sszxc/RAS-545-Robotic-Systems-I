import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH

class Dobot(DHRobot):
    def __init__(self):
        # ! q[3] = -(q[2] + q[1])  # mechinal constraint
        L1 = RevoluteDH(offset=0,       d=0,    a=0,      alpha=-np.pi/2)
        L2 = RevoluteDH(offset=np.pi/2, d=0,    a=-0.15,  alpha=0)
        L3 = RevoluteDH(offset=np.pi/2, d=0,    a=-0.15,  alpha=0)
        L4 = RevoluteDH(offset=0,       d=0,    a=-0.09,  alpha=np.pi/2)
        L5 = RevoluteDH(offset=0,       d=0,    a=0,      alpha=0)

        super().__init__(
            [L1, L2, L3, L4, L5],
            name="Dobot"
        )


if __name__ == "__main__":
    # create robot instance
    my_robot = Dobot()
    print(my_robot)

    # test different joint angles
    q = [0, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # 各关节角度

    print("\nRobot position at joint angles", q, ":")
    print(my_robot.fkine(q))

    # define start and end joint position
    q1 = [0, 0, 0, 0, 0]
    q2 = [np.pi/12, np.pi/8, np.pi/8, -np.pi/4, np.pi]
    print(f"Start generating trajectory from\n{q1}\nto\n{q2}.")

    # generate trajectory
    qt = rtb.jtraj(q1, q2, 50)

    # visualize motion
    output_filename = "dobot_motion.gif"
    my_robot.plot(qt.q, backend='pyplot', movie=output_filename)
    print(f"Trajectory generated as {output_filename}.")
    # plt.show()
