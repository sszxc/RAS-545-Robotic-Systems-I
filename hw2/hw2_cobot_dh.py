import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.spatial.transform import Rotation


np.set_printoptions(precision=4, suppress=True)

class myCobot_Pro_600(DHRobot):
    def __init__(self):
        # GUI definition
        L1 = RevoluteDH(offset=np.pi,    d=219.34,  a=0,      alpha=np.pi/2)
        L2 = RevoluteDH(offset=0,        d=0,       a=-250,   alpha=0)
        L3 = RevoluteDH(offset=0,        d=0,       a=-250,   alpha=0)
        L4 = RevoluteDH(offset=np.pi,    d=108,     a=0,      alpha=-np.pi/2)
        L5 = RevoluteDH(offset=0,        d=109.10,  a=0,      alpha=np.pi/2)
        L6 = RevoluteDH(offset=0,        d=75.86,   a=0,      alpha=np.pi)
        # TA definition
        # L1 = RevoluteDH(offset=0,        d=219.34,  a=0,      alpha=np.pi/2)
        # L2 = RevoluteDH(offset=-np.pi/2, d=0,       a=-250,   alpha=0)
        # L3 = RevoluteDH(offset=0,        d=0,       a=-250,   alpha=0)
        # L4 = RevoluteDH(offset=np.pi/2,  d=-108,    a=0,      alpha=-np.pi/2)
        # L5 = RevoluteDH(offset=0,        d=109.10,  a=0,      alpha=np.pi/2)
        # L6 = RevoluteDH(offset=0,        d=-75.86,  a=0,      alpha=np.pi)
        
        super().__init__(
            [L1, L2, L3, L4, L5, L6],
            name="myCobot_Pro_600"
        )


if __name__ == "__main__":
    # create robot instance
    my_robot = myCobot_Pro_600()
    print(my_robot)

    # test different joint angles
    target_q = [-14.98, -115.6, -44.69, -94.74, -19.42, 0.0]
    # target_q = [0.0, -110.4, -42.179, -90.7, 0.43, 0.0]
    target_q = [q/180*np.pi for q in target_q]

    # target_q = [0, np.pi/6, np.pi/6, np.pi/6, -np.pi/4, 0]
    # target_q = np.random.rand(my_robot.n) * 2 * np.pi

    print("\nRobot position at joint angles", target_q, ":")
    print(my_robot.fkine(target_q))

    rot = Rotation.from_matrix(np.array(my_robot.fkine(target_q))[:3, :3 ])
    print(f"End effector orientation (roll, pitch, yaw):{rot.as_euler('xyz', degrees=True)}\n")

    # ================================

    # generate trajectory
    print(f"Start generating trajectory...")
    q0 = np.zeros(my_robot.n)
    q1 = q0.copy()
    q1[0] -= np.pi/4
    traj01 = rtb.jtraj(q0, q1, 20)

    q2 = q1.copy()
    q2[1] += np.pi/6
    traj12 = rtb.jtraj(q1, q2, 20)

    q3 = q2.copy()
    q3[2] += np.pi/6
    traj23 = rtb.jtraj(q2, q3, 20)

    q4 = q3.copy()
    q4[3] += np.pi/4
    traj34 = rtb.jtraj(q3, q4, 20)

    q5 = q4.copy()
    q5[4] -= np.pi/4
    traj45 = rtb.jtraj(q4, q5, 20)

    q6 = q5.copy()
    q6[5] += np.pi/2
    traj56 = rtb.jtraj(q5, q6, 20)

    # visualize motion
    output_filename = "cobot_motion.gif"
    my_robot.plot(np.vstack((traj01.q, traj12.q, traj23.q, traj34.q, traj45.q, traj56.q)), backend='pyplot', movie=output_filename)
    print(f"Trajectory generated as {output_filename}.")
    # plt.show()
