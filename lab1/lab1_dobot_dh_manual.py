import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=4, suppress=True)

class Dobot:
    def __init__(self):
        # DH parameters [theta, d, a, alpha]
        self.dh_params = [
            [0, 0, 0, -np.pi/2],
            [-np.pi/2, 0, 0.15, 0],
            [np.pi/2, 0, 0.15, 0],
            [0, 0, 0.09, np.pi/2],
            [0, 0, 0, 0]
        ]
        self.n_joints = len(self.dh_params)
        
    def transform_matrix(self, theta, d, a, alpha):
        """
        Calculate the DH transformation matrix for a single joint
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1]
        ])
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics
        Parameters:
            joint_angles: list of joint angles
        Returns:
            total transformation matrix from base to end effector
        """
        T_total = np.eye(4)
        for i in range(self.n_joints):
            theta = joint_angles[i] + self.dh_params[i][0]  # use input joint angles
            d = self.dh_params[i][1]
            a = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            
            # calculate transformation matrix for current joint
            Ti = self.transform_matrix(theta, d, a, alpha)
            
            # update total transformation matrix
            T_total = np.dot(T_total, Ti)
            
        return T_total


if __name__ == "__main__":
    robot = Dobot()
    
    # test_angles = [0, np.pi/4, np.pi/4, np.pi/4, np.pi/4]
    # test_angles = np.random.rand(robot.n_joints) * 2 * np.pi
    
    q = [24.0200, 40.2303, 56.4400, 0.0000, -8.1800]
    q[3] = -(q[2] + q[1])
    test_angles = [x/180*np.pi for x in q]

    T = robot.forward_kinematics(test_angles)
    rot = Rotation.from_matrix(np.array(T[:3, :3]))

    print(f"Test joint angles (radians):\n{test_angles}\n")
    print(f"Final transformation matrix:\n{T}\n")
    print(f"End effector position (x, y, z):\n{T[0:3, 3]}")
    print(f"End effector orientation (roll, pitch, yaw):\n{rot.as_euler('xyz', degrees=True)}")
