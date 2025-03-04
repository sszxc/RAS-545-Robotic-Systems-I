import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, simplify, atan2, sqrt, Eq, solve, Float, Symbol
import roboticstoolbox as rtb
from scipy.optimize import minimize

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab1.lab1_dobot_dh import Dobot


def dobot_ik_analytical(x_target, y_target, z_target):
    # 定义关节角度符号变量
    theta1, theta2 = symbols('theta1 theta2')
    theta3 = -(theta2 + theta1)

    l1 = Symbol('0.15')
    l2 = Symbol('0.09')

    # T01 = Matrix([
    #     [cos(theta0), 0, sin(theta0), 0],
    #     [sin(theta0), 0, -cos(theta0), 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 0, 1]
    # ])
    T12 = Matrix([
        [sin(theta1), cos(theta1), 0, l1*sin(theta1)],
        [-cos(theta1), sin(theta1), 0, -l1*cos(theta1)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T23 = Matrix([
        [-sin(theta2), -cos(theta2), 0, -l1*sin(theta2)],
        [cos(theta2), -sin(theta2), 0, l1*cos(theta2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T34 = Matrix([
        [cos(theta3), 0, sin(theta3), l2*cos(theta3)],
        [sin(theta3), 0, -cos(theta3), l2*sin(theta3)],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    # T45 ignored

    # 计算总变换矩阵
    T_total = T12 * T23 * T34
    T_total = simplify(T_total)  # 简化表达式
    # ⎡1  0  0   0.15⋅sin(θ₂) + 0.15⋅cos(θ₂ + θ₃) + 0.09⎤
    # ⎢0  0  -1     0.15⋅sin(θ₂ + θ₃) - 0.15⋅cos(θ₂)    ⎥
    # ⎢0  1  0                      0                   ⎥
    # ⎣0  0  0                      1                   ⎦

    px = T_total[0, 3]
    py = T_total[1, 3]
    pz = T_total[2, 3]

    # ==========

    theta0 = atan2(y_target, x_target)

    transformed_x_target = x_target**2 + y_target**2
    transformed_y_target = z_target
    transformed_z_target = 0.0

    equations = [
        Eq(px, transformed_x_target),
        Eq(py, transformed_y_target),
        Eq(pz, transformed_z_target)
    ]

    solutions = solve(equations, (theta1, theta2, theta3), dict=True)
    # TODO strange answer here
    # TODO need to add offset for angles here
    return solutions

def dobot_ik_numerical(x_target, y_target, z_target, x0=[0, 0]):
    '''
    x0: initial guess for theta1 and theta2
    '''
    # use end-to-end matrix
    def objective(theta, target):
        theta0 = theta[0]
        theta1 = theta[1] - np.pi/2
        theta2 = theta[2] + np.pi/2
        px = (0.15*np.sin(theta1) + 0.15*np.cos(theta1 + theta2) + 0.09)*np.cos(theta0)
        py = (0.15*np.sin(theta1) + 0.15*np.cos(theta1 + theta2) + 0.09)*np.sin(theta0)
        pz = -0.15*np.sin(theta1 + theta2) + 0.15*np.cos(theta1)
        return (px - target[0])**2 + (py - target[1])**2 + (pz - target[2])**2
    # 添加角度限制：theta1 和 theta2 都在 [-pi, pi] 之间
    bounds = [(-np.pi, np.pi), (-np.pi/4, np.pi/2), (-np.pi/3, np.pi/3)]
    # 使用 scipy.optimize.minimize 求解
    result = minimize(objective, x0, args=([x_target, y_target, z_target]), bounds=bounds)
    # 合并 theta0
    return np.hstack([result.x, -(result.x[2] + result.x[1])])  # + theta3


    # optimize only theta1 and theta2
    # theta0 = np.arctan2(y_target, x_target)
    # transformed_target = [x_target**2 + y_target**2, z_target, 0.0]

    # def objective(theta, target):
    #     theta1 = theta[0] - np.pi/2
    #     theta2 = theta[1] + np.pi/2
    #     theta3 = -(theta2 + theta1)
    #     px = 0.15*np.sin(theta1) + 0.15*np.sin(theta1 + theta2) + 0.09*np.sin(theta3)
    #     py = -0.15*np.cos(theta1) - 0.15*np.cos(theta1 + theta2) - 0.09*np.cos(theta3)
    #     pz = 0
    #     return (px - target[0])**2 + (py - target[1])**2 + (pz - target[2])**2
    # # 添加角度限制：theta1 和 theta2 都在 [-pi, pi] 之间
    # bounds = [(-np.pi/4, np.pi/2), (-np.pi/3, np.pi/3)]
    # # 使用 scipy.optimize.minimize 求解
    # result = minimize(objective, x0, args=(transformed_target), bounds=bounds)
    # # 合并 theta0
    # return np.hstack([theta0, result.x, -(result.x[1] + result.x[0])])


if __name__ == "__main__":
    my_robot = Dobot()
    input_q = [24.0200, 40.2303, 56.4400, 0.0000, -8.1800]
    q = input_q.copy()
    q[2] = -input_q[1] + input_q[2]
    q[3] = -input_q[2]
    target_q = [x/180*np.pi for x in q]

    print("\nRobot position at joint angles", target_q, ":")
    print(my_robot.fkine(target_q))

    x_target, y_target, z_target = my_robot.fkine(target_q).A[0:3, 3]
    # ik_solutions = dobot_ik_analytical(x_target, y_target, z_target)  # TODO not working

    x_target, y_target, z_target = 0.0, 0.2, 0.1
    x0 = [0, 0, 0]
    ik_solutions_list = []  # 存储所有 IK 解
    for x_target in np.linspace(-0.1, 0.1, 21):
        ik_solutions = dobot_ik_numerical(x_target, y_target, z_target, x0=x0)
        ik_solutions_list.append(np.hstack([ik_solutions.copy(), 0]))  # add theta4=0
        x0 = ik_solutions[:3]
        print(f"Target ee position: {x_target:>5.2f}, {y_target:>5.2f}, {z_target:>5.2f}", end="  ")
        print(f"IK solutions: {ik_solutions}")
    
    # # 将 IK 解转换为轨迹
    # q0 = [0] * my_robot.n  # 初始位置
    # qt = rtb.jtraj(q0, ik_solutions_list[0], 50)  # 从初始位置到第一个 IK 解

    # # generate trajectory
    # print(f"Start generating trajectory...")
    # traj_list = []
    # for index in range(len(ik_solutions_list)-1):
    #     traj_list.append(rtb.jtraj(ik_solutions_list[index], ik_solutions_list[index+1], 10).q)
    # traj = np.vstack(traj_list)

    # visualize motion
    # output_filename = "dobot_ik_motion.gif"
    # my_robot.plot(traj, backend='pyplot', movie=output_filename)
    # print(f"Trajectory generated as {output_filename}.")


    import mujoco
    import mujoco.viewer
    from mujoco import MjModel, MjData
    import time
    xml_path = os.path.join(os.path.dirname(__file__), "assets/magician.xml")
    mj_model = MjModel.from_xml_path(xml_path)
    mj_data = MjData(mj_model)
    index = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            mj_data.qpos = ik_solutions_list[index]
            mujoco.mj_step(mj_model, mj_data)
            index += 1
            if index >= len(ik_solutions_list):
                index = 0

            viewer.sync()
            time.sleep(0.2)
