import time
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
import numpy as np
import sympy as sp
from sympy import simplify
import roboticstoolbox as rtb
from scipy.optimize import minimize

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab1.lab1_dobot_dh import Dobot


def dobot_ik_analytical(x_target, y_target, z_target):
    # define symbols
    theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')

    theta4 = -(theta3 + theta2)

    # define each joint's homogeneous transformation matrix
    T01 = sp.Matrix([
        [sp.cos(theta1), 0, sp.sin(theta1), 0],
        [sp.sin(theta1), 0, -sp.cos(theta1), 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    T12 = sp.Matrix([
        [sp.cos(theta2 - sp.pi/2), -sp.sin(theta2 - sp.pi/2), 0, 0.15 * sp.cos(theta2 - sp.pi/2)],
        [sp.sin(theta2 - sp.pi/2), sp.cos(theta2 - sp.pi/2), 0, 0.15 * sp.sin(theta2 - sp.pi/2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T23 = sp.Matrix([
        [sp.cos(theta3 + sp.pi/2), -sp.sin(theta3 + sp.pi/2), 0, 0.15 * sp.cos(theta3 + sp.pi/2)],
        [sp.sin(theta3 + sp.pi/2), sp.cos(theta3 + sp.pi/2), 0, 0.15 * sp.sin(theta3 + sp.pi/2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T34 = sp.Matrix([
        [sp.cos(theta4), 0, sp.sin(theta4), 0.09 * sp.cos(theta4)],
        [sp.sin(theta4), 0, -sp.cos(theta4), 0.09 * sp.sin(theta4)],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    T45 = sp.Matrix([
        [sp.cos(theta5), -sp.sin(theta5), 0, 0],
        [sp.sin(theta5), sp.cos(theta5), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # calculate total transformation matrix
    T05 = T01 * T12 * T23 * T34  # * T45  # temporary ignore theta5
    
    T_total = simplify(T05)  # simplify expression
    sp.pprint(T_total)

    # extract end-effector pose
    position = T_total[:3, 3]  # position part
    orientation = T_total[:3, :3]  # orientation part

    # build equations
    target_position = sp.Matrix([x_target, y_target, z_target])
    
    # solve theta1
    theta1_solution = sp.atan2(target_position[1], target_position[0])
    T05_with_theta1 = T05.subs(theta1, theta1_solution)
    T05_with_theta1 = simplify(T05_with_theta1)  # simplify expression
    position_with_theta1 = T05_with_theta1[:3, 3]
    sp.pprint(T05_with_theta1[:3, 3])

    # solve theta2 and theta3
    solutions = sp.solve(position_with_theta1 - target_position, (theta2, theta3), dict=True)

    # print results
    sp.pprint(solutions)
    # convert symbolic solution to float
    theta1_val = float(theta1_solution.evalf())
    theta2_val = float(solutions[0][theta2].evalf()) 
    theta3_val = float(solutions[0][theta3].evalf())
    theta4_val = float(-(solutions[0][theta2] + solutions[0][theta3]).evalf())
    
    return np.array([theta1_val, theta2_val, theta3_val, theta4_val, 0.0])


def dobot_ik_numerical(x_target, y_target, z_target, x0=[0, 0, 0, 0, 0]):
    '''
    x0: initial guess for theta
    '''
    # use end-to-end matrix
    def objective(theta, target):
        theta1 = theta[0]
        theta2 = theta[1]
        theta3 = theta[2]
        # ⎡cos(θ₁ - θ₅)  sin(θ₁ - θ₅)   0  (0.15⋅sin(θ₂) + 0.15⋅cos(θ₂ + θ₃) + 0.09)⋅cos(θ₁)⎤
        # ⎢sin(θ₁ - θ₅)  -cos(θ₁ - θ₅)  0  (0.15⋅sin(θ₂) + 0.15⋅cos(θ₂ + θ₃) + 0.09)⋅sin(θ₁)⎥
        # ⎢     0              0        1          -0.15⋅sin(θ₂ + θ₃) + 0.15⋅cos(θ₂)        ⎥
        # ⎣     0              0        0                          1                        ⎦
        px = (0.15*np.sin(theta2) + 0.15*np.cos(theta2 + theta3) + 0.09)*np.cos(theta1)
        py = (0.15*np.sin(theta2) + 0.15*np.cos(theta2 + theta3) + 0.09)*np.sin(theta1)
        pz = -0.15*np.sin(theta2 + theta3) + 0.15*np.cos(theta2)
        return (px - target[0])**2 + (py - target[1])**2 + (pz - target[2])**2

    # add joint angle limits
    bounds = [(-np.pi, np.pi), (-np.pi/4, np.pi/2), (-np.pi/3, np.pi/3)]
    # solve optimization problem
    result = minimize(objective, x0[:3], args=([x_target, y_target, z_target]), bounds=bounds)
    # print optimization error
    print(f"Optimization error: {result.fun}")
    # post-process, merge theta 4, theta 5
    result.x[0] += np.pi/2  # mujoco base angle offset
    return np.hstack([result.x, -(result.x[2] + result.x[1]), 0])


def get_square_points(square_vertices=None, num_points_per_edge=20):
    '''define a square path in 3D space
    '''
    if square_vertices is None:
        square_vertices = [  # define the four vertices of the square
            [ 0.0, 0.2, 0.0],
            [ 0.0, 0.2, 0.2],
            [ 0.2, 0.2, 0.2],
            [ 0.2, 0.2, 0.0]
        ]
    target_list = []
    
    for i in range(4):  # interpolate each edge of the square
        start = square_vertices[i]
        end = square_vertices[(i+1)%4]  # loop to the first point
        # interpolate each edge linearly
        for t in range(num_points_per_edge):
            alpha = t / num_points_per_edge
            point = [
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha, 
                start[2] + (end[2] - start[2]) * alpha
            ]
            target_list.append(point)
    return target_list


if __name__ == "__main__":
    my_robot = Dobot()
    
    # ↓ analytical solution, not working
    # x_target, y_target, z_target = 0.0, 0.2, 0.1
    # ik_solutions = dobot_ik_analytical(x_target, y_target, z_target)
    
    # ↓ numerical solution, working
    target_list = get_square_points()
    x0 = [0, 0, 0, 0, 0]  # initial guess
    ik_solutions_list = []
    for target in target_list:
        ik_solution = dobot_ik_numerical(target[0], target[1], target[2], x0=x0)
        ik_solutions_list.append(ik_solution)
        x0 = ik_solution  # 使用本次解作为下一次的初始值
        print(f"Target ee position: {target[0]:>5.2f}, {target[1]:>5.2f}, {target[2]:>5.2f}", end="  ")
        print(f"IK solutions: {ik_solution}")

    # # generate trajectory using rtbox
    # print(f"Start generating trajectory...")
    # traj_list = []
    # for index in range(len(ik_solutions_list)-1):
    #     traj_list.append(rtb.jtraj(ik_solutions_list[index], ik_solutions_list[index+1], 2).q)
    # traj = np.vstack(traj_list)

    # # visualize motion
    # output_filename = "dobot_ik_motion.gif"
    # my_robot.plot(traj, backend='pyplot', movie=output_filename)
    # print(f"Trajectory generated as {output_filename}.")

    # visualize motion using mujoco
    xml_path = os.path.join(os.path.dirname(__file__), "assets/magician.xml")
    mj_model = MjModel.from_xml_path(xml_path)
    mj_data = MjData(mj_model)
    index = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            mj_data.qpos[:5] = ik_solutions_list[index]
            mj_data.qpos[-3:] = target_list[index]
            mujoco.mj_step(mj_model, mj_data)
            index += 1
            if index >= len(ik_solutions_list):
                index = 0
            viewer.sync()
            time.sleep(0.1)
