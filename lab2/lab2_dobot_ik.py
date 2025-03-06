import time
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
import numpy as np
import sympy as sp
from sympy import simplify
import roboticstoolbox as rtb
from scipy.optimize import minimize
from collections import deque

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
            [-0.1, 0.2, 0.0],
            [-0.1, 0.2, 0.2],
            [ 0.1, 0.2, 0.2],
            [ 0.1, 0.2, 0.0]
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


def get_circle_points(center=[0, -0.15, 0.2], radius=0.08, num_points=100, normal=[0, 0, 1]):
    '''define a circle path in 3D space
    Args:
        center: circle center position [x, y, z]
        radius: circle radius
        num_points: number of sampling points
        normal: circle plane normal vector, default is [0, 0, 1] for xy plane
    '''
    # normalize the input vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # calculate the rotation matrix from z-axis to the target normal vector
    z_axis = np.array([0, 0, 1])
    # if the normal vector is close to the opposite direction of z-axis, use x-axis as the rotation axis
    if np.allclose(normal, -z_axis):
        rotation_axis = np.array([1, 0, 0])
    else:
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # calculate the rotation angle
    cos_theta = np.dot(z_axis, normal)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # build the rotation matrix (Rodrigues' formula)
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - cos_theta) * np.dot(K, K)
    
    # generate points on the circle and rotate
    target_list = []
    for i in range(num_points):
        alpha = 2 * np.pi * i / num_points
        # generate points on the xy plane
        point = np.array([
            radius * np.cos(alpha),
            radius * np.sin(alpha),
            0
        ])
        # rotate to the target plane
        rotated_point = np.dot(R, point)
        # translate to the target position
        final_point = rotated_point + center
        target_list.append(final_point.tolist())
    
    return target_list


if __name__ == "__main__":
    my_robot = Dobot()
    
    # ↓ ====== analytical solution, not working ======
    # x_target, y_target, z_target = 0.0, 0.2, 0.1
    # ik_solutions = dobot_ik_analytical(x_target, y_target, z_target)
    
    # ↓ ====== numerical solution, working ======
    x_target, y_target, z_target = 0.248, 0.111, -0.008
    ik_solution = dobot_ik_numerical(x_target, y_target, z_target, x0=[0, 0, 0, 0, 0])
    ik_solution = [angle * 180 / np.pi for angle in ik_solution]
    print(f"Target ee position: {x_target:>5.2f}, {y_target:>5.2f}, {z_target:>5.2f}")
    ik_solution[0] -= 90  # mujoco base angle offset
    print(f"IK solutions: {ik_solution}")


    # ↓ ====== continuous trajectory ======
    target_list = get_square_points()
    # target_list = get_circle_points(center=[0, -0.2, 0.15], normal=[0, -1, 1])
    x0 = [0, 0, 0, 0, 0]  # initial guess
    ik_solutions_list = []
    for target in target_list:
        ik_solution = dobot_ik_numerical(target[0], target[1], target[2], x0=x0)
        ik_solutions_list.append(ik_solution)
        x0 = ik_solution  # use this solution as the initial value for the next iteration
        print(f"Target ee position: {target[0]:>5.2f}, {target[1]:>5.2f}, {target[2]:>5.2f}", end="  ")
        print(f"IK solutions: {ik_solution}")

    # ====== generate trajectory using rtbox ======
    # print(f"Start generating trajectory...")
    # traj_list = []
    # for index in range(len(ik_solutions_list)-1):
    #     traj_list.append(rtb.jtraj(ik_solutions_list[index], ik_solutions_list[index+1], 2).q)
    # traj = np.vstack(traj_list)

    # # visualize motion
    # output_filename = "dobot_ik_motion.gif"
    # my_robot.plot(traj, backend='pyplot', movie=output_filename)
    # print(f"Trajectory generated as {output_filename}.")

    # ====== visualize motion using mujoco ======
    trail_length = 80  # length of trail (number of history positions)
    trail_opacity = 0.8  # initial opacity of trail
    trail_history = deque(maxlen=trail_length)  # queue for storing history positions

    xml_path = os.path.join(os.path.dirname(__file__), "assets/magician.xml")
    mj_model = MjModel.from_xml_path(xml_path)
    mj_data = MjData(mj_model)
    index = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            mj_data.qpos[:5] = ik_solutions_list[index]
            # ====== isolated indicator ======
            # mj_data.qpos[-3:] = target_list[index]  

            # ====== draw trail ======
            trail_history.appendleft(target_list[index])
            # update geometry in user_scn
            with viewer.lock():  # ensure thread safety
                viewer.user_scn.ngeom = 0  # clear previous geometry
                for i, pos in enumerate(trail_history):
                    # fade alpha with time, simulate trail effect
                    fade_alpha = trail_opacity * (1 - i / trail_length)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],  # index of geometry
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # use sphere to represent trajectory point
                        size=[0.004, 0, 0],  # sphere size
                        pos=[pos[0], pos[1], pos[2]+0.06],  # position (compensate for the height of the base)
                        mat=np.eye(3).flatten(),  # orientation (unit matrix)
                        rgba=np.array([0.83, 0.98, 0.98, fade_alpha])  # color and transparency
                    )
                    viewer.user_scn.ngeom += 1

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            time.sleep(0.1)
            index += 1
            if index >= len(ik_solutions_list):
                index = 0
