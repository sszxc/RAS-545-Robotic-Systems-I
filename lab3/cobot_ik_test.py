# FK&IK solver, make the real robot and the simulation robot move the same trajectory

import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
from cobot_ikpy_model import Cobot_ikpy
from collections import deque
from cobot_digital_twin import CobotDigitalTwin
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)


def get_square_points(square_vertices=None, num_points_per_edge=20):
    '''define a square path in 3D space
    '''
    if square_vertices is None:
        square_vertices = [  # define the four vertices of the square
            [ 0.4, 0.2, 0.5],
            [ 0.2, 0.2, 0.5],
            [ 0.2, 0.4, 0.5],
            [ 0.4, 0.4, 0.5]
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
    if np.allclose(normal, -z_axis) or np.allclose(normal, z_axis):
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
    cobot = CobotDigitalTwin(real=False, sim=True)
    cobot_ikpy = Cobot_ikpy()

    # # ↓ ====== single point inverse kinematics ======
    # target_position = [0.430591, 0.010565, 0.537452]  # target position
    # # target_position = [0.3, 0.1, 0.3]  # target position
    # target_orientation = [0, 1, 0]  # direction vector
    # initial_position = [-20, -80, 67, 80, 0, 0]
    # initial_position = [q/180*np.pi for q in initial_position]
    # joint_angles = cobot_ikpy.ik(target_position, target_orientation, initial_position=initial_position)

    # # print the result
    # print("calculated joint angles (radian):")
    # for i, angle in enumerate(joint_angles[:]):  # skip the first and last fixed joints
    #     print(f"joint{i}: {angle/np.pi*180:.4f}°")

    # # verify forward kinematics
    # fk_position = cobot_ikpy.fk(joint_angles)[:3, 3]
    # print("\nforward kinematics verification:")
    # print(f"target position: {np.round(target_position, 4)}")
    # print(f"calculated position: {np.round(fk_position, 4)}")

    # # visualize the robot
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")
    # cobot_ikpy.chain.plot(joint_angles, ax, target=target_position)
    # plt.show()
    # exit()

    # ↓ ====== continuous trajectory inverse kinematics ======
    # target_list = get_square_points()
    target_list = get_circle_points(center=[0.4, 0.1, 0.5], radius=0.1, normal=[0, -1, 1])
    # target_list = get_circle_points(center=[0, 0, 0.6], radius=0.3, normal=[0, 0, 1])

    last_joint_angles = [0, 0, 0, 0, 0, 0]

    # ====== visualize motion using mujoco ======
    trail_length = 80  # length of trail (number of history positions)
    trail_opacity = 0.8  # initial opacity of trail
    trail_history = deque(maxlen=trail_length)  # queue for storing history positions

    while cobot.sim.viewer.is_running():
        for index, target in enumerate(target_list):
            joint_angles = cobot_ikpy.ik(target, target_orientation=[0, 0, -1], initial_position=last_joint_angles)
            last_joint_angles = joint_angles
            # print(f"joint angles: {joint_angles}")
            fk_position = cobot_ikpy.fk(joint_angles)[:3, 3]
            # print(f"target position: {np.round(target, 4)}")
            # print(f"calculated position: {np.round(fk_position, 4)}")

            cobot.sim.send_joint_angles(joint_angles)
            cobot.real.send_joint_angles(joint_angles, speed=1000)
            time.sleep(0.1)

            link6_pos = cobot.sim.mj_data.geom_xpos[cobot.sim.mj_model.geom("link6_geom").id]
            # trail_history.appendleft(link6_pos)
            print(f"Tracking error: link6_pos - fk_position = {np.round(link6_pos - fk_position, 4)}")

            # ====== draw trail ======
            trail_history.appendleft(target_list[index])
            # update geometry in user_scn
            with cobot.sim.viewer.lock():  # ensure thread safety
                cobot.sim.viewer.user_scn.ngeom = 0  # clear previous geometry
                for i, pos in enumerate(trail_history):
                    # fade alpha with time, simulate trail effect
                    fade_alpha = trail_opacity * (1 - i / trail_length)
                    mujoco.mjv_initGeom(
                        cobot.sim.viewer.user_scn.geoms[i],  # index of geometry
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # use sphere to represent trajectory point
                        size=[0.004, 0, 0],  # sphere size
                        pos=[
                            pos[0],
                            pos[1],
                            pos[2] - 0.12,    # position (compensate for the height of the base)
                        ],
                        mat=np.eye(3).flatten(),  # orientation (unit matrix)
                        rgba=np.array(
                            [0.83, 0.98, 0.98, fade_alpha]
                        ),  # color and transparency
                    )
                    cobot.sim.viewer.user_scn.ngeom += 1
