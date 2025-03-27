import time
import numpy as np
from ikpy import chain
from ikpy.link import OriginLink, URDFLink
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
from cobot_ikpy_model import Cobot_ikpy
from collections import deque
from cobot_communicate import send_tcp_packet

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
    cobot_ikpy = Cobot_ikpy()

    # # # ↓ ====== 单点逆解 ======
    # target_position = [0.430591, 0.010565, 0.537452]  # 目标位置
    # # target_position = [0.3, 0.1, 0.3]  # 目标位置
    # target_orientation = [0, 1, 0]  # 方向向量
    # initial_position = [-20, -80, 67, 80, 0, 0]
    # joint_angles = cobot_ikpy.ik(target_position, target_orientation, initial_position=[q/180*np.pi for q in initial_position])

    # # 打印结果
    # print("计算得到的关节角度（弧度）:")
    # for i, angle in enumerate(joint_angles[:]):  # 跳过第一个和最后一个固定关节
    #     print(f"关节{i}: {angle/np.pi*180:.4f}°")

    # # 验证正运动学
    # fk_position = cobot_ikpy.fk(joint_angles)[:3, 3]
    # print("\n正向运动学验证:")
    # print(f"目标位置: {np.round(target_position, 4)}")
    # print(f"计算位置: {np.round(fk_position, 4)}")

    # # 可视化机械臂
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")
    # cobot_ikpy.chain.plot(joint_angles, ax, target=target_position)
    # plt.show()
    # exit()

    # ↓ ====== 连续轨迹逆解 ======
    mj_model = MjModel.from_xml_path("model/my_cobot.xml")
    mj_data = MjData(mj_model)
    # target_list = get_square_points()
    target_list = get_circle_points(center=[0.4, 0.1, 0.5], radius=0.1, normal=[0, -1, 1])
    # target_list = get_circle_points(center=[0, 0, 0.6], radius=0.3, normal=[0, 0, 1])

    last_joint_angles = [0, 0, 0, 0, 0, 0]

    # ====== visualize motion using mujoco ======
    trail_length = 80  # length of trail (number of history positions)
    trail_opacity = 0.8  # initial opacity of trail
    trail_history = deque(maxlen=trail_length)  # queue for storing history positions

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            for index, target in enumerate(target_list):
                joint_angles = cobot_ikpy.ik(target, target_orientation=[0, 0, -1], initial_position=last_joint_angles)
                last_joint_angles = joint_angles
                # print(f"关节角度: {joint_angles}")
                fk_position = cobot_ikpy.fk(joint_angles)[:3, 3]
                # print(f"目标位置: {np.round(target, 4)}")
                # print(f"计算位置: {np.round(fk_position, 4)}")

                SERVER_IP = "192.168.1.159"
                SERVER_PORT = 5001
                joint_angles_shifted = [j/np.pi*180 for j in joint_angles]
                joint_angles_shifted[1] -= 90
                joint_angles_shifted[3] -= 90
                MESSAGE = (
                    f"set_angles({','.join([str(j) for j in joint_angles_shifted])}, 1000)"
                )
                # MESSAGE = f"set_angles(0, -90, 0, -90, 0, 0, 500)"
                send_tcp_packet(SERVER_IP, SERVER_PORT, MESSAGE)
                time.sleep(0.2)

                mj_data.ctrl[:] = joint_angles
                for _ in range(50):
                    mujoco.mj_step(mj_model, mj_data)
                    viewer.sync()

                link6_pos = mj_data.geom_xpos[mj_model.geom("link6_geom").id]
                # trail_history.appendleft(link6_pos)
                print(f"link6_pos - fk_position: {np.round(link6_pos - fk_position, 4)}")

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
                            pos=[
                                pos[0],
                                pos[1],
                                pos[2] - 0.12,
                            ],  # position (compensate for the height of the base)
                            mat=np.eye(3).flatten(),  # orientation (unit matrix)
                            rgba=np.array(
                                [0.83, 0.98, 0.98, fade_alpha]
                            ),  # color and transparency
                        )
                        viewer.user_scn.ngeom += 1
