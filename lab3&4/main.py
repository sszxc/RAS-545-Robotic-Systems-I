import re
import os
import cv2
import time
import rerun as rr
from rich import print
import numpy as np
import mujoco.viewer
from mujoco import MjModel, MjData
from dm_control import mjcf

from camera_calibration import RGBCamera, April
from utils.transform_utils import Transform, Rotation
from utils.rerun_board import RerunBoard
from cobot_digital_twin import CobotDigitalTwin, CobotSim
from cobot_ikpy_model import Cobot_ikpy
from utils.misc_utils import DummyClass
from line_detection import detect_line
from cal_homography import pixel_to_world_homography


def get_line_3D(
    img,
    H_pixel2world: np.ndarray,
    is_curve: bool = False,
) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
    try:
        img, interpolated_coords = detect_line(img, is_curve=is_curve, block_imshow=True)
    except Exception as e:
        print(f"Error: {e}")
        return [], img

    points_3D_Fbase = [
        pixel_to_world_homography((px, py), H_pixel2world)
        for px, py in interpolated_coords
    ]
    return points_3D_Fbase, img


def plot_plane_in_mujoco(H_pixel2world: np.ndarray,
                         height: float,
                         viewer: mujoco.viewer.Handle,
                         pixel_range: tuple[int, int] = (1920, 1080),
                         pixel_pts_white: list[tuple[int, int]] | None = None):
    """根据 H 矩阵，把像素空间和工作空间可视化在 MuJoCo 中
    """
    # 计算平面在mujoco中的位置和方向
    corner_points = np.array([
        [0, 0],
        [pixel_range[0], 0],
        [0, pixel_range[1]],
        [pixel_range[0], pixel_range[1]],
        [pixel_range[0] / 2, pixel_range[1] / 2],
    ])
    world_points = np.array([
        pixel_to_world_homography((px, py), H_pixel2world)
        for px, py in corner_points
    ])
    center_point = world_points[-1]
    width = np.linalg.norm(world_points[0] - world_points[1])
    length = np.linalg.norm(world_points[0] - world_points[2])

    # 在mujoco viewer中绘制一个平面
    with viewer.lock():  # 确保线程安全
        viewer.user_scn.ngeom = 0  # 清除之前的几何体
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],  # 几何体索引
            type=mujoco.mjtGeom.mjGEOM_PLANE,  # 平面类型
            size=[width / 2, length / 2, 0],  # 平面的尺寸 [半宽度, 半长度, spacing]
            pos=[center_point[0], center_point[1], height],  # 平面的位置
            mat=np.eye(3).flatten(),  # 方向（单位矩阵）
            rgba=[0.6, 0.5450, 0.4666, 1],  # 颜色
        )
        viewer.user_scn.ngeom += 1
        viewer.sync()

    # 绘制工作空间的四个角
    if pixel_pts_white is not None:
        pts_3D = [pixel_to_world_homography((px, py), H_pixel2world) for px, py in pixel_pts_white]
        pts_3D = [[pt[0], pt[1], height] for pt in pts_3D]
        with viewer.lock():  # 确保线程安全
            for pt in pts_3D:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],  # 几何体索引
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,  # 球体
                    size=[0.01, 0.01, 0.01],  # 尺寸
                    pos=[pt[0], pt[1], height],  # 位置
                    mat=np.eye(3).flatten(),  # 方向（单位矩阵）
                    rgba=[0.9, 0.9, 0.9, 1],  # 颜色
                )
                viewer.user_scn.ngeom += 1
            viewer.sync()

def plot_pts_in_mujoco(pts_3D: list[tuple[float, float, float]],
                       viewer: mujoco.viewer.Handle):
    with viewer.lock():
        for pt in pts_3D:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],  # 几何体索引
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # 球体
                size=[0.005, 0.005, 0.005],  # 尺寸
                pos=[pt[0], pt[1], pt[2]],  # 位置
                mat=np.eye(3).flatten(),  # 方向（单位矩阵）
                rgba=[1.0, 0.6470, 0.3019, 1],  # 颜色
            )
            viewer.user_scn.ngeom += 1
        viewer.sync()

def log_robot_in_rerun(board: RerunBoard, cobot_sim: CobotSim | None = None):
    if cobot_sim is None:
        return
    mj_model = cobot_sim.mj_model
    mj_data = cobot_sim.mj_data
    mj_renderer = mujoco.Renderer(mj_model, 640, 640)
    for name in ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']:
        board.log(f'joint_angle/{name}', rr.Scalar(mj_data.qpos[mj_model.joint(name).id]))
    for name in ['base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6']:
        # TODO geometry 的旋转角很有问题, 直接打 body 没问题
        # geom_id = mj_model.geom(name).id
        # link_pose = Transform(Rotation.from_matrix(mj_data.geom_xmat[geom_id].reshape(3,3)), mj_data.geom_xpos[geom_id])
        body_id = mj_model.body(name).id
        link_pose = Transform(Rotation.from_matrix(mj_data.xmat[body_id].reshape(3,3)), mj_data.xpos[body_id])
        board.log_axes(link_pose, name=name, axis_size=0.05, label=name)

    mj_renderer.update_scene(mj_data, camera="demo-cam")
    board.log("mujoco", rr.Image(mj_renderer.render(), color_model="RGB"))


if __name__ == "__main__":
    cobot = CobotDigitalTwin(real=True, sim=True)
    cobot_ikpy = Cobot_ikpy()
    # board = RerunBoard(f"Lab_{time.strftime('%m_%d_%H_%M', time.localtime())}", template="3D")
    board = DummyClass()

    if cobot.real is not None:
        # 初始化关节角度
        print("Move to home position.")
        HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
        cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
        input("Press Enter to continue...")

    # 初始化关节角度
    # input("Ready to move to home position?")
    HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
    cobot.sim.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    # input("Press Enter to continue...")

    # 在 mujoco 里检查下 H_matrix
    H_pixel2world = np.array(
        [[ 0.00018732,   0.00000716, -0.4880172 ],
         [ 0.0000061  , -0.00020846, -0.15732019],
         [-0.00001028 , -0.00000929 , 1.        ]]
    )
    workspace_height = 0.1859
    table_height = workspace_height - 0.18
    pixel_points = [
        (1342, 901),
        (596, 892),
        (611, 142),
        (1362, 157),
    ]
    camera_node = RGBCamera(
        source=0,
        # intrinsic_path="lab3/calibration_output/intrinsic_03_27_14_42",
    )

    for i in range(1):
        plot_plane_in_mujoco(
            H_pixel2world, table_height, cobot.sim.viewer, pixel_pts_white=pixel_points
        )

        while True:
            print("Try get one img...")
            img = camera_node.get_img(with_info_overlay=False)
            if img is not None:
                cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}.jpg", img)
                print("Camera is ready.")
                break
        # img = cv2.imread("media/example_straight.png")
        # img = cv2.imread("media/example_curve.png")
        pt3d_list, img = get_line_3D(img, H_pixel2world, is_curve=True)
        board.log("camera", rr.Image(img, color_model="BGR"))
        cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}_noted.jpg", img)

        # current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.real.get_joint_angles()))
        current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.sim.get_joint_angles()))
        final_pts = []
        final_pts_mujoco = []
        for pt in pt3d_list:
            # pt[0] -= 0.02
            # pt[1] += 0
            # pt[2] += 0.2
            final_pts.append(np.array([pt[0], pt[1], workspace_height]))  # 标定时的距离
            final_pts_mujoco.append(np.array([pt[0], pt[1], table_height]))
        plot_pts_in_mujoco(final_pts_mujoco, cobot.sim.viewer)
        time.sleep(0.1)

    # send to robot
    input("Press Enter to start the tracking...")
    for index, pt in enumerate(final_pts):
        joint_angles = cobot_ikpy.ik(
            pt, target_orientation=[0, 0, -1], initial_position=cobot.sim.get_joint_angles()
        )
        cobot.sim.send_joint_angles(joint_angles)
        # input("Press Enter to send to real robot...")
        cobot.real.send_joint_angles(joint_angles, speed=1000)
        if index == 0:
            time.sleep(2)
        log_robot_in_rerun(board, cobot.sim)
        time.sleep(0.2)

    # 回到 home position
    HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
    cobot.sim.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    input("Finished tracking!")
    exit()
