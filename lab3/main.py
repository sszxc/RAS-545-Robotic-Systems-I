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
from cobot_digital_twin import CobotDigitalTwin
from cobot_ikpy_model import Cobot_ikpy
from line_detection import detect_line
from cal_homography import pixel_to_world_homography


def get_camera_to_line(
    img, H_pixel2world: np.ndarray
) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
    img, interpolated_coords = detect_line(img)

    points_3D_Fbase = [
        pixel_to_world_homography((px, py), H_pixel2world)
        for px, py in interpolated_coords
    ]

    # points_3D_Fbase_list = []
    # for index, pt3d in enumerate(points_3D):
    #     points_3D_Fbase = (camera_pose.as_matrix() @ np.hstack((pt3d, [1])))[:3]
    #     board.log(f"world/point_{index}",
    #             rr.Points3D(positions=points_3D_Fbase, # colors=finger_config["color"][_ftp_index],
    #                         radii=0.003,
    #             ))
    #     points_3D_Fbase_list.append(points_3D_Fbase)
    return points_3D_Fbase, img


def plot_plane_in_mujoco(H_pixel2world: np.ndarray,
                         height: float,
                         viewer: mujoco.viewer.Handle,
                         pixel_range: tuple[int, int] = (1680, 1050)):
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
        
        # 初始化一个平面几何体
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],  # 几何体索引
            type=mujoco.mjtGeom.mjGEOM_PLANE,  # 平面类型
            size=[width / 2, length / 2, 0.01],  # 平面的尺寸 [半宽度, 半长度, spacing]
            pos=[center_point[0], center_point[1], height],  # 平面的位置
            mat=np.eye(3).flatten(),  # 方向（单位矩阵）
            rgba=[0.9, 0.9, 0.9, 1],  # 颜色 (浅灰色)
        )
        viewer.user_scn.ngeom += 1
        viewer.sync()


if __name__ == "__main__":
    cobot = CobotDigitalTwin(real=False, sim=True)
    cobot_ikpy = Cobot_ikpy()

    # 初始化关节角度
    # input("Ready to move to home position?")
    HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
    cobot.sim.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    # input("Press Enter to continue...")

    # 在 mujoco 里检查下 H_matrix
    H_pixel2world = np.array(
        [
            [2.10692651e-04, 4.04400481e-06, -5.27370580e-02],
            [-9.48085900e-06, 2.96163977e-04, -3.31884242e-02],
            [-9.91626067e-05, 8.25412302e-05, 1.00000000e00],
        ]
    )
    work_space_height = 0.1859
    plot_plane_in_mujoco(H_pixel2world, work_space_height, cobot.sim.viewer)

    camera_node = RGBCamera(
        source=0,
        intrinsic_path="lab3/calibration_output/intrinsic_03_27_14_42",
    )
    while True:
        print("Try get one img...")
        img = camera_node.get_img(with_info_overlay=False)
        if img is not None:
            print("Camera is ready.")
            break

    pt3d_list, img = get_camera_to_line(img, H_pixel2world)

    current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.real.get_joint_angles()))
    final_pts = []
    for pt in pt3d_list:
        # pt[0] -= 0.02
        # pt[1] += 0
        # pt[2] += 0.2
        pt_xyz = [pt[0], pt[1], 0.1859]  # 0.1859 是标定时的距离
        final_pts.append(pt_xyz)
        print(f"pt: {pt_xyz}")

    # send to robot
    input("Press Enter to start the tracking...")
    for pt in final_pts:
        joint_angles = cobot_ikpy.ik(
            pt, target_orientation=[0, 0, -1], initial_position=cobot.real.get_joint_angles()
        )
        breakpoint()
        cobot.real.send_joint_angles(joint_angles, speed=500)

    # 回到 home position
    HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
    exit()

    for name in ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']:
        board.log(f'joint_angle/{name}', rr.Scalar(mj_data.qpos[mj_model.joint(name).id]))
    for name in ['base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'ee_tag']:
        # TODO geometry 的旋转角很有问题, 直接打 body 没问题
        # geom_id = mj_model.geom(name).id
        # link_pose = Transform(Rotation.from_matrix(mj_data.geom_xmat[geom_id].reshape(3,3)), mj_data.geom_xpos[geom_id])
        body_id = mj_model.body(name).id
        link_pose = Transform(Rotation.from_matrix(mj_data.xmat[body_id].reshape(3,3)), mj_data.xpos[body_id])
        board.log_axes(link_pose, name=name, axis_size=0.05, label=name)

    mj_renderer.update_scene(mj_data, camera="demo-cam")
    board.log("mujoco", rr.Image(mj_renderer.render(), color_model="RGB"))
    input("finished log robot")

    # # ===== find table depth & line=====
    # camera_to_tag, img = get_camera_to_tag(img, id=1)
    # if camera_to_tag is not None:
    #     table_depth = camera_to_tag.translation[2]
    #     board.log_axes(base_to_tag, name='tag', label='tag')
    # ↓ fixed, for debug
    table_depth = 0.36830
    pt3d_list, img = get_camera_to_line(img, table_depth, camera_pose)
    board.log("camera", rr.Image(img, color_model="BGR"))

    final_pts = []
    for pt in pt3d_list:
        pt[0] -= 0.02
        pt[1] += 0
        pt[2] += 0.2
        final_pts.append(pt)
