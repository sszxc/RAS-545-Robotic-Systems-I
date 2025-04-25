import cv2
import time
import rerun as rr
from rich import print
import numpy as np

import sys
import mujoco
import mujoco.viewer
sys.path.append("lab3&4")
from camera_calibration import RGBCamera
from utils.transform_utils import Transform, Rotation
from utils.rerun_board import RerunBoard
from cobot_digital_twin import CobotDigitalTwin, CobotSim
from cobot_ikpy_model import Cobot_ikpy
from utils.misc_utils import DummyClass
from cal_homography import pixel_to_world_homography
from maze_solver import solve_maze


def get_valid_space(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (workspace)
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the workspace
    mask = np.zeros_like(binary)
    cv2.drawContours(
        mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED
    )
    # 找到 mask 的外接矩形
    x, y, w, h = cv2.boundingRect(mask)
    # 在原图上画出外接矩形
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
    # 裁剪 img 为外接矩形
    img_valid = img[y:y+h, x:x+w]

    # cv2.imshow("img", img)
    # cv2.imshow("img_valid", img_valid)
    # cv2.waitKey(0)
    return img_valid, (x, y, w, h)


def get_path_3D(
    img,
    H_pixel2world,
) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
    valid_pixel_space, (x, y, w, h) = get_valid_space(img)
    try:
        path, solution_img = solve_maze(valid_pixel_space)
    except Exception as e:
        print(f"Error: {e}")
        return [], img
    img[y:y+h, x:x+w] = solution_img
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    points_3D_Fbase = [
        pixel_to_world_homography((px+x, py+y), H_pixel2world) for py, px in path
    ]
    return points_3D_Fbase, img


def plot_plane_in_mujoco(
    H_pixel2world: np.ndarray,
    height: float,
    viewer: mujoco.viewer.Handle,
    pixel_range: tuple[int, int] = (1920, 1080),
    pixel_pts_white: list[tuple[int, int]] | None = None,
):
    """根据 H 矩阵，把像素空间和工作空间可视化在 MuJoCo 中"""
    # 计算平面在mujoco中的位置和方向
    corner_points = np.array(
        [
            [0, 0],
            [pixel_range[0], 0],
            [0, pixel_range[1]],
            [pixel_range[0], pixel_range[1]],
            [pixel_range[0] / 2, pixel_range[1] / 2],
        ]
    )
    world_points = np.array(
        [pixel_to_world_homography((px, py), H_pixel2world) for px, py in corner_points]
    )
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
        pts_3D = [
            pixel_to_world_homography((px, py), H_pixel2world)
            for px, py in pixel_pts_white
        ]
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


if __name__ == "__main__":
    cobot = CobotDigitalTwin(real=True, sim=False)
    cobot_ikpy = Cobot_ikpy()
    # board = RerunBoard(f"Lab_{time.strftime('%m_%d_%H_%M', time.localtime())}", template="3D")
    board = DummyClass()

    # 初始化关节角度
    HOME_JOINT_ANGLES = [-120, 20, 125, -60, -90, 0]
    print("Move to home position.")
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
    cobot.sim.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    input("Press Enter to continue...")

    # 在 mujoco 里检查下 H_matrix
    H_pixel2world = np.array(
        [
            [ 0.00019992, -0.0000265,  -0.48656604],
            [ 0.00000507, -0.00021405, -0.16767739],
            [-0.00002747,  0.0000617,   1.        ]
        ]
    )
    workspace_height = 0.3159
    table_height = workspace_height - 0.18
    pixel_points = [
        (1297, 836),
        (546, 852),
        (535, 84),
        (1303, 84)
    ]
    camera_node = RGBCamera(source=0)

    for i in range(1):
        while True:
            print("Try get one img...")
            img = camera_node.get_img(with_info_overlay=False)
            if img is not None:
                cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}.jpg", img)
                print("Camera is ready.")
                break
        # img = cv2.imread("lab5/example_scene_maze.jpg")
        pt3d_list, img = get_path_3D(img, H_pixel2world)
        board.log("camera", rr.Image(img, color_model="BGR"))
        cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}_noted.jpg", img)

        if not isinstance(cobot.real, DummyClass):
            current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.real.get_joint_angles()))
        else:
            current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.sim.get_joint_angles()))
        final_pts = []
        final_pts_mujoco = []
        for pt in pt3d_list:
            final_pts.append(np.array([pt[0], pt[1], workspace_height]))
            final_pts_mujoco.append(np.array([pt[0], pt[1], table_height]))
        time.sleep(0.1)

    # send to robot
    input("Press Enter to start the tracking...")
    for index, pt in enumerate(final_pts):
        plot_plane_in_mujoco(H_pixel2world, table_height, cobot.sim.viewer, pixel_pts_white=pixel_points)
        joint_angles = cobot_ikpy.ik(
            pt, target_orientation=[0, 0, -1],
            initial_position=cobot.real.get_joint_angles() if not isinstance(cobot.real, DummyClass) else cobot.sim.get_joint_angles()
        )
        input("Press Enter to send to robot...")
        cobot.real.send_joint_angles(joint_angles, speed=1000)
        cobot.sim.send_joint_angles(joint_angles)
        if index == 0:
            time.sleep(2)
        time.sleep(0.2)

    input("Finished tracking!")
    # 回到 home position
    HOME_JOINT_ANGLES = [-120, 20, 125, -60, -90, 0]
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    cobot.sim.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    exit()
