import cv2
import time
import rerun as rr
from rich import print
import numpy as np

import sys
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
    cv2.imshow("img", img)
    cv2.waitKey(0)

    points_3D_Fbase = [
        pixel_to_world_homography((px+x, py+y), H_pixel2world) for py, px in path
    ]
    return points_3D_Fbase, img


if __name__ == "__main__":
    # camera_node = RGBCamera(source=0)
    # while True:
    #     print("Try get one img...")
    #     img = camera_node.get_img(with_info_overlay=False)
    #     if img is not None:
    #         cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}.jpg", img)
    #         print("Camera is ready.")
    #         break

    # # 在 mujoco 里检查下 H_matrix
    # H_pixel2world = np.array(
    #     [
    #         [ 0.00019992, -0.0000265,  -0.48656604],
    #         [ 0.00000507, -0.00021405, -0.16767739],
    #         [-0.00002747,  0.0000617,   1.        ]
    #     ]
    # )
    # pt3d_list, img = get_path_3D(img, H_pixel2world)
    # cv2.imwrite(f"debug/img_{time.strftime('%m_%d_%H_%M', time.localtime())}_noted.jpg", img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    cobot = CobotDigitalTwin(real=True, sim=False)
    cobot_ikpy = Cobot_ikpy()
    # board = RerunBoard(f"Lab_{time.strftime('%m_%d_%H_%M', time.localtime())}", template="3D")
    board = DummyClass()

    HOME_JOINT_ANGLES = [-120, 20, 125, -60, -90, 0]
    if cobot.real is not None:
        # 初始化关节角度
        print("Move to home position.")
        cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
        input("Press Enter to continue...")

    # 初始化关节角度
    # input("Ready to move to home position?")
    # input("Press Enter to continue...")

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

        current_pose = Transform.from_matrix(cobot_ikpy.fk(cobot.real.get_joint_angles()))
        final_pts = []
        final_pts_mujoco = []
        for pt in pt3d_list:
            # pt[0] -= 0.02
            # pt[1] += 0
            # pt[2] += 0.2
            final_pts.append(np.array([pt[0], pt[1], workspace_height]))  # 标定时的距离
            final_pts_mujoco.append(np.array([pt[0], pt[1], table_height]))
        time.sleep(0.1)

    # send to robot
    input("Press Enter to start the tracking...")
    for index, pt in enumerate(final_pts):
        joint_angles = cobot_ikpy.ik(
            pt, target_orientation=[0, 0, -1],
            initial_position=cobot.real.get_joint_angles()
        )
        input("Press Enter to send to real robot...")
        cobot.real.send_joint_angles(joint_angles, speed=1000)
        if index == 0:
            time.sleep(2)
        time.sleep(0.2)

    input("Finished tracking!")
    # 回到 home position
    HOME_JOINT_ANGLES = [-120, 20, 125, -60, -90, 0]
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, is_radian=False)
    exit()
