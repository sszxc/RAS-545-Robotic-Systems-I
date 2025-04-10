import cv2
import time
from pynput import keyboard
import numpy as np
from cobot_ikpy_model import Cobot_ikpy
from collections import deque
from cobot_digital_twin import CobotDigitalTwin
from utils.transform_utils import Transform, Rotation


np.set_printoptions(precision=4, suppress=True)


def just_angle(pose: Transform):
    rotation_matrix = pose.rotation.as_matrix()  # 转换旋转向量为旋转矩阵
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    projected_euler_angles = (
        np.round(euler_angles / 90) * 90
    )  # 将欧拉角投影到90度的倍数
    projected_rotation = Rotation.from_euler(
        "xyz", projected_euler_angles, degrees=True
    )
    just_pose = Transform(projected_rotation, pose.translation)
    return just_pose


if __name__ == "__main__":
    cobot = CobotDigitalTwin(real=True, sim=False)
    cobot_ikpy = Cobot_ikpy()

    # 初始化关节角度
    print("Move to home position.")
    HOME_JOINT_ANGLES = [-120, 30, 120, -60, -90, 0]
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
    input("Press Enter to continue...")

    # 对正角度
    # last_joint_angles = cobot_communicate.get_robot_joint_angle()
    # current_pose = Transform.from_matrix(cobot_ikpy.fk(last_joint_angles))
    # # just_current_pose = just_angle(current_pose)
    # joint_angles = cobot_ikpy.ik(
    #     current_pose.as_matrix()[:3, 3],
    #     target_orientation=[0, 0, -1],
    #     initial_position=last_joint_angles,
    # )
    # cobot_communicate.send_robot_joint_angle(joint_angles, speed=1000)
    # input("Press Enter to continue...")

    last_joint_angles = cobot.real.get_joint_angles()
    current_pose = Transform.from_matrix(cobot_ikpy.fk(last_joint_angles))

    # 显示一张空白图片
    cv2.imshow("blank", np.zeros((100, 100, 3), dtype=np.uint8))

    # 等待按键 控制末端目标位移
    while True:
        key = cv2.waitKey(0)
        print(f"Got key: {key}")
        if key == ord("q"):
            break
        elif key == ord("w"):  # ↑ 控制末端目标位移
            current_pose.translation[1] += 0.01
        elif key == ord("s"):  # ↓ 控制末端目标位移
            current_pose.translation[1] -= 0.01
        elif key == ord("a"):  # ← 控制末端目标位移
            current_pose.translation[0] -= 0.01
        elif key == ord("d"):  # → 控制末端目标位移
            current_pose.translation[0] += 0.01
        elif key == ord("z"):  # up
            current_pose.translation[2] += 0.01
        elif key == ord("x"):  # down
            current_pose.translation[2] -= 0.01
        elif key == ord("p"):  # 打印当前位姿
            print(current_pose.as_matrix()[:3, 3])

        joint_angles = cobot_ikpy.ik(
            current_pose.as_matrix()[:3, 3],
            target_orientation=[0, 0, -1],
            initial_position=last_joint_angles,
        )
        cobot.real.send_joint_angles(joint_angles, speed=1000, is_radian=True)
        last_joint_angles = cobot.real.get_joint_angles()

    cv2.destroyAllWindows()
