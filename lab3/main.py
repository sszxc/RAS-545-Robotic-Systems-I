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
from line_detection import detect_line
from utils.transform_utils import Transform, Rotation
from utils.rerun_board import RerunBoard
from cobot_communicate import send_tcp_packet

mode = 0  # 0: 找相机, 1: 抓取, 2: 放置

def get_robot_joint_angle():
    SERVER_IP = "192.168.1.159"
    SERVER_PORT = 5001
    MESSAGE = "get_angles()"
    result = send_tcp_packet(SERVER_IP, SERVER_PORT, MESSAGE)

    data = "get_angles:[-152.558280,-40.508492,126.083485,-171.474609,-90.351562,-0.263672]"
    # 使用正则表达式提取数字
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", data)
    # 转换为 float
    angles = list(map(float, numbers))
    angles[1] += 90
    angles[3] += 90
    return [a/180*np.pi for a in angles]

def get_camera_to_tag(img, id=0):
    tag_size = 0.036
    result = april.detect(img, tag_size)
    img = camera_node.limit_resolution(april.visualize(img, result))

    result_ids = [tag.tag_id for tag in result]
    if id in result_ids:
        tag = result[result_ids.index(id)]
        tag_pose = Transform(Rotation.from_matrix(tag.pose_R), tag.pose_t.squeeze())
        return tag_pose, img
    else:
        return None, img

def get_camera_to_line(img) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
    img, interpolated_coords = detect_line(img, block_imshow=True)
    return interpolated_coords, img

def pixel2D_to_camera3D(pixel_coord, intrinsic, depth, base2camera=Transform.identity()):
    # 像素坐标系转到 base 坐标系
    # pixel_coord: [u, v] 像素坐标
    # intrinsic: 相机内参矩阵 3x3
    # depth: 该点的深度值(米)
    # base2camera: 基坐标系到相机坐标系的变换矩阵

    pixel_homo = np.array(
        [pixel_coord[0], pixel_coord[1], 1]
    )  # 构建齐次像素坐标 [u,v,1]
    camera_norm = (
        np.linalg.inv(intrinsic) @ pixel_homo
    )  # 像素坐标转到相机坐标系下的归一化坐标
    camera_3d = camera_norm * depth  # 乘以深度得到相机坐标系下的3D坐标
    camera_3d_homo = np.append(camera_3d, 1)  # 转为齐次坐标 [x,y,z,1]
    base_3d = base2camera.as_matrix() @ camera_3d_homo
    base_3d = base_3d[:3]  # 去除齐次项,得到基坐标系下的3D坐标 [x,y,z]
    return base_3d

def keyboardCallback(keycode):
    """捕获在 mujoco viewer 上的按键
    """
    if chr(keycode) == " ":  # 空格键进下一步
        global mode
        mode += 1  # 好像不行
    # self.dyn_paused_ = not self.dyn_paused_
    # if self.dyn_pmulation paused!")
    # elif chr(keycode) == "ĉ":
    # elif chr(keycode) == "Ĉ":
    # elif chr(keycode) == "ć":
    # elif chr(keycode) == "Ć":
    # elif chr(keycode) == "O":
    # elif chr(keycode) == "P":
    # elif chr(keycode) == "R":
    # elif chr(keycode) == "Ā":


if __name__ == "__main__":
    mj_model = MjModel.from_xml_path("model/my_cobot.xml")
    mj_data = MjData(mj_model)
    mj_renderer = mujoco.Renderer(mj_model, 640, 640)
    board = RerunBoard(f"Lab3_{time.strftime('%m_%d_%H_%M', time.localtime())}", template="3D")
    camera_node = RGBCamera(
        source=0,
        intrinsic_path="calibration_output/intrinsic_03_27_14_42",
    )
    april = April(camera_node.get_intrinsic_list())
    while True:
        print("Try get one img")
        img = camera_node.get_img(with_info_overlay=False)
        if img is not None:
            break

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=keyboardCallback)
    viewer.cam.azimuth = 70
    viewer.cam.elevation = -30
    viewer.cam.distance = 1.2
    viewer.cam.lookat = [0, 0, 0.5]

    current_robot_angle = get_robot_joint_angle()
    mj_data.qpos[:6] = np.array(current_robot_angle)
    # mj_data.qpos[:6] = np.array([0, 0.7, 1.9, -1.0, -1.5, 0])
    mj_data.qpos[6:] = np.array([0.4, 0.1, 0.5, 1, 0, 0, 0])
    mj_data.ctrl[:] = mj_data.qpos[:6]
    mujoco.mj_step(mj_model, mj_data)
    viewer.sync()

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

    # for init
    camera_pose = Transform(Rotation.from_matrix(mj_data.xmat[mj_model.body('camera').id].reshape(3,3)),
                            mj_data.xpos[mj_model.body('camera').id])
    table_depth = 0.2

    # ===== find camera =====
    # _time_stamp = time.time()
    img = camera_node.get_img(with_info_overlay=False)  # undistort=True
    camera_to_tag, img = get_camera_to_tag(img, id=0)
    if camera_to_tag is not None:
        geom_id = mj_model.geom('ee_tag_geom').id
        base_to_tag = Transform(Rotation.from_matrix(mj_data.geom_xmat[geom_id].reshape(3,3)), mj_data.geom_xpos[geom_id])
        camera_pose = base_to_tag * camera_to_tag.inverse()
        mj_data.qpos[6:9] = camera_pose.translation
        mj_data.qpos[9:] = camera_pose.rotation.as_quat(scalar_first=True)
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

        board.log_axes(base_to_tag, name='ee_tag', label='ee_tag')
        board.log_axes(camera_pose, name='camera', label='camera')
    input("finished find camera")

    # ===== find table depth & line=====
    camera_to_tag, img = get_camera_to_tag(img, id=1)
    if camera_to_tag is not None:
        table_depth = camera_to_tag.translation[2]
        board.log_axes(base_to_tag, name='tag', label='tag')

    camera_to_line, img = get_camera_to_line(img)
    if camera_to_line is not None:
        pass
        # board.log_axes(camera_to_line, name='line', label='line')
    board.log("camera", rr.Image(img, color_model="BGR"))
    input("finished find table depth & line")
