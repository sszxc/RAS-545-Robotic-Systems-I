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


def get_camera_to_tag():
    tag_size = 0.036
    img = camera_node.get_img(with_info_overlay=True)  # undistort=True
    result = april.detect(img, tag_size)
    board.log("camera", rr.Image(camera_node.limit_resolution(april.visualize(img, result)), color_model="BGR"))

    result_ids = [tag.tag_id for tag in result]
    if 0 in result_ids:
        tag = result[result_ids.index(0)]
        tag_pose = Transform(Rotation.from_matrix(tag.pose_R), tag.pose_t.squeeze())
        return tag_pose
    else:
        return None


if __name__ == "__main__":
    mj_model = MjModel.from_xml_path("meshes/my_cobot.xml")
    mj_data = MjData(mj_model)
    mj_renderer = mujoco.Renderer(mj_model, 640, 640)
    board = RerunBoard(f"Lab3_{time.strftime('%m_%d_%H_%M', time.localtime())}", template="3D")

    camera_node = RGBCamera(
        source=1,
        intrinsic_path="calibration_output/intrinsic_03_20_17_05",
    )
    april = April(camera_node.get_intrinsic_list())

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 70
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.2
        viewer.cam.lookat = [0, 0, 0.5]

        mj_data.qpos[:6] = np.array([0, 0.7, 1.9, -1.0, -1.5, 0])
        mj_data.qpos[6:] = np.array([0.4, 0.1, 0.5, 1, 0, 0, 0])
        mj_data.ctrl[:] = mj_data.qpos[:6]
        mujoco.mj_step(mj_model, mj_data)

        # mujoco.mj_step(mj_model, mj_data)
        # mj_data.geom_xmat[mj_model.geom('link0_geom').id] = np.eye(3).flatten()  # 设置为单位矩阵
        # breakpoint()

        _time_stamp = time.time()
        while viewer.is_running():
            board.step()

            camera_to_tag = get_camera_to_tag()
            if camera_to_tag is not None:
                geom_id = mj_model.geom('ee_tag_geom').id
                base_to_tag = Transform(Rotation.from_matrix(mj_data.geom_xmat[geom_id].reshape(3,3)), mj_data.geom_xpos[geom_id])
                camera_pose = base_to_tag * camera_to_tag.inverse()
                mj_data.qpos[6:9] = camera_pose.translation
                mj_data.qpos[9:] = camera_pose.rotation.as_quat(scalar_first=True)

                board.log_axes(base_to_tag, name='tag', label='tag')
                board.log_axes(camera_pose, name='camera', label='camera')

            if board.time_tick % 10 == 0:
                # for name in ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']:
                #     board.log(f'joint_angle/{name}', rr.Scalar(mj_data.qpos[mj_model.joint(name).id]))
                # for name in [f'link{i}_geom' for i in range(7)] + ['ee_tag_geom']:
                #     geom_id = mj_model.geom(name).id
                #     link_pose = Transform(Rotation.from_matrix(mj_data.geom_xmat[geom_id].reshape(3,3)), mj_data.geom_xpos[geom_id])
                #     board.log_axes(link_pose, name=name, label=name)
                #     # TODO 这里的旋转角很有问题
                pass

            mj_renderer.update_scene(mj_data, camera="demo-cam")
            board.log("mujoco", rr.Image(mj_renderer.render(), color_model="RGB"))


            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            print(f'real time: {time.time() - _time_stamp:.5f}s, sim time: {mj_model.opt.timestep:.5f}s', end='\r')
            _time_stamp = time.time()

            # time.sleep(0.001)
            # 打印最后一个 link6 的位姿
            # geom_id = mj_model.geom('link5_geom').id
            # print(f'position:\n{mj_data.geom_xpos[geom_id]}')
            # print(f'rotation:\n{mj_data.geom_xmat[geom_id].reshape(3,3)}')
            # print()
