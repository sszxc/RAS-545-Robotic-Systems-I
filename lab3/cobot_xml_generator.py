from dm_control import mjcf
import numpy as np
import mujoco.viewer
from mujoco import MjModel, MjData
import re
import time
import os
from rich import print


np.set_printoptions(precision=4, suppress=True)

def build_coordinate_axes(axis_length=0.1, radius=0.005):
    model = mjcf.RootElement(model="coordinate_axes")

    # 定义颜色
    model.asset.add("material", name="red", rgba=[1, 0, 0, 1])
    model.asset.add("material", name="green", rgba=[0, 1, 0, 1])
    model.asset.add("material", name="blue", rgba=[0, 0, 1, 1])

    # 创建坐标轴的 body
    axes_body = model.worldbody.add("body", name="axes")

    # X 轴（红色），沿全局 X 轴正方向
    axes_body.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, axis_length, 0, 0],  # 从原点到 (axis_length, 0, 0)
        size=[radius],  # 只需要半径，长度由 fromto 控制
        material="red",
        contype=0, conaffinity=0,
    )

    # Y 轴（绿色），沿全局 Y 轴正方向
    axes_body.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, axis_length, 0],  # 从原点到 (0, axis_length, 0)
        size=[radius],
        material="green",
        contype=0, conaffinity=0,
    )

    # Z 轴（蓝色），沿全局 Z 轴正方向
    axes_body.add(
        "geom",
        type="capsule",
        fromto=[0, 0, 0, 0, 0, axis_length],  # 从原点到 (0, 0, axis_length)
        size=[radius],
        material="blue",
        contype=0, conaffinity=0,
    )

    return model

def build_robot():
    model = mjcf.RootElement(model="robot_arm")
    model.option.gravity = [0, 0, 0]  # 关闭重力
    model.compiler.angle = "radian"

    # ========== 添加碰撞排除规则 ==========
    # 排除相邻连杆之间的碰撞
    model.contact.add('exclude', name='link0_1', body1='base', body2='link1')
    model.contact.add('exclude', name='link1_2', body1='link1', body2='link2')
    model.contact.add('exclude', name='link2_3', body1='link2', body2='link3')
    model.contact.add('exclude', name='link3_4', body1='link3', body2='link4')
    model.contact.add('exclude', name='link4_5', body1='link4', body2='link5')
    model.contact.add('exclude', name='link5_6', body1='link5', body2='link6')
    model.contact.add('exclude', name='link3_5', body1='link3', body2='link5')

    # ========== 添加地面和光源 ==========
    model.asset.add('texture', name='groundplane', type="2d", builtin='checker', mark="edge", width=300, height=300,
                    rgb1=[0.2, 0.3, 0.4], rgb2=[0.5, 0.6, 0.7], markrgb=[0.8, 0.8, 0.8] )
    model.asset.add('material', name='groundplane', texture='groundplane', texuniform="true", texrepeat="5 5", reflectance="0.5")
    model.worldbody.add('geom', name='ground', type='plane', size=[5, 5, 1], material="groundplane")
    model.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1])

    model.worldbody.add('camera', name='demo-cam', pos="-0.1 -1.0 0.7", xyaxes="0.966 -0.259 0.000 -0.102 0.380 0.919")
    # 配置全局相机视角和离线渲染尺寸
    model.visual.__getattr__("global").azimuth = 70  # 方位角
    model.visual.__getattr__("global").elevation = -30  # 仰角
    model.visual.__getattr__("global").offwidth = 1280  # 离线渲染宽度
    model.visual.__getattr__("global").offheight = 720  # 离线渲染高度

    # ========== 材质定义 ==========
    # 金属渐变效果
    model.asset.add('texture', name='metal_gradient_bright', type='2d', builtin='gradient',
                    rgb1=[0.8, 0.8, 0.8], rgb2=[0.2, 0.2, 0.2], width=100, height=100)
    model.asset.add('material', name='metal_gradient_bright_material', texture='metal_gradient_bright',
                    texuniform=True, reflectance=0.5)
    model.asset.add('texture', name='metal_gradient_dark', type='2d', builtin='gradient',
                    rgb1=[0.4, 0.4, 0.4], rgb2=[0.8, 0.8, 0.8], width=100, height=100)
    model.asset.add('material', name='metal_gradient_dark_material', texture='metal_gradient_dark',
                    texuniform=True, reflectance=0.5)

    # tag 贴图
    model.asset.add('texture', name='tag_texture', file='meshes/tag_texture.png', gridsize=[3, 4],
                    gridlayout=".U..LBRF.D..")  # TODO 仍然不是很懂这里怎么回事(FB翻了一下)
    model.asset.add('material', name='tag_material', texture='tag_texture', specular="15", shininess=".0")

    # ========== 资源加载 ==========
    model.compiler.meshdir = os.path.join(os.path.dirname(__file__), "meshes")
    model.asset.add("mesh", name="link0", file="link0.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link1", file="link1.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link2", file="link2.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link3", file="link3.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link4", file="link4.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link5", file="link5.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="link6", file="link6.stl", scale=[0.001, 0.001, 0.001])
    model.asset.add("mesh", name="camera", file="camera.stl", scale=[0.004, 0.004, 0.004])
    model.asset.add("mesh", name="tag", file="tag_d36.stl", scale=[0.001, 0.001, 0.001])

    # ========== 搭建机械臂 ==========
    worldbody = model.worldbody
    base_body = worldbody.add("body", name="base", axisangle=[1, 0, 0, 0])
    base_body.add("geom", name="link0_geom", type="mesh", mesh="link0", material="metal_gradient_dark_material", pos=[0.0, 0, 0])
    # base_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())

    # 添加连杆 1
    link1_body = base_body.add("body", name="link1", pos=[0, 0, 0.10967])
    link1_body.add("geom", name="link1_geom", type="mesh", mesh="link1", material="metal_gradient_bright_material", pos=[0.0, 0, -0.10967])
    # link1_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint1 = link1_body.add("joint", name="joint1", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor1", joint=joint1, gear=[1], ctrlrange=[-np.pi, np.pi])

    # # 添加连杆 2
    link2_body = link1_body.add("body", name="link2", pos=[0, 0.108,0.08967])
    link2_body.add("geom", name="link2_geom", type="mesh", mesh="link2", material="metal_gradient_dark_material", pos=[0, -0.108,-0.19934])
    # link2_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint2 = link2_body.add("joint", name="joint2", type="hinge", axis=[0, 1, 0], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor2", joint=joint2, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 3
    link3_body = link2_body.add("body", name="link3", pos=[0, 0, 0.25], axisangle=[1, 0, 0, 0])
    link3_body.add("geom", name="link3_geom", type="mesh", mesh="link3", material="metal_gradient_bright_material", pos=[0, -0.108, -0.451])
    # link3_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint3 = link3_body.add("joint", name="joint3", type="hinge", axis=[0, 1, 0], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor3", joint=joint3, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 4
    link4_body = link3_body.add("body", name="link4", pos=[0,0, 0.25], axisangle=[1, 0, 0, 0])
    link4_body.add("geom", name="link4_geom", type="mesh", mesh="link4", material="metal_gradient_dark_material", pos=[0, -0.108, -0.69934])
    # link4_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint4 = link4_body.add("joint", name="joint4", type="hinge", axis=[0, 1, 0], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor4", joint=joint4, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 5
    link5_body = link4_body.add("body", name="link5", pos=[0, 0, 0.05455], axisangle=[1, 0, 0, 0])
    link5_body.add("geom", name="link5_geom", type="mesh", mesh="link5", material="metal_gradient_bright_material", pos=[0, -0.10786, -0.75389])
    # link5_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint5 = link5_body.add("joint", name="joint5", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor5", joint=joint5, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 6
    link6_body = link5_body.add("body", name="link6", pos=[0, 0.05086, 0.05455 - 0.003], axisangle=[1, 0, 0, 0])
    link6_body.add("geom", name="link6_geom", type="mesh", mesh="link6", material="metal_gradient_dark_material", pos=[0, -0.180786, -0.80389])
    # link6_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint6 = link6_body.add("joint", name="joint6", type="hinge", axis=[0, 1, 0], range=[-np.pi, np.pi], damping="2")
    model.actuator.add("position", name="motor6", joint=joint6, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加末端 tag
    ee_tag = link5_body.add("body", name="ee_tag", pos=[0, -0.048, 0.05455 - 0.003], quat=[0.707, -0.707, 0, 0])
    ee_tag.add("geom", name="ee_tag_geom", type="box", size=[0.036/2, 0.036/2, 0.001], material="tag_material",
               contype=0, conaffinity=0, density="1")
    ee_tag.add("site").attach(build_coordinate_axes(axis_length=0.05, radius=0.003))
    # ee_tag.add("geom", name="ee_tag_geom", type="mesh", mesh="tag", material="tag_material",
    #            contype=0, conaffinity=0, density="1")

    # 添加一个 freejoint camera
    camera = model.worldbody.add('body', name='camera')
    # camera.add('geom', type='mesh', size=[0.01], pos=[0, 0, 0], rgba=[1, 1, 0, 1],  # 半径 0.02
    camera.add("geom", name="camera", type="mesh", mesh="camera", material="metal_gradient_bright_material", pos=[0, 0, 0], quat=[0.5, 0.5, -0.5, 0.5],  # quat=[0, 0.707, 0, -0.707],
                contype=0, conaffinity=0, density="1")
    camera.add("site").attach(build_coordinate_axes(axis_length=0.09, radius=0.003))
    camera.add("freejoint", name="camera_joint")  # 添加自由关节

    # camera_joint_x = camera.add('joint', name='joint_x', type='slide', axis=[1, 0, 0], damping="5")  # , range=[-1, 1]
    # camera_joint_y = camera.add('joint', name='joint_y', type='slide', axis=[0, 1, 0], damping="5")  # , range=[-1, 1]
    # camera_joint_z = camera.add('joint', name='joint_z', type='slide', axis=[0, 0, 1], damping="5")  # , range=[-1, 1]
    # model.actuator.add('position', name=f'camera_x', joint=camera_joint_x, kp="50", kv="5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")
    # model.actuator.add('position', name=f'camera_y', joint=camera_joint_y, kp="50", kv="5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")
    # model.actuator.add('position', name=f'camera_z', joint=camera_joint_z, kp="50", kv="5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")
    return model

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "meshes/my_cobot.xml")

    # 生成模型
    robot_model = build_robot()
    xml_string = robot_model.to_xml_string()

    # 使用正则表达式替换文件名中的哈希值
    processed_xml = re.sub(r'(file="[^"]*)-[a-f0-9]{40}(\.(stl|png)")', r"\1\2", xml_string)

    # 保存处理后的XML
    with open(output_file, "w") as f:
        f.write(processed_xml)

    # 启动 mujoco viewer
    _time_stamp = time.time()
    mj_model = MjModel.from_xml_path(output_file)
    mj_data = MjData(mj_model)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 70
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.2
        viewer.cam.lookat = [0, 0, 0.5]

        while viewer.is_running():
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

            # geom_id = mj_model.geom('link0_geom').id
            # print(f'position:\n{mj_data.geom_xpos[geom_id]}')
            # print(f'rotation:\n{mj_data.geom_xmat[geom_id].reshape(3,3)}')
