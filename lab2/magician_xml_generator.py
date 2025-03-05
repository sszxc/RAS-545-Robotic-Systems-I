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

    # ========== 添加地面和光源 ==========
    model.asset.add('texture', name='groundplane', type="2d", builtin='checker', mark="edge", width=300, height=300,
                    rgb1=[0.2, 0.3, 0.4], rgb2=[0.5, 0.6, 0.7], markrgb=[0.8, 0.8, 0.8] )
    model.asset.add('material', name='groundplane', texture='groundplane', texuniform="true", texrepeat="5 5", reflectance="0.5")
    model.worldbody.add('geom', name='ground', type='plane', size=[5, 5, 1], material="groundplane")
    model.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1])

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

    # ========== 资源加载 ==========
    model.compiler.meshdir = os.path.join(os.path.dirname(__file__), "assets")
    model.asset.add(
        "mesh", name="base_mesh", file="magician_lite_base_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link1_mesh", file="magician_lite_link1_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link2_mesh", file="magician_lite_link2_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link3_mesh", file="magician_lite_link3_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link4_mesh", file="magician_lite_link4_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link5_mesh", file="magician_lite_link5_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    
    # ========== 搭建机械臂 ==========
    worldbody = model.worldbody
    base_body = worldbody.add("body", name="base")
    base_body.add("geom", name="base_geom", type="mesh", mesh="base_mesh", material="metal_gradient_bright_material", pos=[0.02, 0, 0.107])
    base_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())

    # 添加连杆 1
    link1_body = base_body.add("body", name="link1", pos=[0, 0, 0.122+1e-5], quat=[1, 0, 0, 0])
    link1_body.add("geom", name="link1_geom", type="mesh", mesh="link1_mesh", material="metal_gradient_dark_material", pos=[0.02, 0, 0.107-0.122])
    link1_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint1 = link1_body.add("joint", name="joint1", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor1", joint=joint1, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 2
    link2_body = link1_body.add("body", name="link2", pos=[0, 0, 0], quat=[1, 0, 0, 0])
    link2_body.add("geom", name="link2_geom", type="mesh", mesh="link2_mesh", material="metal_gradient_bright_material", pos=[0.02, 0, 0.107-0.122])
    link2_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint2 = link2_body.add("joint", name="joint2", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor2", joint=joint2, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 3
    link3_body = link2_body.add("body", name="link3", pos=[0, -0.02, 0.15], quat=[1, 0, 0, 0])
    link3_body.add("geom", name="link3_geom", type="mesh", mesh="link3_mesh", material="metal_gradient_dark_material", pos=[0.02, 0.02, 0.107-0.122-0.15])
    link3_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint3 = link3_body.add("joint", name="joint3", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor3", joint=joint3, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 4
    link4_body = link3_body.add("body", name="link4", pos=[0, -0.165, 0], quat=[1, 0, 0, 0])
    link4_body.add("geom", name="link4_geom", type="mesh", mesh="link4_mesh", material="metal_gradient_bright_material", pos=[0.02, 0.02+0.165, 0.107-0.122-0.15])
    link4_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint4 = link4_body.add("joint", name="joint4", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor4", joint=joint4, gear=[1], ctrlrange=[-np.pi, np.pi])
    
    # 添加连杆 5
    link5_body = link4_body.add("body", name="link5", pos=[0, -0.055, 0], quat=[1, 0, 0, 0])
    link5_body.add("geom", name="link5_geom", type="mesh", mesh="link5_mesh", material="metal_gradient_dark_material", pos=[0.02, 0.02+0.165+0.055, 0.107-0.122-0.15])
    link5_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint5 = link5_body.add("joint", name="joint5", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor5", joint=joint5, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加一个 freejoint ball 用于指示器
    ball = model.worldbody.add('body', name='ball')
    ball.add('geom', type='sphere', size=[0.01], pos=[0, 0, 0], rgba=[1, 1, 0, 1],  # 半径 0.02
                contype=0, conaffinity=0)
    ball_joint_x = ball.add('joint', name='joint_x', type='slide', axis=[1, 0, 0], damping="5")  # , range=[-1, 1]
    ball_joint_y = ball.add('joint', name='joint_y', type='slide', axis=[0, 1, 0], damping="5")  # , range=[-1, 1]
    ball_joint_z = ball.add('joint', name='joint_z', type='slide', axis=[0, 0, 1], damping="5")  # , range=[-1, 1]
    
    ball.add("site").attach(build_coordinate_axes(axis_length=0.05, radius=0.003))

    # 添加执行器来控制关节
    model.actuator.add('position', name=f'ball_x', joint=ball_joint_x, kp="20", kv="0.5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")
    model.actuator.add('position', name=f'ball_y', joint=ball_joint_y, kp="20", kv="0.5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")
    model.actuator.add('position', name=f'ball_z', joint=ball_joint_z, kp="20", kv="0.5")  # ctrlrange=[-1.5, 1.5], ctrllimited="false")

    return model

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "assets/magician.xml")

    # 生成模型
    robot_model = build_robot()
    xml_string = robot_model.to_xml_string()

    # 使用正则表达式替换文件名中的哈希值
    processed_xml = re.sub(r'(file="[^"]*)-[a-f0-9]{40}(\.stl")', r"\1\2", xml_string)

    # 保存处理后的XML
    with open(output_file, "w") as f:
        f.write(processed_xml)

    # 启动 mujoco viewer
    mj_model = MjModel.from_xml_path(output_file)
    mj_data = MjData(mj_model)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            time.sleep(0.001)
            # 打印最后一个 link5 的位姿
            # geom_id = mj_model.geom('link5_geom').id
            # print(f'position:\n{mj_data.geom_xpos[geom_id]}')
            # print(f'rotation:\n{mj_data.geom_xmat[geom_id].reshape(3,3)}')
            # print()
