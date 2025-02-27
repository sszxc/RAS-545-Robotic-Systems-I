from dm_control import mjcf
import numpy as np
import mujoco.viewer
from mujoco import MjModel, MjData
import re
import time
import os


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
    base_body.add("geom", type="mesh", mesh="base_mesh", material="metal_gradient_bright_material", pos=[0.02, 0, 0.107])
    base_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())

    # 添加连杆 1
    link1_body = base_body.add("body", name="link1", pos=[0, 0, 0.122], quat=[1, 0, 0, 0])
    link1_body.add("geom", type="mesh", mesh="link1_mesh", material="metal_gradient_dark_material", pos=[0.02, 0, 0.107-0.122])
    link1_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint1 = link1_body.add("joint", name="joint1", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor1", joint=joint1, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 2
    link2_body = link1_body.add("body", name="link2", pos=[0, 0, 1e-5], quat=[1, 0, 0, 0])
    link2_body.add("geom", type="mesh", mesh="link2_mesh", material="metal_gradient_bright_material", pos=[0.02, 0, 0.107-0.122])
    link2_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint2 = link2_body.add("joint", name="joint2", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor2", joint=joint2, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 3
    link3_body = link2_body.add("body", name="link3", pos=[0, 0, 0.15], quat=[1, 0, 0, 0])
    link3_body.add("geom", type="mesh", mesh="link3_mesh", material="metal_gradient_dark_material", pos=[0.02, 0, 0.107-0.122-0.15])
    link3_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint3 = link3_body.add("joint", name="joint3", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor3", joint=joint3, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆 4
    link4_body = link3_body.add("body", name="link4", pos=[0, -0.15, 0], quat=[1, 0, 0, 0])
    link4_body.add("geom", type="mesh", mesh="link4_mesh", material="metal_gradient_bright_material", pos=[0.02, 0.15, 0.107-0.122-0.15])
    link4_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint4 = link4_body.add("joint", name="joint4", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor4", joint=joint4, gear=[1], ctrlrange=[-np.pi, np.pi])
    
    # 添加连杆 5
    link5_body = link4_body.add("body", name="link5", pos=[0, -0.09, 0], quat=[1, 0, 0, 0])
    link5_body.add("geom", type="mesh", mesh="link5_mesh", material="metal_gradient_dark_material", pos=[0.02, 0.15+0.09, 0.107-0.122-0.15])
    link5_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint5 = link5_body.add("joint", name="joint5", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor5", joint=joint5, gear=[1], ctrlrange=[-np.pi, np.pi])

    # # 添加碰撞几何体（用简单形状替代复杂网格）
    # base_body.add(
    #     "geom",
    #     type="box",
    #     size=[0.1, 0.1, 0.05],
    #     pos=[0, 0, 0.025],
    #     contype=1,
    #     conaffinity=1,
    # )  # 启用碰撞

    # # ----- 第1关节和连杆 -----
    # link1_body = base_body.add("body", name="link1", pos=[0, 0, 0.1])
    # # 添加旋转关节（绕Z轴）
    # joint1 = link1_body.add(
    #     "joint",
    #     name="joint1",
    #     type="hinge",
    #     axis=[0, 0, 1],
    #     range=[-np.pi / 2, np.pi / 2],
    # )  # -90°~90°
    # # 添加视觉几何体（STL 网格）
    # link1_body.add("geom", type="mesh", mesh="link1_mesh")
    # # 添加碰撞几何体（圆柱近似）
    # link1_body.add(
    #     "geom",
    #     type="cylinder",
    #     size=[0.05, 0.15],
    #     pos=[0, 0, 0.1],
    #     contype=1,
    #     conaffinity=1,
    # )
    # # 设置惯性属性（质量、质心、惯性矩阵）
    # link1_body.add(
    #     "inertial", mass=1.5, pos=[0, 0, 0.1], diaginertia=[0.05, 0.05, 0.01]
    # )

    # # ========== 添加执行器 ==========
    # model.actuator.add("motor", name="motor1", joint=joint1, gear=100)
    # # model.actuator.add("motor", name="motor2", joint=joint2, gear=50)

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
