from dm_control import mjcf
import numpy as np
import re


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
    model.asset.add("material", name="blue", rgba=[0, 0, 1, 0.5])
    model.asset.add("material", name="black", rgba=[0, 0, 0, 0.5])
    model.asset.add("material", name="white", rgba=[1, 1, 1, 0.5])
    # - 颜色质感示例
    # 金属质感材质（高反光）
    model.asset.add("material", name="metal", rgba=[0.8, 0.8, 0.8, 1], 
                    shininess=1.0, specular=1.0, reflectance=0.5)
    # 塑料质感材质（中等反光）
    model.asset.add("material", name="plastic", rgba=[1, 1, 1, 1],
                    shininess=0.5, specular=0.3)
    # 橡胶质感材质（低反光）
    model.asset.add("material", name="rubber", rgba=[0.2, 0.2, 0.2, 1],
                    shininess=0.1, specular=0.1)
    # - 渐变贴图示例（gradient）
    # 从蓝色渐变到白色
    model.asset.add('texture', name='blue_gradient', type='2d', builtin='gradient',
                    rgb1=[0, 0, 1], rgb2=[1, 1, 1], width=100, height=100)
    model.asset.add('material', name='gradient_material', texture='blue_gradient',
                    texuniform=True)
    # 金属渐变效果
    model.asset.add('texture', name='metal_gradient', type='2d', builtin='gradient',
                    rgb1=[0.8, 0.8, 0.8], rgb2=[0.2, 0.2, 0.2], width=100, height=100)
    model.asset.add('material', name='metal_gradient_material', texture='metal_gradient',
                    texuniform=True, reflectance=0.5)
    # - 棋盘格贴图示例（checker）
    # 经典黑白棋盘
    model.asset.add('texture', name='chess', type='2d', builtin='checker',
                    rgb1=[0, 0, 0], rgb2=[1, 1, 1], width=300, height=300, mark='edge',
                    markrgb=[0.8, 0.8, 0.8])
    model.asset.add('material', name='chess_material', texture='chess',
                    texrepeat=[5, 5])  # 重复5x5次
    # - 纯色贴图示例（flat）
    # 简单的纯红色
    model.asset.add('texture', name='red_flat', type='2d', builtin='flat',
                    rgb1=[1, 0, 0], width=300, height=300, mark='edge',
                    markrgb=[0.8, 0.8, 0.8])
    model.asset.add('material', name='red_flat_material', texture='red_flat')


    # ========== 资源加载 ==========
    model.compiler.meshdir = "./assets"
    model.asset.add(
        "mesh", name="base_mesh", file="magician_lite_base_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link1_mesh", file="magician_lite_link1_simple.stl", scale=[0.001, 0.001, 0.001]
    )
    model.asset.add(
        "mesh", name="link2_mesh", file="magician_lite_link2_simple.stl", scale=[0.001, 0.001, 0.001]
    )


    # ========== 搭建机械臂 ==========
    worldbody = model.worldbody
    base_body = worldbody.add("body", name="base")
    base_body.add("geom", type="mesh", mesh="base_mesh", material="plastic", pos=[0.02, 0, 0.107])
    base_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())

    # 添加连杆
    link1_body = base_body.add("body", name="link1", pos=[0, 0, 0.11], quat=[1, 0, 0, 0])
    link1_body.add("geom", type="mesh", mesh="link1_mesh", material="rubber", pos=[0.02, 0, 0.107-0.11+1e-5])
    link1_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint1 = link1_body.add("joint", name="joint1", type="hinge", axis=[0, 0, 1], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor1", joint=joint1, gear=[1], ctrlrange=[-np.pi, np.pi])

    # 添加连杆
    link2_body = link1_body.add("body", name="link2", pos=[0, 0, 1e-5], quat=[1, 0, 0, 0])
    link2_body.add("geom", type="mesh", mesh="link2_mesh", material="metal_gradient_material", pos=[0.02, 0, 0.107-0.11])
    link2_body.add("site", pos=[0, 0, 0]).attach(build_coordinate_axes())
    joint2 = link2_body.add("joint", name="joint2", type="hinge", axis=[1, 0, 0], range=[-np.pi, np.pi], damping="0.2")
    model.actuator.add("position", name="motor2", joint=joint2, gear=[1], ctrlrange=[-np.pi, np.pi])


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
    output_file = "assets/magician.xml"

    # 生成模型
    robot_model = build_robot()
    xml_string = robot_model.to_xml_string()

    # 使用正则表达式替换文件名中的哈希值
    processed_xml = re.sub(r'(file="[^"]*)-[a-f0-9]{40}(\.stl")', r"\1\2", xml_string)

    # 保存处理后的XML
    with open(output_file, "w") as f:
        f.write(processed_xml)
