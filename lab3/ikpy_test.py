import numpy as np
from ikpy import chain
from ikpy.link import OriginLink, URDFLink
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

plt.rcParams["font.family"] = "Heiti TC"  # 替换为你选择的字体
from matplotlib.widgets import Button


class Cobot_ikpy:
    def __init__(self):
        # 创建自定义机械臂的Chain对象
        self.chain = chain.Chain(name='mycobot_manual', links=[
            URDFLink("base_link", origin_translation=[0, 0, 0],     origin_orientation=[0, 0, 0],           rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint1", origin_translation=[0, 0, 0.21934],  origin_orientation=[np.pi/2, 0, 0],     rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint2", origin_translation=[-0.25, 0, 0],    origin_orientation=[0, 0, 0],           rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint3", origin_translation=[-0.25, 0, 0],    origin_orientation=[0, 0, 0],           rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint4", origin_translation=[0, 0, -0.108],   origin_orientation=[-np.pi/2, 0, 0],    rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint5", origin_translation=[0, 0, 0.1091],   origin_orientation=[np.pi/2, 0, 0],     rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
            URDFLink("joint6", origin_translation=[0, 0, -0.07586], origin_orientation=[np.pi, 0, 0],       rotation=[0, 0, 1], bounds=(-np.pi, np.pi)),
        ])

    def ik(self, target_position, target_orientation=None, initial_position=None):
        # 计算逆运动学解
        if initial_position is None:
            initial_position = [0] * len(self.chain.links)
        joint_angles = self.chain.inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation,  # 可添加姿态约束
            orientation_mode='Z' if target_orientation is not None else None,  # 如果不需要姿态约束设为None
            initial_position=initial_position,  # 初始关节角度
        )
        return joint_angles


    def fk(self, joint_angles):
        return self.chain.forward_kinematics(joint_angles)



def keyboard_control():
    """使用键盘控制机械臂关节角度并实时可视化"""
    cobot_ikpy = Cobot_ikpy()

    # 初始化关节角度 (各关节角度设为0)
    # joint_angles = [0] * len(cobot_ikpy.chain.links)
    joint_angles = [0, -np.pi/2, 0, +np.pi/2, 0, 0, 0]

    # 设置关节角度步长
    step = 0.1  # 弧度

    # 创建图形和坐标轴
    plt.ion()  # 打开交互模式
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 设置图形属性
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0, 0.8)
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.set_zlabel("Z轴")
    ax.set_title("机械臂关节控制\n使用键盘1-6增加关节角度，q-y减少关节角度，r重置")

    # 显示当前关节角度的文本
    angle_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # 绘制初始机械臂状态
    cobot_ikpy.chain.plot(joint_angles, ax)
    plt.draw()

    def update_plot():
        """更新机械臂可视化"""
        ax.clear()
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, 0.8)
        ax.set_xlabel("X轴")
        ax.set_ylabel("Y轴")
        ax.set_zlabel("Z轴")
        ax.set_title("机械臂关节控制\n使用键盘1-6增加关节角度，q-y减少关节角度，r重置")

        # 计算末端位置
        end_effector = cobot_ikpy.fk(joint_angles)[:3, 3]

        # 显示当前关节角度和末端位置
        angle_info = "关节角度(弧度):\n"
        for i in range(len(joint_angles)):
            angle_info += f"关节{i+1}: {joint_angles[i]:.2f}\n"
        angle_info += f"\n末端位置:\nX: {end_effector[0]:.3f}\nY: {end_effector[1]:.3f}\nZ: {end_effector[2]:.3f}"
        angle_text = ax.text2D(0.05, 0.95, angle_info, transform=ax.transAxes, va="top")

        # 重新绘制机械臂
        cobot_ikpy.chain.plot(joint_angles, ax, target=end_effector)
        plt.draw()
        plt.pause(0.001)

    def on_key(event):
        """键盘事件处理函数"""
        nonlocal joint_angles

        if event.key in ["1", "2", "3", "4", "5", "6", "7"]:
            # 增加关节角度
            joint_idx = int(event.key) - 1
            joint_angles[joint_idx] = min(joint_angles[joint_idx] + step, np.pi)
            update_plot()

        elif event.key in ["a", "s", "d", "f", "g", "h", "j"]:
            # 减少关节角度
            key_map = {"a": 0, "s": 1, "d": 2, "f": 3, "g": 4, "h": 5, "j": 6}
            joint_idx = key_map[event.key]
            joint_angles[joint_idx] = max(joint_angles[joint_idx] - step, -np.pi)
            update_plot()

        elif event.key == "0":
            # 重置所有关节角度
            joint_angles = [0] * len(cobot_ikpy.chain.links)
            update_plot()

        elif event.key == "escape":
            # 退出程序
            plt.close()

    # 连接键盘事件
    fig.canvas.mpl_connect("key_press_event", on_key)

    # 显示初始信息
    update_plot()

    # 保持窗口打开，直到用户关闭
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    keyboard_control()
