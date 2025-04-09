# ikpy model, and keyboard control for sim model visualization

import ikpy
from ikpy import chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

plt.rcParams["font.family"] = "Heiti TC"  # 替换为你选择的字体
from matplotlib.widgets import Button


class Cobot_ikpy:
    def __init__(self):
        # self.chain = ikpy.chain.Chain.from_urdf_file("lab3/my_cobot.urdf")
        self.chain = chain.Chain(name='mycobot_manual', links=[
            # OriginLink(),
            URDFLink("base_link", origin_translation=[0, 0, 0],        origin_orientation=[0, 0, np.pi],       rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
            URDFLink("joint1", origin_translation=[0, 0, 0.21934],     origin_orientation=[np.pi/2, 0, 0],     rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
            URDFLink("joint2", origin_translation=[0, 0.25, 0],        origin_orientation=[0, 0, 0],           rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
            URDFLink("joint3", origin_translation=[0, 0.25, 0],        origin_orientation=[0, 0, 0],           rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
            URDFLink("joint4", origin_translation=[0, 0.1091, 0.108],  origin_orientation=[-np.pi/2, 0, 0],    rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
            URDFLink("joint5", origin_translation=[0, 0.07586, 0],     origin_orientation=[np.pi/2, 0, 0],     rotation=[0, 0, 1], bounds=(-2*np.pi, 2*np.pi)),
        ])

    def ik(self, target_position, target_orientation=None, initial_position=None):
        if initial_position is None:
            initial_position = [0] * len(self.chain.links)
        joint_angles = self.chain.inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation,  # add rotation constraint
            orientation_mode='Z' if target_orientation is not None else None,  # if no rotation constraint, set to None
            initial_position=initial_position,  # initial joint angle
        )
        return joint_angles

    def fk(self, joint_angles) -> np.ndarray:
        return self.chain.forward_kinematics(joint_angles)


def keyboard_control():
    """use keyboard to control the joint angle and visualize the robot"""
    cobot_ikpy = Cobot_ikpy()

    # initialize the joint angle
    joint_angles = [0] * len(cobot_ikpy.chain.links)
    # joint_angles = [0, -np.pi/2, 0, +np.pi/2, 0, 0, 0]

    # set the step of the joint angle
    step = 0.1  # radian

    # create the figure and the axis
    plt.ion()  # open the interactive mode
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # disable the default matplotlib shortcuts
    plt.rcParams["keymap.save"] = []  # disable 's' save function
    plt.rcParams["keymap.quit"] = []  # disable 'q' quit function
    plt.rcParams["keymap.home"] = []  # disable 'h' home function
    plt.rcParams["keymap.back"] = []  # disable 'left arrow' back function

    # plot the initial robot state
    cobot_ikpy.chain.plot(joint_angles, ax)
    plt.draw()

    def update_plot():
        """update the visualization of the robot"""
        ax.clear()
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, 0.8)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("robot joint control\nuse keyboard 1-8 to increase the joint angle, q-i to decrease the joint angle, 0 to reset")

        # calculate the end position
        end_effector = cobot_ikpy.fk(joint_angles)[:3, 3]

        # show the current joint angle and the end position
        angle_info = "joint angle(radian):\n"
        for i in range(len(joint_angles)):
            angle_info += f"joint{i+1}: {joint_angles[i]:.2f}\n"
        angle_info += f"\nend position:\nX: {end_effector[0]:.3f}\nY: {end_effector[1]:.3f}\nZ: {end_effector[2]:.3f}"
        angle_text = ax.text2D(0.05, 0.95, angle_info, transform=ax.transAxes, va="top")

        # redraw the robot
        cobot_ikpy.chain.plot(joint_angles, ax, target=end_effector)
        plt.draw()
        plt.pause(0.001)

    def on_key(event):
        """handle the keyboard event"""
        nonlocal joint_angles

        if event.key in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            # increase the joint angle
            joint_idx = int(event.key) - 1
            joint_angles[joint_idx] = min(joint_angles[joint_idx] + step, np.pi)
            update_plot()

        elif event.key in ["q", "w", "e", "r", "t", "y", "u", "i"]:
            # decrease the joint angle
            key_map = {"q": 0, "w": 1, "e": 2, "r": 3, "t": 4, "y": 5, "u": 6, "i": 7}
            joint_idx = key_map[event.key]
            joint_angles[joint_idx] = max(joint_angles[joint_idx] - step, -np.pi)
            update_plot()

        elif event.key == "0":
            # reset all the joint angle
            joint_angles = [0] * len(cobot_ikpy.chain.links)
            update_plot()

        elif event.key == "escape":
            # exit the program
            plt.close()

    # connect the keyboard event
    fig.canvas.mpl_connect("key_press_event", on_key)

    # show the initial information
    update_plot()

    # keep the window open until the user closes
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    keyboard_control()
