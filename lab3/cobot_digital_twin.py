import numpy as np
import socket
import re
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
from utils.misc_utils import DummyClass


def check_radian(joint_angles: list[float]) -> bool:
    if all(-2 * np.pi <= angle <= 2 * np.pi for angle in joint_angles):
        return True
    else:
        raise ValueError(f"Joint angles must be in the range of -2π to 2π, did you forget to convert to radian?\n Got joint_angles: {joint_angles}")


class CobotDigitalTwin:
    def __init__(self, real: bool = False, sim: bool = True):
        if real:
            self.real = CobotReal()
        else:
            self.real = DummyClass()
        if sim:
            self.sim = CobotSim()
        else:
            self.sim = DummyClass()


class CobotSim:
    """MuJoCo model"""
    def __init__(self):
        self.mj_model = MjModel.from_xml_path("model/my_cobot.xml")
        self.mj_data = MjData(self.mj_model)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def get_joint_angles(self, is_radian: bool = True) -> list[float]:
        return self.mj_data.qpos if not is_radian else [a / np.pi * 180 for a in self.mj_data.qpos]

    def send_joint_angles(self, joint_angles, is_radian=True):
        if is_radian:
            check_radian(joint_angles)
            self.mj_data.qpos[:] = joint_angles
        else:
            self.mj_data.qpos[:] = [j / 180 * np.pi for j in joint_angles]
        self.mj_data.ctrl[:] = joint_angles
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()

    def __del__(self):
        self.viewer.close()

class CobotReal:
    """Real robot"""
    SERVER_IP = "192.168.1.159"
    SERVER_PORT = 5001

    def __init__(self, server_ip=SERVER_IP, server_port=SERVER_PORT):
        self.server_ip = server_ip
        self.server_port = server_port
        # Create a TCP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set connection timeout to 5 seconds
        self.client_socket.settimeout(5)
        try:
            # Connect to the server
            self.client_socket.connect((server_ip, server_port))
            print(f"Connected to {server_ip}:{server_port}")
        except socket.timeout:
            print(f"Connection timeout: cannot connect to {server_ip}:{server_port} within 5 seconds")
            raise
        except ConnectionRefusedError:
            print(f"Connection refused: {server_ip}:{server_port} server is not running or refusing connection")
            raise

    def send_tcp_packet(self, message):
        response = ''
        try:
            # Send the message
            self.client_socket.sendall(message.encode("utf-8"))
            # print(f"Sent: {message}")
            # Optionally receive a response (if server sends one)
            response = self.client_socket.recv(1024).decode("utf-8")
            # print(f"Received: {response}")
        except socket.error as e:
            print(f"Error: {e}")
        return response
    
    def __del__(self):
        # Close the connection
        self.client_socket.close()
        print("Connection closed.")

    def get_joint_angles(self, is_radian: bool = True) -> list[float]:
        '''return radian
        '''
        MESSAGE = "get_angles()"
        result = self.send_tcp_packet(MESSAGE)

        # example result = "get_angles:[-152.558280,-40.508492,126.083485,-171.474609,-90.351562,-0.263672]"
        # 使用正则表达式提取数字
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", result)
        # 转换为 float
        angles = list(map(float, numbers))
        angles[1] += 90
        angles[3] += 90
        return [a / 180 * np.pi for a in angles] if is_radian else angles

    def send_joint_angles(self, joint_angles, speed=500, is_radian=True):
        if is_radian:
            check_radian(joint_angles)
            joint_angles_shifted = [j/np.pi*180 for j in joint_angles]
        else:
            joint_angles_shifted = joint_angles

        joint_angles_shifted[1] -= 90
        joint_angles_shifted[3] -= 90

        MESSAGE = (
            f"set_angles({','.join([str(j) for j in joint_angles_shifted])}, {speed})"
        )
        response = self.send_tcp_packet(MESSAGE)
        return response

if __name__ == "__main__":
    cobot_real = CobotReal()
    print(cobot_real.get_joint_angles())

    cobot_sim = CobotSim()
    print(cobot_sim.get_joint_angles())
    cobot_sim.send_joint_angles([0, 0, 0, 0, 0, 0])
