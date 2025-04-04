import numpy as np
import socket
import re

SERVER_IP = "192.168.1.159"
SERVER_PORT = 5001

class CobotCommunicate:
    def __init__(self, server_ip=SERVER_IP, server_port=SERVER_PORT):
        self.server_ip = server_ip
        self.server_port = server_port
        # Create a TCP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        self.client_socket.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port}")

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

    def get_robot_joint_angle(self):
        '''return radian
        '''
        MESSAGE = "get_angles()"
        result = self.send_tcp_packet(MESSAGE)

        # result = "get_angles:[-152.558280,-40.508492,126.083485,-171.474609,-90.351562,-0.263672]"
        # 使用正则表达式提取数字
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", result)
        # 转换为 float
        angles = list(map(float, numbers))
        angles[1] += 90
        angles[3] += 90
        return [a / 180 * np.pi for a in angles]

    def send_robot_joint_angle(self, joint_angles, speed=500, is_radian=True):
        if is_radian:
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
    cobot_communicate = CobotCommunicate()
    print(cobot_communicate.get_robot_joint_angle())
