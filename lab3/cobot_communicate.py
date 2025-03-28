import numpy as np
import socket

SERVER_IP = "192.168.1.159"
SERVER_PORT = 5001


def send_tcp_packet(server_ip, server_port, message):
    response = ''
    try:
        # Create a TCP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port}")

        # Send the message
        client_socket.sendall(message.encode("utf-8"))
        print(f"Sent: {message}")

        # Optionally receive a response (if server sends one)
        response = client_socket.recv(1024).decode("utf-8")
        print(f"Received: {response}")

    except socket.error as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        client_socket.close()
        print("Connection closed.")

    return response

def get_robot_joint_angle():
    '''return radian
    '''
    MESSAGE = "get_angles()"
    result = send_tcp_packet(SERVER_IP, SERVER_PORT, MESSAGE)

    # result = "get_angles:[-152.558280,-40.508492,126.083485,-171.474609,-90.351562,-0.263672]"
    # 使用正则表达式提取数字
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", result)
    # 转换为 float
    angles = list(map(float, numbers))
    angles[1] += 90
    angles[3] += 90
    return [a / 180 * np.pi for a in angles]

def send_robot_joint_angle(joint_angles, speed=500):
    joint_angles_shifted = [j/np.pi*180 for j in joint_angles]
    joint_angles_shifted[1] -= 90
    joint_angles_shifted[3] -= 90
    MESSAGE = (
        f"set_angles({','.join([str(j) for j in joint_angles_shifted])}, {speed})"
    )
    response = send_tcp_packet(SERVER_IP, SERVER_PORT, MESSAGE)
    return response
