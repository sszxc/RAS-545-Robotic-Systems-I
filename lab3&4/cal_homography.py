import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# Store pixel points
pixel_points = []

def get_pixel_points(event, x, y, flags, param):
    """Capture four pixel points when clicked."""
    if event == cv2.EVENT_LBUTTONDOWN and len(pixel_points) < 4:
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        print(f"Clicked at: ({x}, {y})")
        pixel_points.append((x, y))

def compute_homography(pixel_points, world_points):
    """Compute homography matrix from pixel to real-world coordinates."""
    if len(pixel_points) != 4 or len(world_points) != 4:
        print("Error: Need exactly four corresponding points for homography.")
        return None

    pixel_points = np.array(pixel_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)[:, :2]  # Ignore Z for 2D transformation
    print(f"Start compute homography for:")
    for i in range(4):
        print(f"{pixel_points[i]} <--> {world_points[i]}")
    H, _ = cv2.findHomography(pixel_points, world_points)
    print("Get homography matrix:\n", H)
    return H


def pixel_to_world_homography(pixel_coords, H):
    """Convert pixel coordinates to real-world coordinates using homography."""
    pixel_homo = np.array([pixel_coords[0], pixel_coords[1], 1], dtype=np.float32)
    world_homo = H @ pixel_homo
    world_coords = world_homo[:2] / world_homo[2]  # Normalize 如果是简单仿射变换，最后一个维度应该接近 1；否则表示透视变换（可以理解成相对位置大小作用到 scale 上）
    return world_coords

def detect_line(frame, block_imshow=False):
    """Detect the longest line in the image and convert pixel points to world coordinates."""
    if block_imshow:
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        workspace = cv2.bitwise_and(binary, mask)
        workspace = cv2.add(workspace, cv2.bitwise_not(mask))

        blurred = cv2.GaussianBlur(workspace, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=2)
        dilated = cv2.bitwise_not(dilated)

        lines = cv2.HoughLinesP(dilated, 9, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        longest_line = None
        max_length = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if mask[y1, x1] == 255 and mask[y2, x2] == 255:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)

        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)

            # Interpolate points along the line
            interpolated_coords = interpolate(x1, y1, x2, y2, 10)

            # Convert pixel points to world coordinates using homography
            world_2D_points = [pixel_to_world_homography((px, py), H) for px, py in interpolated_coords]

            # Print results
            print("Interpolated World Coordinates:")
            for i, pt in enumerate(world_2D_points):
                print(f"Point {i + 1}: {pt}")

            return frame, interpolated_coords, world_2D_points

    return frame, None, None

def interpolate(x1, y1, x2, y2, num_points=10):
    """Interpolate points along a line."""
    index = np.linspace(0, 1, num_points)
    interpolated_coords = [
        (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)) for t in index
    ]
    return interpolated_coords


if __name__ == "__main__":
    # # ===== get frame =====
    # USE_CAMERA = True
    # if USE_CAMERA:
    #     cap = cv2.VideoCapture(0)
    #     for _ in range(10):
    #         ret, frame = cap.read()
    #     while True:
    #         ret, frame = cap.read()
    #         if ret:
    #             break
    #     cap.release()
    # else:
    #     frame = cv2.imread("lab4/example_straight.png")

    # ===== pick 4 points =====
    # while True:
    #     cv2.imshow("Select Corners", frame)
    #     cv2.setMouseCallback("Select Corners", get_pixel_points)
    #     if cv2.waitKey(1) & 0xFF == ord("q") or len(pixel_points) == 4:
    #         break
    # cv2.destroyAllWindows()
    # print(f"pixel_points: {pixel_points}")
    # exit()

    # lab3&4
    # pixel_points = [
    #     (1342, 901),
    #     (596, 892),
    #     (611, 142),
    #     (1362, 157),
    # ]
    # # Manually enter corresponding real-world coordinates (X, Y, Z)
    # world_points = [
    #     [-0.2354, -0.3446,  0.1761],  # Top-left
    #     [-0.3754, -0.3446,  0.1761],  # Top-right
    #     [-0.3754, -0.1846,  0.1761],  # Bottom-left
    #     [-0.2354, -0.1846,  0.1761],  # Bottom-right
    # ]

    # lab5
    pixel_points = [(1297, 836), (546, 852), (535, 84), (1303, 84)]  
    world_points = [
        [-0.2455, -0.3347,  0.3159],
        [-0.3855, -0.3347,  0.3159],
        [-0.3855, -0.1847,  0.3159],
        [-0.2355, -0.1847,  0.3159],
    ]

    # ===== compute homography =====
    H = compute_homography(pixel_points, world_points)

    # ===== test homography =====
    test_pixel_points = [
        # (0, 0),
        # (1920, 1080),
        (1297, 836),
        (546, 852),
        (535, 84),
        (1303, 84)
    ]
    workspace_height = 0.3159

    for px, py in test_pixel_points:
        world_coords = pixel_to_world_homography((px, py), H)
        print(f"pixel: ({px}, {py}) <--> world: {world_coords}")

    # ===== test homography on real robot =====
    from cobot_digital_twin import CobotDigitalTwin
    from cobot_ikpy_model import Cobot_ikpy

    cobot_ikpy = Cobot_ikpy()
    cobot = CobotDigitalTwin(real=True, sim=False)
    HOME_JOINT_ANGLES = [-120, 30, 100, -60, -90, 0]
    cobot.real.send_joint_angles(HOME_JOINT_ANGLES, speed=1000, is_radian=False)
    input("Press Enter to continue...")
    final_pts = [
        pixel_to_world_homography((px, py), H)
        for px, py in test_pixel_points
    ]
    final_3D_pts = []
    for pt in final_pts:
        final_3D_pts.append(np.array([pt[0], pt[1], workspace_height]))

    for pt in final_3D_pts:
        joint_angles = cobot_ikpy.ik(
            pt,
            target_orientation=[0, 0, -1],
            initial_position=cobot.real.get_joint_angles(),
        )
        input("Press Enter to send to real robot...")
        cobot.real.send_joint_angles(joint_angles)
