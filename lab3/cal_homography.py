import cv2
import numpy as np

# Store pixel points
pixel_points = []


def get_pixel_points(event, x, y, flags, param):
    """Capture four pixel points when clicked."""
    if event == cv2.EVENT_LBUTTONDOWN and len(pixel_points) < 4:
        print(f"Clicked at: ({x}, {y})")
        pixel_points.append((x, y))


def compute_homography(pixel_points, world_points):
    """Compute homography matrix from pixel to real-world coordinates."""
    if len(pixel_points) != 4 or len(world_points) != 4:
        print("Error: Need exactly four corresponding points for homography.")
        return None

    pixel_points = np.array(pixel_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)[:, :2]  # Ignore Z for 2D transformation
    H, _ = cv2.findHomography(pixel_points, world_points)
    return H


def pixel_to_world_homography(pixel_coords, H):
    """Convert pixel coordinates to real-world coordinates using homography."""
    pixel_homo = np.array([pixel_coords[0], pixel_coords[1], 1], dtype=np.float32)
    world_homo = H @ pixel_homo
    world_coords = world_homo[:2] / world_homo[2]  # Normalize
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
    # # ===== detect lines =====
    # cap = cv2.VideoCapture(0)

    # intrinsic = np.array([
    #     [1180.5678174909947, 0.0, 678.0955796013238],
    #     [0.0, 1180.693516730357, 358.51815164541397],
    #     [0.0, 0.0, 1.0]
    # ])

    # depth_value = 0.36830

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame, _, _ = detect_line(frame, intrinsic, depth_value)

    #     cv2.imshow('Processed Line Detection', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # # ===== cal homography =====
    # # Open camera to select four corner points
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     cv2.imshow("Select Corners", frame)
    #     cv2.setMouseCallback("Select Corners", get_pixel_points)

    #     if cv2.waitKey(1) & 0xFF == ord("q") or len(pixel_points) == 4:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # print("Selected Pixel Points:", pixel_points)

    # Manually enter corresponding real-world coordinates (X, Y, Z)
    pixel_points = [
        (1340, 897),
        (590, 884),
        (1361, 158),
        (606, 135)
    ]

    world_points = [
        [-0.2358, -0.3345, 0.1859],  # Top-left
        [-0.3858, -0.3345, 0.1859],  # Top-right
        [-0.2358, -0.1945, 0.1859],  # Bottom-left
        [-0.3858, -0.1945, 0.1859],  # Bottom-right
    ]

    # Compute Homography
    H = compute_homography(pixel_points, world_points)
    # H = np.array(
    # )
    print("Homography Matrix:\n", H)

    # Test the homography
    test_pixel_points = [
        (1340, 897),
        (1350, 500),
        (1361, 158),
    ]

    for px, py in test_pixel_points:
        print(f"pixel: ({px}, {py})")
        world_coords = pixel_to_world_homography((px, py), H)
        print(f"world: {world_coords}")
