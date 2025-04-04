# Author: Tracy

import cv2
import numpy as np


def pixel2D_to_camera3D(pixel_coord, intrinsic, depth, base2camera=np.eye(4)):
    pixel_homo = np.array([pixel_coord[0], pixel_coord[1], 1])
    camera_norm = np.linalg.inv(intrinsic) @ pixel_homo
    camera_3d = camera_norm * depth
    camera_3d_homo = np.append(camera_3d, 1)
    base_3d = base2camera @ camera_3d_homo
    return base_3d[:3]

def detect_line(frame, intrinsic=None, depth_map=1.0, base2camera=np.eye(4), block_imshow=False):
    if block_imshow:  # disable opencv imshow()
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: None
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (workspace)
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a mask for the workspace
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # Restrict processing to only the workspace
        workspace = cv2.bitwise_and(binary, mask)
        workspace = cv2.add(workspace, cv2.bitwise_not(mask))
        cv2.imshow('mask', mask)
        cv2.imshow('workspace', workspace)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(workspace, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=2)
        dilated = cv2.bitwise_not(dilated)
        cv2.imshow("dilated", dilated)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            dilated, 9, np.pi / 180, 100, minLineLength=100, maxLineGap=10
        )
        
        # Initialize variables to store start and end points of the longest line
        longest_line = None
        max_length = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Ensure the line is inside the detected workspace
                if mask[y1, x1] == 255 and mask[y2, x2] == 255:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)
        
        # If a valid line is detected
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            # Draw the detected line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)
            
            # Interpolate 10 points between the start and end points
            interpolated_coords = interpolate(x1, y1, x2, y2, 10)

            # Magenta dots (2D visualization)
            for px, py in interpolated_coords:
                cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)

            # # convert to 3D coordinates
            # points_3D = []
            # for (px, py) in interpolated_coords:
            #     pt3D = pixel2D_to_camera3D((px, py), intrinsic, depth_map, base2camera)
            #     points_3D.append(pt3D)

            # print the result
            print(f"Start Point: ({x1}, {y1}) End Point: ({x2}, {y2})")
            # print("Interpolated 3D points:")
            # for i, pt in enumerate(points_3D):
            #     print(f"Point {i+1}: {pt}")

            return frame, interpolated_coords  #, points_3D

    return frame, None  #, None

def interpolate(x1, y1, x2, y2, num_points=10):
    index = np.linspace(0, 1, num_points)
    interpolated_coords = [
        (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)) for t in index
    ]
    return interpolated_coords


if __name__ == "__main__":
    # Open camera (change index if needed)
    cap = cv2.VideoCapture(0)

    # camera intrinsic matrix
    intrinsic = np.array([
        [1180.5678174909947, 0.0, 678.0955796013238],
        [0.0, 1180.693516730357, 358.51815164541397],
        [0.0, 0.0, 1.0]
    ])

    # depth setting
    # depth_value = 0.36830  # meter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _ = detect_line(frame)  # , intrinsic, depth_value)

        cv2.imshow('Processed Line Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
