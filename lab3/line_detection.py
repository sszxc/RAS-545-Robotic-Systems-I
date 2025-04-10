# Author: Tracy

import cv2
import numpy as np
import time
from scipy.interpolate import splprep, splev


def pixel2D_to_camera3D(pixel_coord, intrinsic, depth, base2camera=np.eye(4)):
    pixel_homo = np.array([pixel_coord[0], pixel_coord[1], 1])
    camera_norm = np.linalg.inv(intrinsic) @ pixel_homo
    camera_3d = camera_norm * depth
    camera_3d_homo = np.append(camera_3d, 1)
    base_3d = base2camera @ camera_3d_homo
    return base_3d[:3]

def detect_line(frame, is_curve=False, block_imshow=False):
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

        if is_curve:
            return _curve_line_detection(frame, dilated, mask)
        else:
            return _straight_line_detection(frame, dilated, mask)
    
    return frame, None


def _straight_line_detection(frame, dilated, mask):
    def _interpolate(x1, y1, x2, y2, num_points=10):
        index = np.linspace(0, 1, num_points)
        interpolated_coords = [
            (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)) for t in index
        ]
        return interpolated_coords

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
        interpolated_coords = _interpolate(x1, y1, x2, y2, 10)
        # Magenta dots (2D visualization)
        for px, py in interpolated_coords:
            cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
        # print the result
        print(f"Start Point: ({x1}, {y1}) End Point: ({x2}, {y2})")
        # print("Interpolated 3D points:")
        # for i, pt in enumerate(points_3D):
        #     print(f"Point {i+1}: {pt}")
        return frame, interpolated_coords


def _curve_line_detection(frame, dilated, mask):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = dilated.shape
    valid_contours = [cnt.squeeze() for cnt in contours if cnt.squeeze().ndim == 2 and len(cnt.squeeze()) >= 10
                      and not (np.any(cnt.squeeze()[:, 0] <= 5) or np.any(cnt.squeeze()[:, 0] >= w - 5)
                               or np.any(cnt.squeeze()[:, 1] <= 5) or np.any(cnt.squeeze()[:, 1] >= h - 5))]

    if not valid_contours:
        print("No valid contours found.")
        return frame, None

    curve = max(valid_contours, key=lambda c: len(c))
    # cv2.drawContours(frame, [curve], -1, (0, 255, 0), 2)  # 可视化轮廓
    # cv2.imshow("Curve Detection Result", frame)
    curve_reduced = curve[::5]

    _time_start = time.time()
    max_dist = 0
    start_pt, end_pt = None, None
    for i in range(len(curve_reduced)):
        for j in range(i + 1, len(curve_reduced)):
            dist = np.linalg.norm(curve_reduced[i] - curve_reduced[j])
            if dist > max_dist:
                max_dist = dist
                start_pt, end_pt = curve_reduced[i], curve_reduced[j]
    print(f"Time taken for finding start and end points: {time.time() - _time_start} seconds.")

    def sort_contour_by_proximity(contour, start, end):
        sorted_pts = [start.tolist()]
        remaining = contour.tolist()
        remaining.remove(start.tolist())
        while remaining:
            last = sorted_pts[-1]
            dists = [np.linalg.norm(np.array(last) - np.array(p)) for p in remaining]
            nearest = remaining[np.argmin(dists)]
            sorted_pts.append(nearest)
            remaining.remove(nearest)
            if np.allclose(nearest, end.tolist(), atol=5.0):
                break
        return np.array(sorted_pts)

    _time_start = time.time()
    sorted_curve = sort_contour_by_proximity(curve_reduced, start_pt, end_pt)
    print(f"Time taken for sorting contour: {time.time() - _time_start} seconds.")

    num_points = 50
    pts = sorted_curve[::max(1, len(sorted_curve) // num_points)]
    # # 可视化 pts
    # num_pts = len(pts)
    # for i, pt in enumerate(pts):
    #     color = (
    #         int(255 * (i / num_pts)),
    #         int(255 * (i / num_pts)),
    #         255 - int(255 * (i / num_pts)),
    #     )  # 渐变颜色
    #     cv2.circle(frame, (pt[0], pt[1]), 5, color, -1)
    # cv2.imshow("Sorted Curve", frame)

    x, y = pts[:, 0], pts[:, 1]
    if len(x) < 4:
        print("Less than 4 points for interpolation. Failed.")
        return []

    try:
        # default for 3-order B-spline
        tck, u = splprep([x, y], s=5.0, per=False)  # s for smoothing factor, per for periodic
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
    except Exception as e:
        print(f"Curve interpolation failed: {e}")
        return []

    interpolated_pts = np.array([[xi, yi] for xi, yi in zip(x_new, y_new)], dtype=np.float32)
    # 可视化 interpolated_pts
    for i, pt in enumerate(interpolated_pts):
        color = (
            int(255 * (i / num_points)),
            int(255 * (i / num_points)),
            255 - int(255 * (i / num_points)),
        )  # 渐变颜色
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
    return frame, interpolated_pts


if __name__ == "__main__":
    # Open camera (change index if needed)
    # cap = cv2.VideoCapture(0)

    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     break
        # frame = cv2.imread("lab4/example_straight.png")
        # frame, _ = detect_line(frame, is_curve=False)
        frame = cv2.imread("lab4/example_curve.png")
        frame, _ = detect_line(frame, is_curve=True)
        cv2.imshow('Processed Line Detection', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()
