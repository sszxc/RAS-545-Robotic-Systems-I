import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple

# ==================== H Matrix Initialization ====================
def initialize_H_matrix(frame, workspace_size=(300, 300)) -> np.ndarray:
    print("👉 請依次點擊四個角落：左上、右上、右下、左下")
    selected_corners = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected_corners) < 4:
            selected_corners.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("選取工作區域四個角落", frame)

    cv2.imshow("選取工作區域四個角落", frame)
    cv2.setMouseCallback("選取工作區域四個角落", mouse_callback)
    while len(selected_corners) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    src_pts = np.array(selected_corners, dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [workspace_size[0] - 1, 0],
        [workspace_size[0] - 1, workspace_size[1] - 1],
        [0, workspace_size[1] - 1]
    ], dtype=np.float32)

    H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("\n✅ H Matrix 初始化完成：")
    print(H_matrix)
    return H_matrix

# ==================== Utility Functions ====================
def interpolate(x1, y1, x2, y2, num_points=10):
    index = np.linspace(0, 1, num_points)
    interpolated_coords = [
        (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)) for t in index
    ]
    return interpolated_coords


def preprocess_image(frame, H_matrix):
    warped = cv2.warpPerspective(frame, H_matrix, (300, 300))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return warped, blurred


def convert_to_world_coords(points, H_matrix, fixed_z, visualize_img=None, labels=None):
    dst_size = (300, 300)
    dst_pts = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype=np.float32)

    world_plane_pts = np.array([[0, 0], [150, 0], [150, 150], [0, 150]], dtype=np.float32)
    H_world, _ = cv2.findHomography(dst_pts, world_plane_pts)
    world_coords = cv2.perspectiveTransform(points, H_world)

    robot_offset_x = -235.8
    robot_offset_y = -194.5

    result = []
    for i, ([[x, y]]) in enumerate(points):
        if visualize_img is not None:
            cv2.circle(visualize_img, (int(x), int(y)), 4, (255, 0, 0), -1)
            if labels:
                cv2.putText(visualize_img, labels[i], (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        wx, wy = world_coords[i][0]
        wx += robot_offset_x
        wy += robot_offset_y
        result.append((wx, wy, fixed_z))

    return result


def print_world_coords(label, world_coords):
    print(f"\n🧩 {label} 偵測結果：")
    for i, (x, y, z) in enumerate(world_coords):
        point_label = "START" if i == 0 else "END" if i == len(world_coords) - 1 else f"Point {i}"
        print(f"{point_label}: ({x:.2f}, {y:.2f}, {z:.1f})")


# ==================== General Detect Line Function ====================
def detect_line(img: np.ndarray, straight_or_curve: bool, H_matrix: np.ndarray, num_points: int = 20, fixed_z: float = 300) -> List[Tuple[float, float, float]]:
    if straight_or_curve:
        return detect_line_logic(img, H_matrix, num_points, fixed_z)
    else:
        return detect_curve_logic(img, H_matrix, num_points, fixed_z)


# ==================== LINE DETECTION ====================
def detect_line_logic(frame, H_matrix, num_points, fixed_z):
    warped, preprocessed = preprocess_image(frame, H_matrix)

    _, binary = cv2.threshold(preprocessed, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    dilated = cv2.bitwise_not(dilated)

    lines = cv2.HoughLinesP(
        dilated, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is not None:
        longest_line = max(lines, key=lambda line: np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1]))
        x1, y1, x2, y2 = longest_line[0]
        interpolated_coords = interpolate(x1, y1, x2, y2, num_points)

        points = np.array([[[px, py]] for (px, py) in interpolated_coords], dtype=np.float32)
        labels = ["START"] + [f"Point {i}" for i in range(1, len(points) - 1)] + ["END"]
        world_coords = convert_to_world_coords(points, H_matrix, fixed_z, visualize_img=warped, labels=labels)

        cv2.imshow("Line Detection Result", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print_world_coords("直線", world_coords)
        return world_coords

    print("❌ 未偵測到有效直線。")
    return []


# ==================== CURVE DETECTION (根據你的要求完整優化版) ====================
def detect_curve_logic(frame, H_matrix, num_points, fixed_z):
    warped, preprocessed = preprocess_image(frame, H_matrix)

    binary = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=11, C=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = binary.shape
    valid_contours = [cnt.squeeze() for cnt in contours if cnt.squeeze().ndim == 2 and len(cnt.squeeze()) >= 10
                      and not (np.any(cnt.squeeze()[:, 0] <= 5) or np.any(cnt.squeeze()[:, 0] >= w - 5)
                               or np.any(cnt.squeeze()[:, 1] <= 5) or np.any(cnt.squeeze()[:, 1] >= h - 5))]

    if not valid_contours:
        print("❌ 未偵測到有效曲線。")
        return []

    curve = max(valid_contours, key=lambda c: len(c))

    max_dist = 0
    start_pt, end_pt = None, None
    for i in range(len(curve)):
        for j in range(i + 1, len(curve)):
            dist = np.linalg.norm(curve[i] - curve[j])
            if dist > max_dist:
                max_dist = dist
                start_pt, end_pt = curve[i], curve[j]

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

    sorted_curve = sort_contour_by_proximity(curve, start_pt, end_pt)
    pts = sorted_curve[::max(1, len(sorted_curve) // num_points)]

    x, y = pts[:, 0], pts[:, 1]
    if len(x) < 4:
        print("❌ 插值點不足，無法進行曲線擬合。")
        return []

    try:
        tck, u = splprep([x, y], s=5.0, per=False)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
    except Exception as e:
        print(f"❌ 曲線插值失敗：{e}")
        return []

    interpolated_pts = np.array([[[xi, yi]] for xi, yi in zip(x_new, y_new)], dtype=np.float32)
    labels = ["START"] + [f"Point {i}" for i in range(1, len(interpolated_pts) - 1)] + ["END"]
    world_coords = convert_to_world_coords(interpolated_pts, H_matrix, fixed_z, visualize_img=warped, labels=labels)

    cv2.imshow("Curve Detection Result", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print_world_coords("曲線", world_coords)
    return world_coords


# ==================== 主程式測試區 ====================
if __name__ == "__main__":
    img = cv2.imread("lab3/example_curve.png")
    if img is None:
        print("❌ 無法讀取圖片，請確認檔案路徑和名稱是否正確。")
        exit()

    cv2.imshow("Loaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H_matrix = initialize_H_matrix(img)

    coords_line = detect_line(img, straight_or_curve=True, H_matrix=H_matrix)
    coords_curve = detect_line(img, straight_or_curve=False, H_matrix=H_matrix)

    cv2.destroyAllWindows()


# ✅ 確認一下輸出
if coords_line:
    print_world_coords("直線", coords_line)
if coords_curve:
    print_world_coords("曲線", coords_curve)

# ✅ 整理成 final_pts
final_pts = []
if coords_line:
    final_pts.extend(coords_line)
if coords_curve:
    final_pts.extend(coords_curve)

# ✅ 印出最後要存的 final_pts 確認一下
print("✅ 最終存入 final_pts.npy 的內容：")
for pt in final_pts:
    print(pt)

# ✅ 轉成公尺（m 單位）
final_pts_m = np.array(final_pts) * 0.001
print("✅ 最終存入 final_pts.npy 的內容（m 單位）：")
for pt in final_pts_m:
    print(pt)

# ✅ 存成 .npy 檔案，給主程式使用
np.save("lab3/final_pts.npy", np.array(final_pts))
np.save("lab3/H_pixel2world.npy", H_matrix)
print("✅ H matrix 已經成功儲存到 lab3/H_pixel2world.npy")
print("✅ 路徑點已經成功儲存到 lab3/final_pts.npy")
