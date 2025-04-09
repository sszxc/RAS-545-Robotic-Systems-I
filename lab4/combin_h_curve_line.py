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
    # 圖片預處理：透視轉換、灰階化、高斯模糊
    warped = cv2.warpPerspective(frame, H_matrix, (300, 300))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return warped, blurred


def convert_to_world_coords(points, H_matrix, fixed_z, visualize_img=None):
    # 像素座標轉換成世界座標
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

    if visualize_img is not None:
        for [[x, y]] in points:
            cv2.circle(visualize_img, (int(x), int(y)), 4, (0, 0, 255), -1)

    return [(float(x) + robot_offset_x, float(y) + robot_offset_y, float(fixed_z)) for [[x, y]] in world_coords]


def print_world_coords(label, world_coords):
    print(f"\n🧩 {label} 偵測結果：")
    for i, (x, y, z) in enumerate(world_coords):
        print(f"點 {i}: ({x:.2f}, {y:.2f}, {z:.1f})")


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
        world_coords = convert_to_world_coords(points, H_matrix, fixed_z, visualize_img=warped)

        # 視覺化結果
        cv2.imshow("Line Detection Result", warped)
        cv2.waitKey(1)

        print_world_coords("直線", world_coords)
        return world_coords

    print("❌ 未偵測到有效直線。")
    return []


# ==================== CURVE DETECTION ====================
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
    world_coords = convert_to_world_coords(interpolated_pts, H_matrix, fixed_z, visualize_img=warped)

    # 視覺化結果
    cv2.imshow("Curve Detection Result", warped)
    cv2.waitKey(1)

    print_world_coords("曲線", world_coords)
    return world_coords


# ==================== 主程式測試區 ====================
if __name__ == "__main__":
    # ======== 先前：即時相機讀取，改為備註保留 ========
    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()
    # if not ret:
    #     print("❌ 無法讀取相機影像。")
    #     cap.release()
    #     exit()

    # ======== 新增：讀取圖片模式 ========
    img = cv2.imread("2.png")  # 請更換為你的圖片檔案名稱
    if img is None:
        print("❌ 無法讀取圖片，請確認檔案路徑和名稱是否正確。")
        exit()

    # 顯示讀取的圖片
    cv2.imshow("Loaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 初始化 H matrix
    H_matrix = initialize_H_matrix(img)

    # 偵測直線
    coords_line = detect_line(img, straight_or_curve=True, H_matrix=H_matrix)

    # 偵測曲線
    coords_curve = detect_line(img, straight_or_curve=False, H_matrix=H_matrix)

    cv2.destroyAllWindows()
