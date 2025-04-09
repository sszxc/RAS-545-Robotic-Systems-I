import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def get_curve_path_from_workspace(fixed_z=300, num_points=20):
    # === 1. CAPTURE FRAME FROM CAMERA ===
    # Open the camera and wait for the user to press the space key to capture a frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera.")
        return []

    print("🎥 Showing live feed. Press SPACE to capture workspace frame.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            return []
        cv2.imshow("Live Feed - Press SPACE to Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            workspace_frame = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()

    # === 2. MANUAL SELECTION OF 4 CORNERS ===
    # Prompt user to click four corners of the workspace in a specific order. Note order below
    print("👉 Click 4 corners in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    selected_corners = []

    # Mouse callback function to capture corner clicks
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected_corners) < 4:
            selected_corners.append([x, y])
            cv2.circle(workspace_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click Workspace Corners", workspace_frame)

    # Show image and set mouse callback
    cv2.imshow("Click Workspace Corners", workspace_frame)
    cv2.setMouseCallback("Click Workspace Corners", mouse_callback)
    while len(selected_corners) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # === 3. PERSPECTIVE TRANSFORMATION SETUP ===
    # Map selected corners to a 300x300 destination image using a warp matrix
    src_pts = np.array(selected_corners, dtype=np.float32)
    dst_size = (300, 300)
    dst_pts = np.array([
        [0, 0],
        [dst_size[0]-1, 0],
        [dst_size[0]-1, dst_size[1]-1],
        [0, dst_size[1]-1]
    ], dtype=np.float32)

    print("\n📍 Clicked Corners (Pixel Coordinates):")
    for i, pt in enumerate(selected_corners):
        print(f"Corner {i+1}: Pixel=({pt[0]}, {pt[1]})")

    # Compute perspective transform matrices
    warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inv_warp_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped_workspace = cv2.warpPerspective(workspace_frame, warp_matrix, dst_size)

    # === 4. CURVE EXTRACTION FROM IMAGE ===
    # Convert warped image to binary for contour detection
    gray = cv2.cvtColor(warped_workspace, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Find and filter contours within the workspace
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = binary.shape
    valid_contours = []

    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 10:
            continue
        if np.any(cnt[:, 0] <= 5) or np.any(cnt[:, 0] >= w - 5) or \
           np.any(cnt[:, 1] <= 5) or np.any(cnt[:, 1] >= h - 5):
            continue
        valid_contours.append(cnt)

    if not valid_contours:
        print("❌ No valid curve detected inside workspace.")
        return []

    # Select the longest valid contour (assumed to be the curve)
    curve = max(valid_contours, key=lambda c: len(c))

    # === 5. FIND START AND END POINTS ===
    # Find the two farthest points on the curve
    max_dist = 0
    start_pt, end_pt = None, None
    for i in range(len(curve)):
        for j in range(i+1, len(curve)):
            dist = np.linalg.norm(curve[i] - curve[j])
            if dist > max_dist:
                max_dist = dist
                start_pt, end_pt = curve[i], curve[j]

    # === 6. SORT CONTOUR POINTS FROM START TO END ===
    # Sort contour points from start to end using proximity
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
    pts = sorted_curve[::10]  # Subsample points for interpolation

    # === 7. INTERPOLATE THE CURVE (SPLINE FIT) ===
    x, y = pts[:, 0], pts[:, 1]
    if len(x) < 4:
        print("❌ Not enough points for interpolation.")
        return []

    try:
        tck, u = splprep([x, y], s=5.0, per=False)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
    except Exception as e:
        print(f"❌ Interpolation failed: {e}")
        return []

    # === 8. TRANSFORM INTERPOLATED POINTS TO ORIGINAL SPACE ===
    interpolated_pts = np.array([[[xi, yi]] for xi, yi in zip(x_new, y_new)], dtype=np.float32)
    original_pixels = cv2.perspectiveTransform(interpolated_pts, inv_warp_matrix)

    # === 9. CONVERT TO REAL-WORLD COORDINATES ===
    world_plane_pts = np.array([[0, 0], [150, 0], [150, 150], [0, 150]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, world_plane_pts)

    print("\n📐 Homography Matrix (Pixels → Real World):")
    print(np.array2string(H, precision=3, suppress_small=True))

    world_coords = cv2.perspectiveTransform(original_pixels, H)
    world_coords = [(float(x), float(y), float(fixed_z)) for [[x, y]] in world_coords]

    # === 10. PRINT PIXEL AND WORLD COORDINATES ===
    print("\n📍 Interpolated Start & End (Pixel Coordinates):")
    print(f"START Pixel: ({x_new[0]:.1f}, {y_new[0]:.1f})")
    print(f"END   Pixel: ({x_new[-1]:.1f}, {y_new[-1]:.1f})")

    print("\n🌍 World Coordinates for Start & End (before offset):")
    print(f"START World: ({world_coords[0][0]:.2f}, {world_coords[0][1]:.2f}, {world_coords[0][2]:.1f})")
    print(f"END   World: ({world_coords[-1][0]:.2f}, {world_coords[-1][1]:.2f}, {world_coords[-1][2]:.1f})")

    print("\n🧭 Pixel Coordinates → World Coordinates:")
    for i, ([[px, py]], (wx, wy, wz)) in enumerate(zip(original_pixels, world_coords)):
        label = "START" if i == 0 else "END" if i == len(world_coords)-1 else f"Point {i}"
        print(f"{label}: Pixel=({px:.1f}, {py:.1f}) → World=({wx:.2f}, {wy:.2f}, {wz:.1f})")

    # === ✅ APPLY ROBOT OFFSET HERE ===
    # Paper is placed at this distance (X mm, Y mm) in the robot's world frame
    robot_offset_x = -235.8
    robot_offset_y = -194.5
    world_coords = [(x + robot_offset_x, y + robot_offset_y, z) for (x, y, z) in world_coords]

    print(f"\n🔧 World Coordinates AFTER applying robot offset {robot_offset_x} X, {robot_offset_y} Y):")
    for i, (x, y, z) in enumerate(world_coords):
        label = "START" if i == 0 else "END" if i == len(world_coords)-1 else f"Point {i}"
        print(f"{label}: World=({x:.2f}, {y:.2f}, {z:.1f})")

    # === 11. DRAW VISUALIZATION (CURVE, LABELS) ===
    for i, (x, y) in enumerate(zip(x_new, y_new)):
        cv2.circle(warped_workspace, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.putText(warped_workspace, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    start = (int(x_new[0]), int(y_new[0]))
    end = (int(x_new[-1]), int(y_new[-1]))

    cv2.circle(warped_workspace, start, 6, (0, 255, 0), 2)
    cv2.putText(warped_workspace, "Start", (start[0]+5, start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.circle(warped_workspace, end, 6, (0, 0, 255), 2)
    cv2.putText(warped_workspace, "End", (end[0]+5, end[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Warped Curve Detection", warped_workspace)
    print("\n✅ Press any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === 12. RETURN FINAL REAL-WORLD COORDINATES ===
    return world_coords


# === RUN FUNCTION ===
coords = get_curve_path_from_workspace()


# === OUTPUT FORMAT ===
# The final coordinates returned by this function are in real-world units (millimeters).
# Each point is a 3D tuple: (x, y, z), where:
# - x and y are based on the homography mapping from image to workspace
# - z is a fixed value defined by the variable `fixed_z`
#
# ASSUMPTIONS AND THINGS TO CHANGE FOR ROBOT:
#
# 1. world_plane_pts:
#    This is set to a 150mm x 150mm workspace:
#    [[0, 0], [150, 0], [150, 150], [0, 150]]
#    👉 Change these values if your workspace is a different size.
#
# 2. fixed_z:
#    This defines the vertical height (Z) the robot will follow.
#    👉 fixed_z = 300 mm (or whatever keeps your end-effector safe)
#
# 3. Robot Origin Alignment:
#    👉 Add an (x, y) offset to all coordinates to place them into robot space
#    👉 Example: paper is at (-235.8, -194.5) mm → offset all coordinates
#
# 4. Units:
#    👉 Coordinates are in mm. Convert to meters if your robot expects it.
