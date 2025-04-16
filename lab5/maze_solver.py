import cv2
import numpy as np
import math


def is_black_region(img, center, half_size, black_thresh=100):
    """
    Determine if the proportion of black pixels in a square region centered at 'center' is large enough.
    Parameters:
      img: Original grayscale image (H, W).
      center: (cy, cx), image coordinates (y first, x second).
      half_size: (ly, lx), half size of the box in pixels.
      black_thresh: Black threshold (pixels with grayscale value less than this are considered black).
    Returns:
      True / False, indicating whether the region is "black enough" (can be considered as a wall).
    """
    H, W = img.shape
    cy, cx = center
    ly, lx = half_size
    y1 = int(max(0, cy - ly))
    y2 = int(min(H, cy + ly))
    x1 = int(max(0, cx - lx))
    x2 = int(min(W, cx + lx))

    # Visualize the selected region
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("img", img_bgr)
    # cv2.waitKey(0)

    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return False

    # Count the number of black pixels
    black_pixels = np.sum(region < black_thresh)
    ratio = black_pixels / region.size

    # The threshold can be adjusted as needed, e.g., consider it a wall if black pixels exceed 50%
    if ratio > 0.5:
        return True
    return False


def find_color_center(img_bgr, lower_bound, upper_bound):
    """
    Find a specific color region in BGR image (using inRange), and return the centroid of this color region.
    If multiple regions are found, return the average coordinates of all pixels.
    lower_bound and upper_bound are BGR ranges.
    Return (cy, cx), i.e., row and column coordinates in the image (float); return None if not found.
    """
    mask = cv2.inRange(img_bgr, lower_bound, upper_bound)
    points = np.argwhere(mask > 0)
    if len(points) == 0:
        return None
    cy, cx = np.mean(points, axis=0)  # Calculate average of all pixels
    return (cy, cx)


def nearest_grid_node(cy, cx, cell_h, cell_w, row_count, col_count):
    """
    Given image coordinates (cy, cx) and grid dimensions,
    find the nearest grid corner node (r, c), where r ∈ [0, row_count], c ∈ [0, col_count].
    """
    # Convert image coordinates to "grid coordinate system"
    fr = cy / cell_h - 0.5
    fc = cx / cell_w - 0.5
    # Round to find the nearest point
    r_near = int(round(fr))
    c_near = int(round(fc))
    # Boundary clipping
    r_near = max(0, min(row_count, r_near))
    c_near = max(0, min(col_count, c_near))
    return (r_near, c_near)


def dfs_all_paths(graph, start, end):
    """
    In an undirected graph 'graph', perform DFS from 'start' to 'end', finding all feasible paths.
    graph representation: graph[(r,c)] = [(r1,c1), (r2,c2), ...] indicating reachable neighbors.
    Returns a list of all paths, each path being a list of grid coordinates.
    """
    stack = [(start, [start])]  # (current node, path to current node)
    all_paths = []
    visited_global = set()  # Global deduplication (not necessarily needed, can be removed to get more branches)

    while stack:
        node, path = stack.pop()
        if node == end:
            all_paths.append(path)
            continue

        for nxt in graph[node]:
            if nxt not in path:
                new_path = path + [nxt]
                stack.append((nxt, new_path))

    return all_paths


def draw_path_on_image(img_bgr, path, color, width=2):
    """
    Draw lines between adjacent grid points on the original image with the given color.
    path: [(r0,c0), (r1,c1), ...]
    """
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        # Convert to image pixel coordinates (just take the center or corner of the grid points)
        y0 = int((r0 + 0.5) * cell_h)
        x0 = int((c0 + 0.5) * cell_w)
        y1 = int((r1 + 0.5) * cell_h)
        x1 = int((c1 + 0.5) * cell_w)
        cv2.line(img_bgr, (x0, y0), (x1, y1), color, width)


if __name__ == "__main__":
    # === 1) Read image ===#
    img_path = "lab5/maze/0.png"
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Cannot read image, please check the path!")
        exit(1)

    # Convert to grayscale for black and white detection
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    H, W = img_gray.shape

    # === 2) Define grid rows and columns (given by the problem) ===#
    row_count = 5  # For example, 5 rows
    col_count = 5  # For example, 5 columns

    # === 3) Calculate pixel size of each cell ===#
    cell_h = H / row_count
    cell_w = W / col_count

    # === 4) Construct undirected graph, determine which adjacent corners can be connected ===#
    graph = {}

    def add_edge(a, b):
        """Establish undirected edge between a->b and b->a"""
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

    # Determine if (r, c) and (r+1, c) are connected
    for r in range(row_count - 1):
        for c in range(col_count):
            # Find the pixel midpoint between these two corners
            mid_y = (r + 1) * cell_h
            mid_x = (c + 0.5) * cell_w
            ly = 0.1 * cell_h
            lx = 0.4 * cell_w
            print(f"Checking edge {(r, c)} -> {(r + 1, c)}", end="")
            # Determine if it is a wall
            if not is_black_region(img_gray, (mid_y, mid_x), (ly, lx)):
                add_edge((r, c), (r + 1, c))
                print(f"\033[94m -> No wall\033[0m")
            else:
                print(f"\033[91m -> Wall\033[0m")

    # Determine if (r, c) and (r, c+1) are connected
    for r in range(row_count):
        for c in range(col_count - 1):
            mid_y = (r + 0.5) * cell_h
            mid_x = (c + 1) * cell_w
            ly = 0.4 * cell_h
            lx = 0.1 * cell_w
            print(f"Checking edge {(r, c)} -> {(r, c + 1)}", end="")
            if not is_black_region(img_gray, (mid_y, mid_x), (ly, lx)):
                add_edge((r, c), (r, c + 1))
                print(f"\033[94m -> No wall\033[0m")
            else:
                print(f"\033[91m -> Wall\033[0m")

    # === 5) Find start and end grid points based on red/green pixels ===#
    #   Here we use BGR thresholds as a simple example, like red (0,0,~255), green (~0,255,0)
    #   The order of lowerBound / upperBound is (B, G, R)
    red_lower = np.array([0, 0, 150])
    red_upper = np.array([80, 80, 255])
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([80, 255, 80])

    red_center = find_color_center(img_bgr, red_lower, red_upper)
    green_center = find_color_center(img_bgr, green_lower, green_upper)

    if red_center is None or green_center is None:
        print("No red/green circles detected, please check color range or image content")
        exit(1)

    start_node = nearest_grid_node(
        red_center[0], red_center[1], cell_h, cell_w, row_count, col_count
    )
    end_node = nearest_grid_node(
        green_center[0], green_center[1], cell_h, cell_w, row_count, col_count
    )

    print("Start node:", start_node)
    print("End node:", end_node)

    # === 6) Use DFS to find all possible solutions, then find the shortest path (may be multiple) ===#
    all_paths = dfs_all_paths(graph, start_node, end_node)
    if not all_paths:
        print("No feasible paths found")
        exit(0)

    # Find the shortest path length
    min_len = min(len(p) for p in all_paths)
    # All shortest paths
    shortest_paths = [p for p in all_paths if len(p) == min_len]
    other_paths = [p for p in all_paths if len(p) != min_len]

    print(
        f"Number of all feasible paths: {len(all_paths)}, shortest path length: {min_len}, number of shortest paths: {len(shortest_paths)}"
    )

    # === 7) Draw all shortest paths on the original image with different shades of blue ===#

    for idx, path in enumerate(other_paths):
        # Generate a random very light color
        r = np.random.randint(200, 255)
        g = np.random.randint(200, 255)
        b = np.random.randint(200, 255)
        color = (b, g, r)
        draw_path_on_image(img_bgr, path, color)

    #   Colors assigned based on the number of paths
    num_sp = len(shortest_paths)
    for idx, path in enumerate(shortest_paths):
        # Generate a BGR color from light blue to deep blue
        alpha = idx / max(1, (num_sp - 1))  # Between [0,1]
        # Light blue (200, 150, 50), deep blue (255, 0, 0)
        b = int(200 + alpha * (255 - 200))
        g = int(150 + alpha * (0 - 150))
        r = int(50 + alpha * (0 - 50))
        color = (b, g, r)
        draw_path_on_image(img_bgr, path, color, width=4)

    # === Output or display results ===#
    cv2.imwrite("lab5/maze_solved.png", img_bgr)
    print("Results written to maze_solved.png, you can check the blue paths in it.")
    cv2.imshow("img", img_bgr)
    cv2.waitKey(0)
