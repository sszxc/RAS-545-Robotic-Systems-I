import cv2
import numpy as np
import math


def is_black_region(img, center, half_size, black_thresh=100):
    """
    判断以 center 为中心的正方形区域内，黑色像素所占比例是否足够大。
    参数:
      img: 原始灰度图 (H, W)。
      center: (cy, cx)，图像坐标 (y在前, x在后)。
      half_size: (ly, lx), 方框半边长像素。
      black_thresh: 黑色阈值（灰度值小于该值视为黑）。
    返回:
      True / False，表示该区域是否“足够黑”(可以视为有墙)。
    """
    H, W = img.shape
    cy, cx = center
    ly, lx = half_size
    y1 = int(max(0, cy - ly))
    y2 = int(min(H, cy + ly))
    x1 = int(max(0, cx - lx))
    x2 = int(min(W, cx + lx))

    # 可视化这个选定的区域
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("img", img_bgr)
    # cv2.waitKey(0)

    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return False

    # 统计黑色像素数量
    black_pixels = np.sum(region < black_thresh)
    ratio = black_pixels / region.size

    # 这里阈值可以根据需要调节，比如说黑色像素占比超过 50% 就认为是有墙
    if ratio > 0.5:
        return True
    return False


def find_color_center(img_bgr, lower_bound, upper_bound):
    """
    在 BGR 图像中寻找特定颜色区域(通过 inRange)，返回该颜色区域的质心坐标。
    若找到多个区域，则返回所有像素的平均坐标。
    lower_bound 和 upper_bound 是 BGR 范围。
    返回 (cy, cx)，即在图像中的行、列坐标(浮点)；若未找到则返回 None。
    """
    mask = cv2.inRange(img_bgr, lower_bound, upper_bound)
    points = np.argwhere(mask > 0)
    if len(points) == 0:
        return None
    cy, cx = np.mean(points, axis=0)  # 求所有像素的平均值
    return (cy, cx)


def nearest_grid_node(cy, cx, cell_h, cell_w, row_count, col_count):
    """
    给定图像坐标 (cy, cx)，以及网格尺寸，
    找到与其距离最近的网格角点 (r, c)，其中 r ∈ [0, row_count], c ∈ [0, col_count]。
    """
    # 将图像坐标转换到“格点坐标系”
    fr = cy / cell_h - 0.5
    fc = cx / cell_w - 0.5
    # 四舍五入或就近寻找
    r_near = int(round(fr))
    c_near = int(round(fc))
    # 边界裁剪
    r_near = max(0, min(row_count, r_near))
    c_near = max(0, min(col_count, c_near))
    return (r_near, c_near)


def dfs_all_paths(graph, start, end):
    """
    在无向图 graph 中，从 start 到 end 做 DFS，找到所有可行路径。
    graph 的表示方式: graph[(r,c)] = [(r1,c1), (r2,c2), ...] 表示可达邻居。
    返回所有路径的列表，每条路径都是格点坐标的列表。
    """
    stack = [(start, [start])]  # (当前节点, 到当前节点的路径)
    all_paths = []
    visited_global = set()  # 全局去重(不一定需要，可以去掉以获取更多分支)

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
    在原图上用给定颜色在相邻格点之间画线。
    path: [(r0,c0), (r1,c1), ...]
    """
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        # 转换到图像像素坐标 (只要取格点中心或 corner 即可)
        y0 = int((r0 + 0.5) * cell_h)
        x0 = int((c0 + 0.5) * cell_w)
        y1 = int((r1 + 0.5) * cell_h)
        x1 = int((c1 + 0.5) * cell_w)
        cv2.line(img_bgr, (x0, y0), (x1, y1), color, width)


if __name__ == "__main__":
    # === 1) 读取图像 ===#
    img_path = "lab5/maze/0.png"  # 这里替换成实际路径
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("无法读取图像，请检查路径！")
        exit(1)

    # 转灰度图，以便判断黑白
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    H, W = img_gray.shape

    # === 2) 定义网格行列数(题目给出) ===#
    #   假设题目告诉我们行数和列数:
    row_count = 5  # 比如 5 行
    col_count = 5  # 比如 5 列
    # 具体值要根据实际迷宫分块来定

    # === 3) 计算每个格子的像素边长 ===#
    cell_h = H / row_count
    cell_w = W / col_count

    # === 4) 构造无向图，判断哪些相邻角点可以连通 ===#
    #   角点总共有 (row_count+1)*(col_count+1) 个, 每个角点可记为 (r, c)
    graph = {}

    def add_edge(a, b):
        """给 a->b 和 b->a 建立无向连边"""
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

    # 判断 (r, c) 和 (r+1, c) 是否可通
    for r in range(row_count - 1):
        for c in range(col_count):
            # 找到这两个角点的像素坐标中点
            mid_y = (r + 1) * cell_h
            mid_x = (c + 0.5) * cell_w
            ly = 0.1 * cell_h
            lx = 0.4 * cell_w
            print(f"Checking edge {(r, c)} -> {(r + 1, c)}", end="")
            # 判断是否是墙
            if not is_black_region(img_gray, (mid_y, mid_x), (ly, lx)):
                add_edge((r, c), (r + 1, c))
                print(f"\033[94m -> No wall\033[0m")
            else:
                print(f"\033[91m -> Wall\033[0m")

    # 判断 (r, c) 和 (r, c+1) 是否可通
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

    # === 5) 根据红色/绿色像素找到起点终点对应格点 ===#
    #   这里用 BGR 阈值做一个简单的例子，比如红色 (0,0, ~255)，绿色 (~0,255,0)
    #   请根据实际情况调整上下界
    #   lowerBound / upperBound 的顺序是 (B, G, R)
    red_lower = np.array([0, 0, 150])
    red_upper = np.array([80, 80, 255])
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([80, 255, 80])

    red_center = find_color_center(img_bgr, red_lower, red_upper)
    green_center = find_color_center(img_bgr, green_lower, green_upper)

    if red_center is None or green_center is None:
        print("没有检测到红/绿圆点，请检查颜色范围或者图片内容")
        exit(1)

    start_node = nearest_grid_node(
        red_center[0], red_center[1], cell_h, cell_w, row_count, col_count
    )
    end_node = nearest_grid_node(
        green_center[0], green_center[1], cell_h, cell_w, row_count, col_count
    )

    print("起点节点:", start_node)
    print("终点节点:", end_node)

    # === 6) 用 DFS 找到所有可行解，再找最短路径（可能有多条）===#
    all_paths = dfs_all_paths(graph, start_node, end_node)
    if not all_paths:
        print("未找到任何可行路径")
        exit(0)

    # 找出最短路径长度
    min_len = min(len(p) for p in all_paths)
    # 所有最短路径
    shortest_paths = [p for p in all_paths if len(p) == min_len]
    other_paths = [p for p in all_paths if len(p) != min_len]

    print(
        f"所有可行路径数: {len(all_paths)}，最短路径长度: {min_len}，最短路径条数: {len(shortest_paths)}"
    )

    # === 7) 在原图上用不同深浅的蓝色绘制所有最短路径 ===#

    for idx, path in enumerate(other_paths):
        # 生成一个随机的特别浅的颜色
        r = np.random.randint(200, 255)
        g = np.random.randint(200, 255)
        b = np.random.randint(200, 255)
        color = (b, g, r)
        draw_path_on_image(img_bgr, path, color)

    #   可以根据路径数量来分配颜色深浅，这里做一个简单示例
    num_sp = len(shortest_paths)
    for idx, path in enumerate(shortest_paths):
        # 生成一个从浅蓝到深蓝的 BGR 颜色
        alpha = idx / max(1, (num_sp - 1))  # [0,1] 之间
        # 浅蓝 (200, 150, 50)，深蓝 (255, 0, 0) 这里随意演示
        b = int(200 + alpha * (255 - 200))
        g = int(150 + alpha * (0 - 150))
        r = int(50 + alpha * (0 - 50))
        color = (b, g, r)
        draw_path_on_image(img_bgr, path, color, width=4)

    # === 结果输出或显示 ===#
    cv2.imwrite("lab5/maze_solved.png", img_bgr)
    print("结果已写入 maze_solved.png，你可以查看其中的蓝色路径。")
