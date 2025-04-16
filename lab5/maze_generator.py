import random
from PIL import Image, ImageDraw
import math

# --- Configuration (Unchanged) ---
GRID_SIZE = 5
IMAGE_SIZE = 800
CELL_SIZE = IMAGE_SIZE // GRID_SIZE
WALL_COLOR = (0, 0, 0)  # Black
BACKGROUND_COLOR = (255, 255, 255)  # White
ENTRANCE_COLOR = (255, 0, 0)  # Red
EXIT_COLOR = (0, 128, 0)  # Green
LINE_WIDTH = max(4, CELL_SIZE // 6)  # Keep thick setting
NUM_EXTRA_WALLS_TO_REMOVE = 3


# --- Maze Generation (Modified DFS - unchanged) ---
def generate_maze(rows, cols):
    """Generates a maze grid using modified DFS. Returns connections."""
    # (Code is identical to the previous version - omitted for brevity)
    connections = {(r, c): set() for r in range(rows) for c in range(cols)}
    visited = set()
    stack = []
    potential_internal_walls = []
    start_cell = (0, 0)
    visited.add(start_cell)
    stack.append(start_cell)
    while stack:
        current_r, current_c = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = current_r + dr, current_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                wall = tuple(sorted(((current_r, current_c), (nr, nc))))
                if wall not in potential_internal_walls:
                    potential_internal_walls.append(wall)
                if (nr, nc) not in visited:
                    neighbors.append((nr, nc))
        if neighbors:
            next_r, next_c = random.choice(neighbors)
            connections[(current_r, current_c)].add((next_r, next_c))
            connections[(next_r, next_c)].add((current_r, current_c))
            visited.add((next_r, next_c))
            stack.append((next_r, next_c))
        else:
            stack.pop()
    removed_walls_count = 0
    possible_walls_to_remove = list(potential_internal_walls)
    random.shuffle(possible_walls_to_remove)
    for wall in possible_walls_to_remove:
        if removed_walls_count >= NUM_EXTRA_WALLS_TO_REMOVE:
            break
        (r1, c1), (r2, c2) = wall
        if (r2, c2) not in connections[(r1, c1)]:
            connections[(r1, c1)].add((r2, c2))
            connections[(r2, c2)].add((r1, c1))
            removed_walls_count += 1
    return connections


# --- Drawing (Unified Wall Drawing Strategy) ---


def draw_maze(
    connections, rows, cols, image_size, cell_size, filename="unsolved_maze.png"
):
    """Draws the maze using a unified approach for all walls."""
    img = Image.new("RGB", (image_size, image_size), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    entrance_cell = (0, 0)
    exit_cell = (rows - 1, cols - 1)
    wall_thickness = LINE_WIDTH
    walls_to_draw = set()  # Store walls as ('H'/'V', row_idx, col_idx)

    # 1. Determine which WALL SEGMENTS need to be drawn
    # Check connections between adjacent cells for INTERNAL walls
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            # Check wall below (South): Horizontal line at y=(r+1)*CS
            if r + 1 < rows:
                cell_S = (r + 1, c)
                if cell_S not in connections.get(cell, set()):
                    walls_to_draw.add(
                        ("H", r + 1, c)
                    )  # H line at grid row r+1, starting at grid col c

            # Check wall right (East): Vertical line at x=(c+1)*CS
            if c + 1 < cols:
                cell_E = (r, c + 1)
                if cell_E not in connections.get(cell, set()):
                    walls_to_draw.add(
                        ("V", r, c + 1)
                    )  # V line starting at grid row r, at grid col c+1

    # Check BOUNDARY walls (Add unless it's the entrance/exit opening)
    # Horizontal boundaries (Top: r_idx=0, Bottom: r_idx=rows)
    for c in range(cols):
        # Top boundary segment: H line at y=0 (row 0)
        if entrance_cell != (
            0,
            c,
        ):  # Add unless this segment is the entrance cell's top opening
            walls_to_draw.add(("H", 0, c))
        # Bottom boundary segment: H line at y=rows*CS (row rows)
        if exit_cell != (
            rows - 1,
            c,
        ):  # Add unless this segment is the exit cell's bottom opening
            walls_to_draw.add(("H", rows, c))

    # Vertical boundaries (Left: c_idx=0, Right: c_idx=cols)
    for r in range(rows):
        # Left boundary segment: V line at x=0 (col 0)
        if entrance_cell != (
            r,
            0,
        ):  # Add unless this segment is the entrance cell's left opening
            walls_to_draw.add(("V", r, 0))
        # Right boundary segment: V line at x=cols*CS (col cols)
        if exit_cell != (
            r,
            cols - 1,
        ):  # Add unless this segment is the exit cell's right opening
            walls_to_draw.add(("V", r, cols))

    # 2. Draw all required wall segments using slightly overlapping rectangles
    for wall_type, r_idx, c_idx in walls_to_draw:
        if wall_type == "H":  # Horizontal wall centered at y = r_idx * CS
            y_center = r_idx * cell_size
            x_start = c_idx * cell_size
            x_end = (c_idx + 1) * cell_size

            # Calculate coordinates for overlapping rectangle
            x1 = x_start - wall_thickness // 2  # Extend left for overlap
            y1 = y_center - wall_thickness // 2
            x2 = x_end + (wall_thickness + 1) // 2  # Extend right for overlap
            y2 = y_center + (wall_thickness + 1) // 2
            # Clip coordinates to image bounds to be safe
            x1, x2 = max(0, x1), min(image_size, x2)
            y1, y2 = max(0, y1), min(image_size, y2)
            draw.rectangle([x1, y1, x2, y2], fill=WALL_COLOR)

        elif wall_type == "V":  # Vertical wall centered at x = c_idx * CS
            x_center = c_idx * cell_size
            y_start = r_idx * cell_size
            y_end = (r_idx + 1) * cell_size

            # Calculate coordinates for overlapping rectangle
            x1 = x_center - wall_thickness // 2
            y1 = y_start - wall_thickness // 2  # Extend up for overlap
            x2 = x_center + (wall_thickness + 1) // 2
            y2 = y_end + (wall_thickness + 1) // 2  # Extend down for overlap
            # Clip coordinates to image bounds to be safe
            x1, x2 = max(0, x1), min(image_size, x2)
            y1, y2 = max(0, y1), min(image_size, y2)
            draw.rectangle([x1, y1, x2, y2], fill=WALL_COLOR)

    # 3. Add Entrance and Exit Markers (Draw last, on top)
    er, ec = entrance_cell
    xr, xc = exit_cell
    marker_radius = cell_size / 4
    # Entrance (Red Circle)
    center_x_entrance = ec * cell_size + cell_size / 2
    center_y_entrance = er * cell_size + cell_size / 2
    entrance_bbox = [
        center_x_entrance - marker_radius,
        center_y_entrance - marker_radius,
        center_x_entrance + marker_radius,
        center_y_entrance + marker_radius,
    ]
    draw.ellipse(entrance_bbox, fill=ENTRANCE_COLOR, outline=ENTRANCE_COLOR)
    # Exit (Green Circle)
    center_x_exit = xc * cell_size + cell_size / 2
    center_y_exit = xr * cell_size + cell_size / 2
    exit_bbox = [
        center_x_exit - marker_radius,
        center_y_exit - marker_radius,
        center_x_exit + marker_radius,
        center_y_exit + marker_radius,
    ]
    draw.ellipse(exit_bbox, fill=EXIT_COLOR, outline=EXIT_COLOR)

    # Save the image
    try:
        img.save(filename)
        print(f"Maze saved as {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating a {GRID_SIZE}x{GRID_SIZE} maze...")
    maze_connections = generate_maze(GRID_SIZE, GRID_SIZE)
    print(f"Drawing maze ({IMAGE_SIZE}x{IMAGE_SIZE} pixels) with unified walls...")
    draw_maze(maze_connections, GRID_SIZE, GRID_SIZE, IMAGE_SIZE, CELL_SIZE)
    print("Done.")
