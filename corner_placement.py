import matplotlib.pyplot as plt
import matplotlib.patches as patches


def read_data(f_path):
    with open(f_path) as f:
        lines = f.readlines()
        num_rectangles = int(lines[0])
        W, H = map(int, lines[1].split())

        rectangles = []
        for line in lines[2:]:
            w, h = map(int, line.split())
            rectangles.append((w, h))

        # Sort rectangles by area (w * h) in descending order
        rectangles.sort(key=lambda x: x[0] * x[1], reverse=True)

        print(f"W, H = {W}, {H}  # Dimensions of the enveloping rectangle")
        print(f"Rectangle amount {num_rectangles}")
        print(f"rectangles = {rectangles}  # Sorted by area")
        return W, H, num_rectangles, rectangles

def plot_container(W, H, placed_rectangles):
    """
    Plot the container and the placed rectangles, with 'combed' areas where no rectangles exist.

    Args:
    - W (int): Width of the container.
    - H (int): Height of the container.
    - placed_rectangles (list of tuples): List containing details of placed rectangles.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    container_filled = patches.Rectangle((0, 0), W, H, edgecolor='black', facecolor='none', hatch='////', linewidth=0.4)
    ax.add_patch(container_filled)

    for rect in placed_rectangles:
        rect_id, x1, y1, x2, y2, _ = rect
        width, height = x2 - x1, y2 - y1
        rectangle = patches.Rectangle((x1, y1), width, height, edgecolor='black', facecolor='white', linewidth=0.4)
        ax.add_patch(rectangle)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    #ax.set_xlabel('Width')
    #ax.set_ylabel('Height')
    #ax.set_title('Container Packing Visualization with Combed Areas')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('./plot_drawing.png')
    plt.show()


def place_rectangle(rect_idx, x1,y1,rotation,placed_rectangles, rectangles, unplaced_rectangles):

    rect_w, rect_h = rectangles[rect_idx]


    if rotation == 0:
        x2 = x1 + rect_w
        y2 = y1 + rect_h
    else:
        x2 = x1 + rect_h
        y2 = y1 + rect_w

    placed_rectangles.append((rect_idx, x1, y1, x2, y2, rotation))
    unplaced_rectangles[rect_idx] = None

def is_placement_valid(rect_idx, x1_1, y1_1, rotation, placed_rectangles, rectangles, W, H):
    rect_w, rect_h = rectangles[rect_idx]

    if rotation == 0:
        x2_1 = x1_1 + rect_w
        y2_1 = y1_1 + rect_h
    else:
        x2_1 = x1_1 + rect_h
        y2_1 = y1_1 + rect_w

    # Check container boundaries
    if x2_1 > W or y2_1 > H or x1_1 < 0 or y1_1 < 0:
        return False  # Rectangle goes out of the container

    # Check for overlap with already placed rectangles
    for idx, x1_2, y1_2, x2_2, y2_2, rotation in placed_rectangles:
        # If any of the following conditions are true, rectangles do not overlap
        if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
            continue
        else:
            return False  # Overlap detected



    return True  # Placement is valid

def create_configuration_matrix(W, H):
    return [[0] * W for _ in range(H)]

def update_configuration_matrix(matrix, rectangle):
    # Extracting rectangle information
    _, x1, y1, x2, y2, _ = rectangle

    # Transform y-coordinates to match matrix representation
    matrix_y1 = H - y2
    matrix_y2 = H - y1

    for i in range(matrix_y1, matrix_y2):
        for j in range(x1, x2):
            matrix[i][j] = 1

def find_corners(matrix, W, H):
    corners = []
    for x in range(H):  # Iterate over rows (vertical)
        for y in range(W):  # Iterate over columns (horizontal)
            if matrix[x][y] == 0:  # Check only empty cells
                forming_items = []

                # Check and record the neighbors
                if x == 0 or matrix[x - 1][y] == 1:
                    forming_items.append('top' if x == 0 else 'rect_top')
                if y == W - 1 or matrix[x][y + 1] == 1:
                    forming_items.append('right' if y == W - 1 else 'rect_right')
                if x == H - 1 or matrix[x + 1][y] == 1:
                    forming_items.append('bottom' if x == H - 1 else 'rect_bottom')
                if y == 0 or matrix[x][y - 1] == 1:
                    forming_items.append('left' if y == 0 else 'rect_left')

                # If it's a corner, append the corner with its forming items
                if len(forming_items) >= 2:
                    corners.append((y, H - x - 1, forming_items))
    return corners

def determine_relative_position(ccoa_rect, placed_rect):
    positions = []
    ccoa_x1, ccoa_y1, ccoa_x2, ccoa_y2 = ccoa_rect
    placed_x1, placed_y1, placed_x2, placed_y2 = placed_rect

    # Logic to determine relative position (e.g., 'rect_top', 'rect_bottom', etc.)
    # This involves checking the coordinates of ccoa_rect and placed_rect
    # and returning the relative position

    if placed_y2 <= ccoa_y1:
        positions.append('rect_bottom')
    if placed_y1 >= ccoa_y2:
        positions.append('rect_top')
    if placed_x2 <= ccoa_x1:
        positions.append('rect_left')
    if placed_x1 >= ccoa_x2:
        positions.append('rect_right')

    return positions

def find_ccoas(rectangles, placed_rectangles, unplaced_rectangles, corners, ccoas, W, H):
    for corner in corners:
        for rect_idx, rectangle in enumerate(unplaced_rectangles):
            if rectangle is None:
                continue
            for rotation in range(0,2):
                placement_bl_corner = (corner[0], corner[1])
                if rotation == 0:
                    rect_w, rect_h = rectangle
                    placement_bl_corner = (corner[0], corner[1])
                    placement_br_corner = (corner[0] - rect_w + 1, corner[1])
                    placement_tl_corner = (corner[0], corner[1] - rect_h + 1)
                    placement_tr_corner = (corner[0] - rect_w + 1, corner[1] - rect_h + 1)

                if rotation == 1:
                    rect_h, rect_w = rectangle
                    placement_bl_corner = (corner[0], corner[1])
                    placement_br_corner = (corner[0] - rect_w + 1, corner[1])
                    placement_tl_corner = (corner[0], corner[1] - rect_h + 1 )
                    placement_tr_corner = (corner[0] - rect_w + 1, corner[1] -  rect_h + 1)


                if(is_placement_valid(rect_idx, placement_bl_corner[0], placement_bl_corner[1], rotation, placed_rectangles, rectangles, W, H)):
                    # print(f'Rectangle {rect_idx} w:{rect_w}, h:{rect_h} r:{rotation} can be placed at {placement_bl_corner}, bottom left')
                    ccoas.append((rect_idx, placement_bl_corner[0], placement_bl_corner[1], rotation, corner[2]))
                if (is_placement_valid(rect_idx, placement_br_corner[0], placement_br_corner[1], rotation,placed_rectangles, rectangles, W, H)):
                    # print(f'Rectangle {rect_idx} w:{rect_w}, h:{rect_h} r:{rotation} can be placed at {placement_br_corner}, bottom right')
                    ccoas.append((rect_idx, placement_br_corner[0], placement_br_corner[1], rotation, corner[2]))
                if (is_placement_valid(rect_idx, placement_tl_corner[0], placement_tl_corner[1], rotation,placed_rectangles, rectangles, W, H)):
                    # print(f'Rectangle {rect_idx} w:{rect_w}, h:{rect_h} r:{rotation} can be placed at {placement_tl_corner}, top left')
                    ccoas.append((rect_idx, placement_tl_corner[0], placement_tl_corner[1], rotation, corner[2]))
                if (is_placement_valid(rect_idx, placement_tr_corner[0], placement_tr_corner[1], rotation,placed_rectangles, rectangles, W, H)):
                    # print(f'Rectangle {rect_idx} w:{rect_w}, h:{rect_h} r:{rotation} can be placed at {placement_tr_corner}, top right')
                    ccoas.append((rect_idx, placement_tr_corner[0], placement_tr_corner[1], rotation, corner[2]))

    return ccoas

def find_ccoa_degree(ccoa, placed_rectangles, r, W, H):
    rect_idx, x1, y1, rotation, forming_items = ccoa
    # print(forming_items)
    rect_w, rect_h = rectangles[rect_idx]
    if rotation == 1:
        rect_w, rect_h = rect_h, rect_w

    ccoa_rect = (x1, y1, x1 + rect_w, y1 + rect_h)
    d_min = float('inf')

    for placed_rect in placed_rectangles:
        placed_rect_idx, placed_x1, placed_y1, placed_x2, placed_y2, _ = placed_rect
        if placed_rect[0] == rect_idx:
            continue  # Skip the rectangle itself

        relative_positions = determine_relative_position(ccoa_rect, (placed_x1, placed_y1, placed_x2, placed_y2))

        if any(pos in forming_items for pos in relative_positions):
            continue  # Skip this rectangle if any relative position is a forming item


        distance = calculate_min_distance(ccoa_rect, (placed_rect[1], placed_rect[2], placed_rect[3], placed_rect[4]))
        d_min = min(d_min, distance)

    # Calculate distance to container sides aswell same logic should apply here

    # Calculate distance to container sides
    for side in ['top', 'bottom', 'left', 'right']:
        if side in forming_items:
            continue
        distance_to_side = calculate_distance_to_side(ccoa_rect, side, W, H)
        d_min = min(d_min, distance_to_side)

    # print(d_min, rect_idx)


    # Calculate the degree
    k = 1 - (d_min / (r * ((rect_w + rect_h) / 2)))



    return k

def calculate_min_distance(rect1, rect2):
    # Extracting the coordinates
    x1_bl, y1_bl, x1_tr, y1_tr = rect1
    x2_bl, y2_bl, x2_tr, y2_tr = rect2

    # Finding the edges of the rectangles
    left1, right1, top1, bottom1 = x1_bl, x1_tr, y1_tr, y1_bl
    left2, right2, top2, bottom2 = x2_bl, x2_tr, y2_tr, y2_bl

    # Calculating the horizontal and vertical distances
    horizontal_distance = max(0, max(left1, left2) - min(right1, right2))
    vertical_distance = max(0, max(bottom1, bottom2) - min(top1, top2))

    # If one rectangle is above or to the right of the other, return the direct distance
    if horizontal_distance > 0 or vertical_distance > 0:
        return abs((horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5)

    # If the rectangles overlap or touch, the minimum distance is 0
    return 0

def calculate_distance_to_side(ccoa_rect, side, W, H):
    ccoa_x1, ccoa_y1, ccoa_x2, ccoa_y2 = ccoa_rect

    if side == 'top':
        return H - ccoa_y2
    elif side == 'bottom':
        return ccoa_y1
    elif side == 'left':
        return ccoa_x1
    elif side == 'right':
        return W - ccoa_x2

    return float('inf')

def find_most_feasible_ccoa(ccoas, placed_rectangles, r, W, H):
    best_ccoa = None
    highest_degree = -1
    min_distance_to_origin = float('inf')

    for ccoa in ccoas:
        degree = find_ccoa_degree(ccoa, placed_rectangles, r, W, H)
        x, y = ccoa[1], ccoa[2]  # Coordinates of the bottom-left corner
        distance_to_origin = x * x + y * y

        if degree > highest_degree or (degree == highest_degree and distance_to_origin < min_distance_to_origin):
            highest_degree = degree
            best_ccoa = ccoa
            min_distance_to_origin = distance_to_origin

    return best_ccoa, highest_degree

def algorithm_A0(W, H, rectangles):
    placed_rectangles = []
    unplaced_rectangles = rectangles.copy()
    C = create_configuration_matrix(W, H)  # Config matrix
    # update_configuration_matrix(C, placed_rectangles)

    corners = find_corners(C, W, H)
    L = find_ccoas(rectangles, placed_rectangles, unplaced_rectangles, corners, [], W, H)

    while L:
        ccoas_with_degrees = [(ccoa, find_ccoa_degree(ccoa, placed_rectangles, 1, W, H)) for ccoa in L]
        best_ccoa, best_degree = max(ccoas_with_degrees, key=lambda x: x[1], default=(None, -1))

        if best_ccoa is None:
            break  # No feasible placement found

        place_rectangle(best_ccoa[0], best_ccoa[1], best_ccoa[2], best_ccoa[3], placed_rectangles, rectangles,
                        unplaced_rectangles)
        update_configuration_matrix(C, placed_rectangles[-1])
        corners = find_corners(C, W, H)
        L = find_ccoas(rectangles, placed_rectangles, unplaced_rectangles, corners, [], W, H)

    return placed_rectangles, unplaced_rectangles


if __name__ == '__main__':

    path = './Original'
    W, H, num_rectangles, rectangles = read_data(r'./original_Hopper_Turton/Original/C1_3')
    placed_rectangles, unplaced_rectangles = algorithm_A0(W, H, rectangles)

    area = W * H
    placed_area = 0
    for rect in placed_rectangles:
        rect_idx = rect[0]
        placed_area += rectangles[rect_idx][0] * rectangles[rect_idx][1]

    area_utilization = placed_area / area

    print(area_utilization)

    print(placed_rectangles)

    plot_container(W, H, placed_rectangles)




















