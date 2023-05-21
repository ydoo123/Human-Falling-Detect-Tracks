from PIL import Image
import matplotlib.pyplot as plt
import yaml
import os
import json
import numpy as np
import cv2
import quaternion
import math


# Define the YAML file path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
UPPER_PATH = os.path.dirname(CURRENT_PATH)
MAP_PATH = os.path.join(UPPER_PATH, "Map")
YAML_FILE = os.path.join(MAP_PATH, "map_112_0516.yaml")
LOG_PATH = os.path.join(MAP_PATH, "coord_log.json")
# Load the YAML file
with open(YAML_FILE, "r") as stream:
    map_data = yaml.safe_load(stream)

# Open the PGM file
IMG_PATH = os.path.join(MAP_PATH, map_data["image"])
img = Image.open(IMG_PATH)

# get the map config
CONFIG_PATH = os.path.join(MAP_PATH, "map_config.json")
with open(CONFIG_PATH, "r") as f:
    map_config_data = json.load(f)
    map_vertices = np.array(map_config_data["vertices"], np.float32)

# get the cam config
CAM_CONFIG_PATH = os.path.join(MAP_PATH, "cam_config.json")
with open(CAM_CONFIG_PATH, "r") as f:
    cam_config_data = json.load(f)
    cam_vertices = np.array(cam_config_data["vertices"], np.float32)
    # rotate the cam_vertices
    # cam_vertices = np.roll(cam_vertices, 1, axis=0)


def get_head_body_coord():
    with open(LOG_PATH, "r") as f:
        log_data = json.load(f)
        head_coord = log_data["head_coord"]
        body_coord = log_data["body_coord"]
    return head_coord, body_coord


def dump_log(head_coord, body_coord):
    log_data = {
        "head_coord": head_coord,
        "body_coord": body_coord,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=4)
    return None


# get the image size
width, height = img.size

# Get the origin
origin = map_data["origin"]
origin_coord = (
    -origin[0] / map_data["resolution"],
    height - (-origin[1] / map_data["resolution"]),
)


fig, ax = plt.subplots()
# make rectangle from vertices and color is yellow
rectangle = plt.Polygon(
    map_vertices, closed=True, fill=False, linewidth=2, color="orange"
)
ax.add_patch(rectangle)


def convert_coord(coord):
    """
    convert the camera coordinate to map coordinate(pixel, pixel)
    return: (map_transformed_x, map_transformed_y)
    """
    transformation_matrix = cv2.getPerspectiveTransform(cam_vertices, map_vertices)

    # convert the coordinate
    cam_transformed_x, cam_transformed_y = coord[0], coord[1]
    map_transformed_x, map_transformed_y = cv2.perspectiveTransform(
        np.array([[[cam_transformed_x, cam_transformed_y]]], dtype=np.float32),
        transformation_matrix,
    )[0][0]

    return (map_transformed_x, map_transformed_y)


def get_center(head_x, head_y, body_x, body_y):
    """
    get the center of the person
    return: center_x, center_y
    """
    center_x = (head_x + body_x) / 2
    center_y = (head_y + body_y) / 2
    return center_x, center_y


def get_inverse_coord(head_coord, body_coord, meter=1.5):
    midpoint = (
        (head_coord[0] + body_coord[0]) / 2,
        (head_coord[1] + body_coord[1]) / 2,
    )

    # Calculate slope of line between head and body
    slope = (body_coord[1] - head_coord[1]) / (body_coord[0] - head_coord[0])

    # Calculate slope of perpendicular line
    perp_slope = -1 / slope

    # Calculate y-intercept of perpendicular line
    y_intercept = midpoint[1] - perp_slope * midpoint[0]

    # Define two points on perpendicular line
    length = (
        (body_coord[0] - head_coord[0]) ** 2 + (body_coord[1] - head_coord[1]) ** 2
    ) ** 0.5

    ratio = meter / (length * map_data["resolution"])

    length *= ratio

    x_inverse_1 = midpoint[0] + (length / 2) * (1 / (1 + perp_slope**2)) ** 0.5
    y_inverse_1 = perp_slope * x_inverse_1 + y_intercept

    x_inverse_2 = midpoint[0] - (length / 2) * (1 / (1 + perp_slope**2)) ** 0.5
    y_inverse_2 = perp_slope * x_inverse_2 + y_intercept

    inverse_coord = [[x_inverse_1, y_inverse_1], [x_inverse_2, y_inverse_2]]
    return inverse_coord


def select_short_coord(inverse_coord, origin_coord):
    inverse_coord_1 = inverse_coord[0]
    inverse_coord_2 = inverse_coord[1]

    # get the distance between the origin and the inverse coord
    distance_1 = (
        (inverse_coord_1[0] - origin_coord[0]) ** 2
        + (inverse_coord_1[1] - origin_coord[1]) ** 2
    ) ** 0.5
    distance_2 = (
        (inverse_coord_2[0] - origin_coord[0]) ** 2
        + (inverse_coord_2[1] - origin_coord[1]) ** 2
    ) ** 0.5

    if distance_1 <= distance_2:
        return inverse_coord_1

    return inverse_coord_2


def convert_pixel_to_real_coord(pixel_coord):
    """
    convert the pixel coordinate to real coordinate(meter, meter)
    return: (real_x, real_y)
    """
    # func(origin_coord) = (0, 0)
    real_x = (pixel_coord[0] - origin_coord[0]) * map_data["resolution"]
    real_y = (pixel_coord[1] - origin_coord[1]) * map_data["resolution"]

    return real_x, real_y


def get_rotation(start_coord, end_coord):
    # get the angle
    dx = end_coord[0] - start_coord[0]
    dy = end_coord[1] - start_coord[1]
    rads = math.atan2(dy, dx)

    # calculate the angle in quaternion
    q = quaternion.from_euler_angles(0, 0, rads)  # w, x, y, z

    return q.w, q.z


def get_real_coord(head_coord, body_coord):
    """
    This is the main function to get the real coordinate
    """
    map_head_coord = convert_coord(head_coord)
    map_body_coord = convert_coord(body_coord)

    inverse_coord = get_inverse_coord(
        map_head_coord,
        map_body_coord,
    )

    center_coord = get_center(
        map_head_coord[0], map_head_coord[1], map_body_coord[0], map_body_coord[1]
    )

    inverse_coord = select_short_coord(inverse_coord, origin_coord)

    w, z = get_rotation(inverse_coord, center_coord)
    x, y = convert_pixel_to_real_coord(inverse_coord)

    return x, y, z, w


def main():
    ax.imshow(img, cmap="gray")
    ax.scatter(
        origin_coord[0],
        origin_coord[1],
        marker="+",
        color="red",
    )
    ax.axis("equal")
    ax.axis("off")

    head_coord, body_coord = get_head_body_coord()

    map_head_coord = convert_coord(head_coord)
    map_body_coord = convert_coord(body_coord)

    inverse_coord = get_inverse_coord(
        map_head_coord,
        map_body_coord,
    )

    inverse_coord = select_short_coord(inverse_coord, origin_coord)

    ax.scatter(
        map_head_coord[0],
        map_head_coord[1],
        marker="o",
        color="green",
    )
    ax.scatter(
        map_body_coord[0],
        map_body_coord[1],
        marker="o",
        color="blue",
    )
    ax.scatter(
        inverse_coord[0],
        inverse_coord[1],
        marker="x",
        color="orange",
    )

    real_origin = convert_pixel_to_real_coord(origin_coord)
    real_dest = convert_pixel_to_real_coord(inverse_coord)
    print("real_origin: ", real_origin)
    print("real_dest: ", real_dest)

    plt.axis("scaled")

    x, y, z, w = get_real_coord(head_coord, body_coord)
    print(f"x: {x}, y: {y}, z: {z}, w: {w}")
    plt.show()

    # Save the plot as a PNG file
    # plt.savefig("your_map.png", bbox_inches="tight")
    return None


if __name__ == "__main__":
    main()
