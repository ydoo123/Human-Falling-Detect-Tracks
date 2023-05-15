from PIL import Image
import matplotlib.pyplot as plt
import yaml
import os
import json
import numpy as np
import cv2


# Define the YAML file path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
UPPER_PATH = os.path.dirname(CURRENT_PATH)
MAP_PATH = os.path.join(UPPER_PATH, "Map")
YAML_FILE = os.path.join(MAP_PATH, "map_112_0510.yaml")
# Load the YAML file
with open(YAML_FILE, "r") as stream:
    data = yaml.safe_load(stream)

# Open the PGM file
IMG_PATH = os.path.join(MAP_PATH, data["image"])
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

    point_cam = np.array([[coord[0], coord[1]]], dtype=np.float32)
    point_map_transformed = cv2.transform(
        point_cam.reshape(1, -1, 2), transformation_matrix
    )

    map_transformed_x = point_map_transformed[0][0][0]
    map_transformed_y = point_map_transformed[0][0][1]

    return (map_transformed_x, map_transformed_y)


def get_center(head_x, head_y, body_x, body_y):
    """
    get the center of the person
    return: center_x, center_y
    """
    center_x = (head_x + body_x) / 2
    center_y = (head_y + body_y) / 2
    return center_x, center_y


def get_inverse_coord(head_coord, body_coord, ratio=2.0):
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

    length *= ratio

    x_inverse_1 = midpoint[0] + (length / 2) * (1 / (1 + perp_slope**2)) ** 0.5
    y_inverse_1 = perp_slope * x_inverse_1 + y_intercept

    x_inverse_2 = midpoint[0] - (length / 2) * (1 / (1 + perp_slope**2)) ** 0.5
    y_inverse_2 = perp_slope * x_inverse_2 + y_intercept

    inverse_coord = [[x_inverse_1, y_inverse_1], [x_inverse_2, y_inverse_2]]
    return inverse_coord


def select_short_coord(inverse_coord, origin):
    inverse_coord_1 = inverse_coord[0]
    inverse_coord_2 = inverse_coord[1]

    # get the distance between the origin and the inverse coord
    distance_1 = (
        (inverse_coord_1[0] - origin[0]) ** 2 + (inverse_coord_1[1] - origin[1]) ** 2
    ) ** 0.5
    distance_2 = (
        (inverse_coord_2[0] - origin[0]) ** 2 + (inverse_coord_2[1] - origin[1]) ** 2
    ) ** 0.5

    if distance_1 <= distance_2:
        return inverse_coord_1

    return inverse_coord_2


def get_rotation():
    return None


def main():
    # Get the origin
    origin = data["origin"]

    # get the image size
    width, height = img.size

    # Plot the image and origin

    ax.imshow(img, cmap="gray")
    ax.scatter(
        -origin[0] / data["resolution"],
        height - (-origin[1] / data["resolution"]),
        marker="+",
        color="red",
    )
    ax.axis("equal")
    ax.axis("off")

    head_coord = (100, 100)
    body_coord = (200, 200)

    map_head_coord = convert_coord(head_coord)
    map_body_coord = convert_coord(body_coord)

    inverse_coord = get_inverse_coord(
        map_head_coord,
        map_body_coord,
    )

    inverse_coord = select_short_coord(inverse_coord, origin)

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

    plt.axis("scaled")
    plt.show()

    # Save the plot as a PNG file
    # plt.savefig("your_map.png", bbox_inches="tight")
    return None


if __name__ == "__main__":
    main()
