from PIL import Image
import matplotlib.pyplot as plt
import yaml
import os
import json
import numpy as np

global vertices, dragging
dragging = None


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
    config_data = json.load(f)
    vertices = np.array(config_data["vertices"], np.int32)


fig, ax = plt.subplots()
# make rectangle from vertices
rectangle = plt.Polygon(vertices, closed=True, fill=False, linewidth=2)
ax.add_patch(rectangle)

# put the text of the vertices
for i, vertex in enumerate(vertices):
    ax.text(vertex[0], vertex[1], str(i), color="orange", fontsize=20)


def mouse_callback(event):
    global vertices, dragging
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        if event.button == 1:  # Left button
            if dragging is None:
                for i, vertex in enumerate(vertices):
                    if abs(vertex[0] - x) < 3 and abs(vertex[1] - y) < 3:
                        dragging = i
            else:
                vertices[dragging] = [x, y]
                rectangle.set_xy(vertices)

        elif event.button == 3:  # Right button
            # quit the plt
            plt.close()

        else:
            dragging = None

    else:
        dragging = None
        fig.canvas.release_mouse(event)

    plt.draw()


# Connect the mouse callback function to the figure


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

    fig.canvas.mpl_connect("button_press_event", mouse_callback)
    fig.canvas.mpl_connect("button_release_event", mouse_callback)
    fig.canvas.mpl_connect("motion_notify_event", mouse_callback)

    plt.axis("scaled")
    plt.show()

    # save the vertices to json file
    config_data["vertices"] = vertices.tolist()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f)
        print("Saved the vertices to json file.")

    # Save the plot as a PNG file
    # plt.savefig("your_map.png", bbox_inches="tight")
    return None


if __name__ == "__main__":
    main()
