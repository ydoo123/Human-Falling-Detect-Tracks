from PIL import Image
import matplotlib.pyplot as plt
import yaml
import os

# Define the YAML file path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(CURRENT_DIR, "Map")
YAML_FILE = os.path.join(MAP_DIR, "map_112_0510.yaml")
# Load the YAML file
with open(YAML_FILE, "r") as stream:
    data = yaml.safe_load(stream)

# Open the PGM file
IMG_PATH = os.path.join(MAP_DIR, data["image"])
img = Image.open(IMG_PATH)

# Get the origin
origin = data["origin"]

# get the image size
width, height = img.size

# Plot the image and origin
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.scatter(
    -origin[0] / data["resolution"],
    height - (-origin[1] / data["resolution"]),
    marker="+",
    color="red",
)
ax.axis("equal")
ax.axis("off")

plt.show()
# Save the plot as a PNG file
# plt.savefig("your_map.png", bbox_inches="tight")
