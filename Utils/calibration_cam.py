import os
import cv2
import numpy as np
import json

global dragging_vertex, vertices, dots, dragging_dot
dragging_vertex = None
dragging_dot = None
dots = [(200, 200), (100, 100)]

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
UPPER_PATH = os.path.dirname(CURRENT_PATH)
CAM_CONFIG_PATH = os.path.join(UPPER_PATH, "MAP", "cam_config.json")

# get vertices from json file
with open(CAM_CONFIG_PATH, "r") as f:
    data = json.load(f)
    vertices = np.array(data["vertices"], np.int32)


def mouse_callback(event, x, y, flags, param):
    global vertices, dragging_vertex, dragging_dot
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, vertex in enumerate(vertices):
            if np.sqrt((vertex[0] - x) ** 2 + (vertex[1] - y) ** 2) < 10:
                dragging_vertex = i
        for j, dot in enumerate(dots):
            if np.sqrt((dot[0] - x) ** 2 + (dot[1] - y) ** 2) < 10:
                dragging_dot = j
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_vertex is not None:
            vertices[dragging_vertex] = [x, y]
        if dragging_dot is not None:
            dots[dragging_dot] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_vertex = None
        dragging_dot = None


def resizePadding(image, desized_size):
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)  # top: 84, bottom: 84
    left, right = delta_w // 2, delta_w - (delta_w // 2)  # left: 0, right: 0

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image


def main():
    cam = cv2.VideoCapture(0)
    # show frame
    while True:
        ret, frame = cam.read()

        # make black border to keep aspect ratio
        size = (384, 384)
        frame = resizePadding(frame, size)

        # resize the frame with 2x
        frame = cv2.resize(frame, (0, 0), fx=2, fy=2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

    img = frame

    while True:
        print(f"dots: {dots}")
        cv2.setMouseCallback("frame", mouse_callback)

        # Copy the original image to display the quadrilateral on top of it
        img_display = img.copy()

        # Draw the quadrilateral
        cv2.polylines(img_display, [vertices], True, (0, 0, 255), thickness=2)
        # put the name of the vertices
        for i, vertex in enumerate(vertices):
            cv2.putText(
                img_display,
                f"{i}",
                tuple(vertex),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.circle(img_display, tuple(dots[0]), 5, (255, 0, 0), -1)
        cv2.circle(img_display, tuple(dots[1]), 5, (0, 255, 0), -1)

        # Display the image
        cv2.imshow("frame", img_display)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("w"):
            # write to json file
            with open(CAM_CONFIG_PATH, "r") as f:
                data = json.load(f)
                data["vertices"] = vertices.tolist()

            with open(CAM_CONFIG_PATH, "w") as f:
                json.dump(data, f, indent=4)

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
