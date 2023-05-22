import cv2
import requests
import datetime
import os

IMAGE_URL = "http://130.162.152.119/upload_image"


def upload_image(frame):
    def save_frame(frame):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_dir = os.getcwd()
        path = os.path.join(current_dir, "photo", current_time + ".jpg")
        cv2.imwrite(path, frame)
        return path

    def post_image(path):
        response = requests.post(
            IMAGE_URL,
            files={"image": open(path, "rb")},
        )

    path = save_frame(frame)
    post_image(path)
