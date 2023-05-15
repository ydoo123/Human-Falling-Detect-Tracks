import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
import json
import requests
import uuid

# import playsound


ACTION_DICT = {
    "pending..": 0,
    "Standing": 0,
    "Walking": 0,
    "Sitting": 0,
    "Lying Down": 2,
    "Stand up": 0,
    "Sit down": 0,
    "Fall Down": 1,
}

# get url from url.json
with open("url.json") as f:
    URL = json.load(f)
    URL = URL["url"]

source = 2  # 2 for usb_cam

SAVE_FRAME_RATE = 20  # 20 is good and 30 is too fast
ACTION_CHECK_RATE = 0.1  # to check action every 0.1 second
ACTION_COUNT_VALUE = 3
ACTION_CHECK_TIME = 15


def beep():
    # playsound.playsound('TTS/fall_detect_voice.mp3', True)
    os.system('say "넘어짐이 감지되었습니다."')

    return None


def save_photo():
    PATH = "photo"
    uuid_str = str(uuid.uuid4()) + ".jpg"
    photo_path = os.path.join(PATH, uuid_str)
    cv2.imwrite(photo_path, frame)

    return photo_path


def upload_photo():
    photo_path = save_photo()

    # post photo to server
    files = {"file": open(photo_path, "rb")}
    r = requests.post(URL + "/upload_photo", files=files)
    print(r.status_code, r.reason)
    return None


def send_coord(bbox):
    # calculate center of bbox
    cam_id = 0
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    z = 0.0
    w = 0.0

    # coord = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    if args.test:
        x = 0
        y = 0

    # post coord to server
    r = requests.post(
        URL + "/upload_dest",
        json={"cam_id": cam_id, "x": x, "y": y, "z": z, "w": w},
    )
    print(r.status_code, r.reason)
    print(x, y)
    return None


def preproc(image):
    """preprocess function for CameraLoader."""
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        )
    )


if __name__ == "__main__":
    par = argparse.ArgumentParser(description="Human Fall Detection Demo.")
    par.add_argument(
        "-C",
        "--cam",
        default=source,  # required=True,  # default=2,
        help="Source of camera or video file path.",
    )
    par.add_argument(
        "--detection_input_size",
        type=int,
        default=384,
        help="Size of input in detection model in square must be divisible by 32 (int).",
    )
    par.add_argument(
        "--pose_input_size",
        type=str,
        default="224x160",
        help="Size of input in pose model must be divisible by 32 (h, w)",
    )
    par.add_argument(
        "--pose_backbone",
        type=str,
        default="resnet50",
        help="Backbone model for SPPE FastPose model.",
    )
    par.add_argument(
        "--show_detected",
        default=False,
        action="store_true",
        help="Show all bounding box from detection.",
    )
    par.add_argument(
        "--show_skeleton", default=True, action="store_true", help="Show skeleton pose."
    )
    par.add_argument(
        "--save_out", type=str, default="", help="Save display to video file."
    )
    par.add_argument(
        "--device", type=str, default="cpu", help="Device to run model on cpu or cuda."
    )
    par.add_argument(
        "--show_fps", default=False, action="store_true", help="Show FPS of program."
    )
    par.add_argument("--test", default=False, action="store_true", help="Test mode.")
    args = par.parse_args()

    device = args.device

    # for check action
    prev_time = time.time()
    action_history = np.zeros(ACTION_CHECK_TIME)
    action_name = "pending.."
    count = 0

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split("x")
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(
        args.pose_backbone, inp_pose[0], inp_pose[1], device=device
    )

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)  # inp_dets is 384

    cam_source = args.cam
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=3000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(
            int(cam_source) if cam_source.isdigit() else cam_source, preprocess=preproc
        ).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    is_video_out = False
    if args.save_out != "":
        is_video_out = True
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save_out, codec, SAVE_FRAME_RATE, (inp_dets * 2, inp_dets * 2)
        )

    fps_time = 0
    whole_frame = 0
    while cam.grabbed():
        whole_frame += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor(
                [track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32
            )
            detected = (
                torch.cat([detected, det], dim=0) if detected is not None else det
            )

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [
                Detection(
                    kpt2bbox(ps["keypoints"].numpy()),
                    np.concatenate(
                        (ps["keypoints"].numpy(), ps["kp_score"].numpy()), axis=1
                    ),
                    ps["kp_score"].mean().numpy(),
                )
                for ps in poses
            ]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(
                        frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1
                    )

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = "pending.."
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = "{}: {:.2f}%".format(action_name, out[0].max() * 100)
                if action_name == "Fall Down":
                    clr = (255, 0, 0)
                elif action_name == "Lying Down":
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1
                )
                frame = cv2.putText(
                    frame,
                    str(track_id),
                    (center[0], center[1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.4,
                    (255, 0, 0),
                    2,
                )
                frame = cv2.putText(
                    frame,
                    action,
                    (bbox[0] + 5, bbox[1] + 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.4,
                    clr,
                    1,
                )

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2.0, fy=2.0)
        text = " "

        if args.show_fps:
            text = "%d, FPS: %f" % (whole_frame, 1.0 / (time.time() - fps_time))

        frame = cv2.putText(
            frame,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if time.time() - prev_time >= ACTION_CHECK_RATE:
            """
            Check action every 0.1 second.
            Send coordinate to server if action is "Fall Down"
            """

            action_history = np.roll(action_history, -1)
            action_history[-1] = ACTION_DICT[action_name]

            # if tracker.tracks[0].keypoints_list:
            #     head_coord = tracker.tracks[0].keypoints_list[-1][0]
            #     print(head_coord)
            #     body_coord = tracker.tracks[0].keypoints_list[-1][8]
            #     print(body_coord)

            if (
                np.count_nonzero(action_history == 0) <= ACTION_COUNT_VALUE
                and count == 0
            ):
                count += 1
                action_history = np.zeros(ACTION_COUNT_VALUE)
                print("Fall Down")
                head_coord = tracker.tracks[0].keypoints_list[-1][0][:2]
                body_coord = tracker.tracks[0].keypoints_list[-1][8][:2]
                print(head_coord, body_coord)
                beep()

                # send_coord(bbox)
                # upload_photo()

            prev_time = time.time()

        if is_video_out:
            writer.write(frame)

        if not is_video_out:
            # disable show frame when --save_out used.
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clear resource.
    cam.stop()
    if is_video_out:
        writer.release()
    cv2.destroyAllWindows()
