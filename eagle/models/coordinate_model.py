import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from ..utils.pitch import GROUND_TRUTH_POINTS, INTERSECTION_TO_PITCH_POINTS
from tqdm import tqdm
from .keypoint_hrnet import KeypointModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from boxmot import DeepOCSORT
from pathlib import Path

PITCH_WIDTH = 105
PITCH_HEIGHT = 68


class CoordinateModel:

    def __init__(self):
        # TODO: Support GPU inference
        # self.keypoint_model = YOLO("eagle/models/weights/keypoint_detector.onnx", task="pose", verbose=False)
        # self.detector_model = YOLO("eagle/models/weights/detector_medium.onnx", task="detect", verbose=False)
        self.detector_model = YOLO("eagle/models/weights/detector_large.pt").to("mps")
        self.keypoint_model = KeypointModel(57).to("mps")
        self.keypoint_model.load_state_dict(torch.load("eagle/models/weights/keypoints_main.pth"))
        self.keypoint_model.eval()
        self.class_names = {0: "Player", 1: "Goalkeeper", 2: "Ball", 3: "Referee", 4: "Staff members"}
        self.transforms = A.Compose(  # Define the transformations for the keypoint model
            [A.Resize(540, 960), A.Normalize(), ToTensorV2()],
        )
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.tracker = DeepOCSORT(
            model_weights=Path("osnet_x0_25_msmt17.pt"),
            device="mps",
            fp16=False,
        )
        # print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

    def get_coordinates(self, frames: np.ndarray, fps: int, num_homography: int = 1, num_keypoint_detection: int = 1, verbose: bool = True, calibration: bool = False) -> dict:
        """
        Get the coordinates of the players, goalkeepers and the ball
        :param frames: Input frames read using CV2 in BGR format
        :param fps: Frames per second of the video
        :param num_homography: Number of times per second to calculate the homography matrix
        :param num_keypoint_detection: Number of times per second to detect keypoints using the model
        :param verbose: Whether to show the progress bar
        :param calibration: Whether to calibrate the keypoints

        :return: dictionary containing the image coordinates of players, goalkeepers and the ball. The index is the frame number,
        The second level has 3 keys:
        - "Coordinates": Nested dictionary where the first level keys are the Class Names (Player, Goalkeeper, Ball),
        Second level keys are the ids, Second level values are the Bounding Boxes, Confidence and Bottom Center
        - "Time": Time since the start of the video in the format MM:SS
        - "Keypoints": dictionary containing the calibrated keypoints where the key is the pitch point (str) and the value is the image coordinates (useful for plotting)
        """
        homography_interval = int(fps / num_homography)
        keypoint_interval = int(fps / num_keypoint_detection)
        prev_gray = None
        prev_keypoints = {}
        res = {}
        mem = {}
        compute_homography = False  # Whether to compute homography matrix outside of the interval
        for i, frame in tqdm(enumerate(frames), desc="Processing Frames", total=len(frames)) if verbose else enumerate(frames):
            if i in res:  # This only happens when the model predicts less than 4 keypoints and we use the next frame
                continue

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i == 0 or (i % keypoint_interval == 0):
                # Detect keypoints using the model for the first frame or at each homography interval
                keypoints = mem.get(i, self.detect_keypoints(frame))
                mem[i] = keypoints
                if len(keypoints) < 4:  # Model only detected less than 4 keypoints, combine with optical flow
                    if i == 0:
                        # If first frame, find the first subsequent frame that has more than 4 keypoints detected and use optical flow to reverse calculate the first frame
                        for j in range(i + 1, len(frames)):
                            next_frame = frames[j]
                            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                            next_keypoints = mem.get(j, self.detect_keypoints(next_frame))
                            mem[j] = next_keypoints
                            if len(next_keypoints) >= 4:  # Found a frame with more than 4 keypoints detected
                                prev_keypoints = next_keypoints
                                break
                        # Reverse all the way to the first frame
                        for j in range(j - 1, i - 1, -1):
                            prev_frame = frames[j]
                            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                            prev_keypoints = self.calculate_optical_flow(prev_frame, prev_gray, prev_keypoints, next_gray)
                            next_gray = prev_gray
                    else:
                        # Calculate optical flow for keypoints tracking and combine with model detection
                        optical_flow_keypoints = self.calculate_optical_flow(frame, prev_gray, prev_keypoints, curr_gray)
                        keypoints = {**keypoints, **optical_flow_keypoints}

            else:
                # Calculate optical flow for keypoints tracking
                optical_flow_keypoints = self.calculate_optical_flow(frame, prev_gray, prev_keypoints, curr_gray)
                if len(optical_flow_keypoints) < 4:
                    # Fallback to model detection if filtered keypoints are less than 4
                    keypoints = mem.get(i, self.detect_keypoints(frame))
                    mem[i] = keypoints
                    keypoints = {**keypoints, **optical_flow_keypoints}  # Combine predictions
                else:
                    keypoints = {**optical_flow_keypoints, **mem.get(i, {})}  # If we had memoized the model prediction, use it

            if calibration:
                prev_keypoints = self.calibrate_keypoints(frame, keypoints)
            else:
                prev_keypoints = keypoints
            prev_gray = curr_gray

            objects = self.detect_objects(frame)

            if i % homography_interval == 0 or compute_homography:
                img_pts = np.array(list(keypoints.values()), dtype=np.float32)
                world_pts = np.array([GROUND_TRUTH_POINTS[point] for point in keypoints], dtype=np.float32)
                for method in [cv2.RANSAC, cv2.RHO]:
                    new_homography_matrix, mask = cv2.findHomography(img_pts, world_pts, method, 5.0 if method is cv2.RANSAC else None)
                    if new_homography_matrix is not None:
                        break  # Exit the loop if a homography is found
                if new_homography_matrix is not None:
                    prev_keypoints = {k: v for k, v, m in zip(prev_keypoints.keys(), prev_keypoints.values(), mask.flatten()) if m}  # Filter out outliers
                    homography_matrix = new_homography_matrix
                    compute_homography = False
                else:
                    compute_homography = True  # For this frame, use the previous homography matrix but compute a new one next frame

            indiv = {}  # Coordinate information at this current frame
            for class_name, class_dict in objects.items():
                for obj_id, obj_dict in class_dict.items():
                    bottom_center = obj_dict["Bottom_center"]
                    bbox_coords = np.array(obj_dict["BBox"], dtype=np.uint16).tolist()
                    conf = obj_dict["Confidence"]
                    coords = np.array([[bottom_center]], dtype=np.float32)  # Dim needs to be 3
                    transformed_coords = cv2.perspectiveTransform(coords, homography_matrix)[0].astype(int)
                    if transformed_coords[0, 0] < 0 or transformed_coords[0, 0] > PITCH_WIDTH or transformed_coords[0, 1] < 0 or transformed_coords[0, 1] > PITCH_HEIGHT:
                        continue
                    curr = {int(obj_id): {"BBox": bbox_coords, "Confidence": conf, "Transformed_Coordinates": transformed_coords.tolist()[0]}}
                    if class_name not in indiv:
                        indiv[class_name] = curr
                    else:
                        indiv[class_name].update(curr)

            # Find the visible area of the pitch at each frame. Convert image coordinates to pitch coordinates
            height, width = frame.shape[:2]
            bottom_left = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), homography_matrix)[0].astype(int)[0]
            bottom_right = cv2.perspectiveTransform(np.array([[[width, 0]]], dtype=np.float32), homography_matrix)[0].astype(int)[0]
            top_left = cv2.perspectiveTransform(np.array([[[0, height]]], dtype=np.float32), homography_matrix)[0].astype(int)[0]
            top_right = cv2.perspectiveTransform(np.array([[[width, height]]], dtype=np.float32), homography_matrix)[0].astype(int)[0]

            # left equation
            m_left = (bottom_left[1] - top_left[1]) / (bottom_left[0] - top_left[0])
            c_left = bottom_left[1] - m_left * bottom_left[0]
            # right equation
            m_right = (bottom_right[1] - top_right[1]) / (bottom_right[0] - top_right[0])
            c_right = bottom_right[1] - m_right * bottom_right[0]

            # Find the point on the lines that corresponds to y = 0 and y = PITCH_HEIGHT
            x_left_0 = int((0 - c_left) / m_left)
            x_left_height = int((PITCH_HEIGHT - c_left) / m_left)
            x_right_0 = int((0 - c_right) / m_right)
            x_right_height = int((PITCH_HEIGHT - c_right) / m_right)

            res[i] = {"Coordinates": indiv, "Time": f"{i // fps // 60:02d}:{i // fps % 60:02d}", "Keypoints": prev_keypoints, "Boundaries": [(x_left_0, 0), (x_left_height, PITCH_HEIGHT), (x_right_height, PITCH_HEIGHT), (x_right_0, 0)]}
        return res

    def calculate_optical_flow(self, frame: np.ndarray, prev_gray: np.ndarray, prev_keypoints: dict, curr_gray: np.ndarray):
        """
        Calculate optical flow for keypoints tracking
        :param frame: Input frame read using CV2 in BGR format
        :param prev_gray: Previous frame in grayscale
        :param prev_keypoints: Previous frame keypoints
        :param curr_gray: Current frame in grayscale

        :return: dictionary containing the keypoints where the key is the pitch point (str) and the value is the image coordinates
        """
        prev_points = np.array(list(prev_keypoints.values()), dtype=np.float32)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.lk_params)
        new_points = new_points[status[:, 0] == 1]
        prev_points = prev_points[status[:, 0] == 1]  # Maintain shape

        filtered_keypoints = {}
        move_amounts = np.linalg.norm(new_points - prev_points, axis=1)
        mean_move_amount = np.mean(move_amounts)
        std_move_amount = np.std(move_amounts) + 1e-6
        for j, (point, new_point) in enumerate(zip(prev_points, new_points)):
            key = list(prev_keypoints.keys())[j]

            # Filter rule 1: If 1 particular keypoint moves significantly more than the movement of the other keypoints, it is also an error
            curr_move_amount = move_amounts[j]
            curr_z_score = (curr_move_amount - mean_move_amount) / std_move_amount
            if curr_z_score > 2:
                continue

            # Filter rule 2: Color value of the 3x3 pixels grid around the keypoint change significantly, this implies it is occluded
            curr_x, curr_y = new_point.astype(int)
            curr_x_min, curr_x_max = max(0, curr_x - 1), min(frame.shape[1], curr_x + 2)
            curr_y_min, curr_y_max = max(0, curr_y - 1), min(frame.shape[0], curr_y + 2)
            curr_grid = frame[curr_y_min:curr_y_max, curr_x_min:curr_x_max]
            curr_grid = cv2.cvtColor(curr_grid, cv2.COLOR_BGR2HSV)
            avg_hue_curr = np.mean(curr_grid[:, :, 0])

            prev_x, prev_y = point.astype(int)
            prev_x_min, prev_x_max = max(0, prev_x - 1), min(frame.shape[1], prev_x + 2)
            prev_y_min, prev_y_max = max(0, prev_y - 1), min(frame.shape[0], prev_y + 2)
            prev_grid = frame[prev_y_min:prev_y_max, prev_x_min:prev_x_max]
            prev_grid = cv2.cvtColor(prev_grid, cv2.COLOR_BGR2HSV)
            avg_hue_prev = np.mean(prev_grid[:, :, 0])

            if abs(avg_hue_curr - avg_hue_prev) > 25:  # Every color takes up 60deg but opencv uses 180 for 8 bit representation
                continue

            filtered_keypoints[key] = tuple(new_point.astype(int))

        return filtered_keypoints

    @torch.no_grad()
    def detect_keypoints(self, frame: np.ndarray):
        """
        Detect keypoints and returns the homography matrix
        :param frame: Input frame read using CV2 in BGR format. Shape = (H, W, C)

        :return: dictionary containing the calibrated keypoints where the key is the pitch point (str) and the value is the image coordinates
        """
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        frame = self.transforms(image=frame)["image"].to(self.keypoint_model.unnormalized_model[1].weight.data.device).float().unsqueeze(0)
        keypoints = self.keypoint_model.get_keypoints(frame)[0]
        res = {}
        for i, x, y, score in keypoints:
            if score < 0.5:
                continue
            label = INTERSECTION_TO_PITCH_POINTS[i]
            res[label] = (int(x * width), int(y * height))

        return res

    def calibrate_keypoints(self, frame: np.ndarray, keypoints: dict):
        """
        Calibrate keypoints by finding the brightest spot in a grid around the keypoint using HSV color space.
        If the original brightness is sufficient, no adjustment is made.
        :param frame: Input frame read using CV2 in BGR format
        :param keypoints: dictionary containing the keypoints where the key is the pitch point (str) and the value is the image coordinates

        :return: dictionary containing the calibrated keypoints where the key is the pitch point (str) and the value is the image coordinates
        """
        OFFSET = 3  # Define the offset for the grid
        BRIGHTNESS_THRESHOLD = 150  # Define a threshold for brightness to decide if adjustment is needed
        new_keypoints = {}

        for key, (x, y) in keypoints.items():
            original_brightness = cv2.cvtColor(frame[y, x].reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0, 2]

            if original_brightness >= BRIGHTNESS_THRESHOLD:
                new_keypoints[key] = (x, y)
            else:
                x_min, x_max = max(0, x - OFFSET), min(frame.shape[1], x + OFFSET)
                y_min, y_max = max(0, y - OFFSET), min(frame.shape[0], y + OFFSET)
                grid = frame[y_min:y_max, x_min:x_max]
                grid_hsv = cv2.cvtColor(grid, cv2.COLOR_BGR2HSV)
                brightness = grid_hsv[:, :, 2]
                original_brightness = grid_hsv[OFFSET, OFFSET, 2]
                # Find the brightest point in the grid
                bright_y, bright_x = np.unravel_index(np.argmax(brightness), brightness.shape)
                adjusted_x = np.clip(x + bright_x - OFFSET, 0, frame.shape[1] - 1)
                adjusted_y = np.clip(y + bright_y - OFFSET, 0, frame.shape[0] - 1)
                new_keypoints[key] = (adjusted_x, adjusted_y)

        return new_keypoints

    @torch.no_grad()
    def detect_objects(self, frame: np.ndarray):
        """
        Detect objects in the frame and return the bounding boxes
        :param frame: Input frame read using CV2 in BGR format

        :return: Nested Dictionary where the first level keys are the Class Names
        Second level keys are the ids
        Second level values are the Bounding Boxes, Confidence and Bottom Center
        """
        detector_pred = self.detector_model(frame, verbose=False, conf=0.1)
        boxes = detector_pred[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        class_labels = boxes.cls.cpu().numpy()

        # Track only players, goalkeepers
        tracks = self.tracker.update(np.hstack((coords, conf.reshape(-1, 1), class_labels.reshape(-1, 1))), frame)
        res = {"Player": {}, "Goalkeeper": {}}
        for track_item in tracks:
            x1, y1, x2, y2, id, curr_conf, class_idx, idx = track_item
            label_str = self.class_names[class_idx]
            if label_str not in res:  # Ignore staff members and referees
                continue
            res[label_str][id] = {
                "BBox": [x1, y1, x2, y2],
                "Confidence": curr_conf,
                "Bottom_center": [int((x1 + x2) / 2), y2],
            }

        # Detect the ball
        if 2 in class_labels:
            indices = np.where(class_labels == 2)[0]
            for i, idx in enumerate(indices):
                box = coords[idx].astype(int)
                if "Ball" not in res:
                    res["Ball"] = {}
                res["Ball"][i] = {"BBox": box, "Confidence": conf[idx], "Bottom_center": [int((box[0] + box[2]) / 2), box[3]]}

        return res
