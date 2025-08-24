import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from ..utils.pitch import GROUND_TRUTH_POINTS, INTERSECTION_TO_PITCH_POINTS, PITCH_POINTS_TO_INTERSECTION, NOT_ON_PLANE
from tqdm import tqdm
from .keypoint_hrnet import KeypointModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from boxmot import BotSort
from pathlib import Path
from collections import Counter

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
BATCH = 4


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device


def find_x_at_y(pt1, pt2, y_target):
    x1, y1 = pt1
    x2, y2 = pt2
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept (c) using y1 = m*x1 + c
    c = y1 - m * x1

    # Find x when y = y_target
    x_target = (y_target - c) / m

    return x_target


class CoordinateModel:

    def __init__(self, keypoint_conf: float = 0.3, detector_conf: float = 0.35):
        device = get_device()
        self.device = device
        print(f"Using {self.device} for inference")

        if device == "cpu":
            self.detector_model = YOLO("eagle/models/weights/detector_medium.onnx", task="detect", verbose=False)  # by default uses the medium model for cpu
        else:
            self.detector_model = YOLO("eagle/models/weights/detector_large_hd.pt").to(device)
        self.keypoint_model = KeypointModel(57).to(device)
        self.keypoint_model.load_state_dict(torch.load("eagle/models/weights/keypoints_main.pth"))
        self.keypoint_model.eval()
        self.class_names = {0: "Player", 1: "Goalkeeper", 2: "Ball", 3: "Referee", 4: "Staff members"}
        self.transforms = A.Compose(  # Define the transformations for the keypoint model
            [A.Resize(540, 960), A.Normalize(), ToTensorV2()],
        )
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # BotSort does not support MPS; fall back to CPU in that case
        tracker_device = 0 if device == "cuda" else ("cpu" if device == "mps" else device)
        self.tracker = BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device=tracker_device,
            half=False,
        )
        self.keypoint_conf = keypoint_conf
        self.detector_conf = detector_conf

    def _build_pitch_groups(self):
        if hasattr(self, "_pitch_groups_built") and self._pitch_groups_built:
            return
        coord_to_label = {}
        x_groups = {}
        y_groups = {}
        for label, (x, y, z) in GROUND_TRUTH_POINTS.items():
            if z != 0.0:
                continue
            xr = round(float(x), 2)
            yr = round(float(y), 2)
            if (xr, yr) not in coord_to_label:
                coord_to_label[(xr, yr)] = label
            x_groups.setdefault(xr, set()).add(label)
            y_groups.setdefault(yr, set()).add(label)
        self._coord_to_label = coord_to_label
        self._x_groups = x_groups
        self._y_groups = y_groups
        self._pitch_groups_built = True

    @staticmethod
    def _fit_line(points: np.ndarray):
        """
        Fit a line using cv2.fitLine. points: (N,2) float32.
        Returns (vx, vy, x0, y0) or None if unstable.
        """
        if points is None or len(points) < 2:
            return None
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(vx)
            vy = float(vy)
            x0 = float(x0)
            y0 = float(y0)
            if abs(vx) + abs(vy) < 1e-6:
                return None
            return vx, vy, x0, y0
        except Exception:
            return None

    @staticmethod
    def _intersect_lines(line1, line2):
        """
        Intersect two infinite lines given by (vx,vy,x0,y0).
        Returns (x,y) or None if parallel.
        """
        if line1 is None or line2 is None:
            return None
        vx1, vy1, x01, y01 = line1
        vx2, vy2, x02, y02 = line2
        det = vx1 * (-vy2) - vy1 * (-vx2)
        if abs(det) < 1e-8:
            return None
        rhs = np.array([x02 - x01, y02 - y01], dtype=np.float64)
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
        try:
            t, _ = np.linalg.solve(A, rhs)
            x = x01 + t * vx1
            y = y01 + t * vy1
            return float(x), float(y)
        except Exception:
            return None

    def _synthesize_keypoints_with_line_intersections(self, frame_shape, keypoints: dict, min_points_per_line: int = 2, max_new_points: int = 30) -> dict:
        """
        Augment detected keypoints using line fitting on world-horizontal and world-vertical line families.
        Only uses on-plane points (z==0). Returns merged dict.
        """
        self._build_pitch_groups()
        height, width = frame_shape[:2]
        detected = {k: v for k, v in keypoints.items() if PITCH_POINTS_TO_INTERSECTION.get(k, -1) not in NOT_ON_PLANE}

        # Fit horizontal lines (same world Y)
        lines_y = {}
        for y_val, labels in self._y_groups.items():
            pts = [detected[lbl] for lbl in labels if lbl in detected]
            if len(pts) >= min_points_per_line:
                line = self._fit_line(np.array(pts, dtype=np.float32))
                if line is not None:
                    lines_y[y_val] = line

        # Fit vertical lines (same world X)
        lines_x = {}
        for x_val, labels in self._x_groups.items():
            pts = [detected[lbl] for lbl in labels if lbl in detected]
            if len(pts) >= min_points_per_line:
                line = self._fit_line(np.array(pts, dtype=np.float32))
                if line is not None:
                    lines_x[x_val] = line

        # Intersections of fitted horizontal and vertical lines → synthetic keypoints
        added = {}
        for y_val, ly in lines_y.items():
            for x_val, lx in lines_x.items():
                label = self._coord_to_label.get((round(float(x_val), 2), round(float(y_val), 2)))
                if not label or label in keypoints:
                    continue
                pt = self._intersect_lines(ly, lx)
                if pt is None:
                    continue
                xi = int(round(pt[0]))
                yi = int(round(pt[1]))
                added[label] = (xi, yi)
                if len(added) >= max_new_points:
                    break
            if len(added) >= max_new_points:
                break
        if added:
            return {**keypoints, **added}
        return keypoints

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
        homography_interval = max(1, int(fps / max(1, num_homography)))
        keypoint_interval = max(1, int(fps / max(1, num_keypoint_detection)))
        prev_gray = None
        prev_keypoints = {}
        res = {}
        mem = {}
        compute_homography = False  # Whether to compute homography matrix outside of the interval
        homography_matrix = None
        prev_homography_matrix = None

        try:
            indices = list(range(0, len(frames), keypoint_interval))
            if len(indices) > 0:
                batched = []
                batched_idx = []
                for idx in indices:
                    img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                    batched.append(self.transforms(image=img)["image"].to(self.keypoint_model.unnormalized_model[1].weight.data.device).float())
                    batched_idx.append(idx)
                    if len(batched) == BATCH:
                        batch_tensor = torch.stack(batched, dim=0)
                        kp_list = self.keypoint_model.get_keypoints(batch_tensor)
                        for k, frame_idx in enumerate(batched_idx):
                            # Postprocess normalize→image coords, and dedup like detect_keypoints
                            height, width = frames[frame_idx].shape[:2]
                            tmp = {}
                            for i_lab, x_n, y_n, score in kp_list[k]:
                                if score < self.keypoint_conf:
                                    continue
                                label = INTERSECTION_TO_PITCH_POINTS[i_lab]
                                xi = int(x_n * width)
                                yi = int(y_n * height)
                                tmp[label] = (xi, yi, score, i_lab)
                            vals = list(tmp.values())
                            coords = [x[:2] for x in vals]
                            counts = Counter(coords)
                            coords_to_label = {}
                            for kk, vv in tmp.items():
                                if counts[vv[:2]] == 1:
                                    coords_to_label[vv[:2]] = kk
                                else:
                                    if vv[2] == max([xv[2] for xv in vals if xv[:2] == vv[:2]]):
                                        coords_to_label[vv[:2]] = kk
                            mem[frame_idx] = {coords_to_label[kp]: kp for kp in coords_to_label}
                        batched = []
                        batched_idx = []
                if len(batched) > 0:
                    batch_tensor = torch.stack(batched, dim=0)
                    kp_list = self.keypoint_model.get_keypoints(batch_tensor)
                    for k, frame_idx in enumerate(batched_idx):
                        height, width = frames[frame_idx].shape[:2]
                        tmp = {}
                        for i_lab, x_n, y_n, score in kp_list[k]:
                            if score < self.keypoint_conf:
                                continue
                            label = INTERSECTION_TO_PITCH_POINTS[i_lab]
                            xi = int(x_n * width)
                            yi = int(y_n * height)
                            tmp[label] = (xi, yi, score, i_lab)
                        vals = list(tmp.values())
                        coords = [x[:2] for x in vals]
                        counts = Counter(coords)
                        coords_to_label = {}
                        for kk, vv in tmp.items():
                            if counts[vv[:2]] == 1:
                                coords_to_label[vv[:2]] = kk
                            else:
                                if vv[2] == max([xv[2] for xv in vals if xv[:2] == vv[:2]]):
                                    coords_to_label[vv[:2]] = kk
                        mem[frame_idx] = {coords_to_label[kp]: kp for kp in coords_to_label}
        except Exception:
            pass  # If batching fails for any reason, fall back to on-demand detection below
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
                        if len(prev_keypoints) > 0:
                            for j in range(j - 1, i - 1, -1):
                                prev_frame = frames[j]
                                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                                flowed = self.calculate_optical_flow(prev_frame, prev_gray, prev_keypoints, next_gray)
                                prev_keypoints = flowed if len(flowed) > 0 else prev_keypoints
                                # combine
                                mem[j] = {**prev_keypoints, **mem.get(j, {})}
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

            keypoints = {**keypoints, **mem.get(i, {})}
            # Augment with geometry-based synthesis to improve recall
            if len(keypoints) >= 2:
                keypoints = self._synthesize_keypoints_with_line_intersections(frame.shape, keypoints)
            if calibration:
                keypoints = self.calibrate_keypoints(frame, keypoints)
            prev_keypoints = keypoints
            prev_gray = curr_gray
            objects = self.detect_objects(frame)
            if i % homography_interval == 0 or compute_homography:
                # Use only on-plane points for homography
                img_pts = []
                world_pts = []
                used_labels = []
                for label, (xi, yi) in keypoints.items():
                    idx = PITCH_POINTS_TO_INTERSECTION.get(label, -1)
                    if idx in NOT_ON_PLANE:
                        continue
                    wx, wy, wz = GROUND_TRUTH_POINTS[label]
                    if wz != 0.0:
                        continue
                    img_pts.append([xi, yi])
                    world_pts.append([wx, wy])
                    used_labels.append(label)
                img_pts = np.array(img_pts, dtype=np.float32)
                world_pts = np.array(world_pts, dtype=np.float32)
                if len(img_pts) < 4:
                    compute_homography = True
                    # raise ValueError("Not enough keypoints detected to compute homography matrix")
                else:
                    for method in [cv2.RANSAC, cv2.RHO, cv2.LMEDS]:
                        new_homography_matrix, mask = cv2.findHomography(img_pts, world_pts, method, 5.0 if method is cv2.RANSAC else None)
                        if new_homography_matrix is not None:
                            break  # Exit the loop if a homography is found
                    if new_homography_matrix is not None:
                        if mask is not None and mask.size == len(used_labels):
                            # Filter current keypoints to only inliers for stability
                            keypoints = {k: v for k, v, m in zip(used_labels, img_pts.tolist(), mask.flatten()) if m}
                            prev_keypoints = keypoints
                        homography_matrix = new_homography_matrix
                        prev_homography_matrix = homography_matrix
                        compute_homography = False
                    else:
                        compute_homography = True  # For this frame, use the previous homography matrix but compute a new one next frame

            indiv = {}  # Coordinate information at this current frame
            for class_name, class_dict in objects.items():
                for obj_id, obj_dict in class_dict.items():
                    bottom_center = obj_dict["Bottom_center"]
                    bbox_coords = np.array(obj_dict["BBox"], dtype=np.uint16).tolist()
                    conf = obj_dict["Confidence"]
                    if homography_matrix is None and prev_homography_matrix is not None:
                        H_use = prev_homography_matrix
                    else:
                        H_use = homography_matrix
                    if H_use is None:
                        curr = {int(obj_id): {"BBox": bbox_coords, "Confidence": conf, "Transformed_Coordinates": None, "Image_Bottom_center": bottom_center}}
                    else:
                        coords = np.array([[bottom_center]], dtype=np.float32)  # Dim needs to be 3
                        transformed_coords = cv2.perspectiveTransform(coords, H_use)[0].astype(int)
                        tx, ty = transformed_coords[0, 0], transformed_coords[0, 1]
                        if tx < 0 or tx > PITCH_WIDTH or ty < 0 or ty > PITCH_HEIGHT:
                            curr = {int(obj_id): {"BBox": bbox_coords, "Confidence": conf, "Transformed_Coordinates": None, "Image_Bottom_center": bottom_center}}
                        else:
                            curr = {int(obj_id): {"BBox": bbox_coords, "Confidence": conf, "Transformed_Coordinates": transformed_coords.tolist()[0]}}
                    if class_name not in indiv:
                        indiv[class_name] = curr
                    else:
                        indiv[class_name].update(curr)

            # Find the visible area of the pitch at each frame. Convert image coordinates to pitch coordinates
            # 0, 0 is top left in cv2
            height, width = frame.shape[:2]
            top_left = top_right = bottom_left = bottom_right = None
            H_use = homography_matrix if homography_matrix is not None else prev_homography_matrix
            if H_use is not None:
                top_left = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), H_use)[0].astype(int)[0].tolist()
                top_right = cv2.perspectiveTransform(np.array([[[width, 0]]], dtype=np.float32), H_use)[0].astype(int)[0].tolist()
                bottom_left = cv2.perspectiveTransform(np.array([[[0, height]]], dtype=np.float32), H_use)[0].astype(int)[0].tolist()
                bottom_right = cv2.perspectiveTransform(np.array([[[width, height]]], dtype=np.float32), H_use)[0].astype(int)[0].tolist()

            boundaries = [None, None, None, None]
            if top_left is not None and top_right is not None and bottom_left is not None and bottom_right is not None:
                try:
                    top_left = (find_x_at_y(top_left, bottom_left, PITCH_HEIGHT), PITCH_HEIGHT)
                    top_right = (find_x_at_y(top_right, bottom_right, PITCH_HEIGHT), PITCH_HEIGHT)
                    bottom_left = (find_x_at_y(bottom_left, top_left, 0), 0)
                    bottom_right = (find_x_at_y(bottom_right, top_right, 0), 0)
                    boundaries = [bottom_left, top_left, top_right, bottom_right]
                except Exception:
                    pass
            res[i] = {"Coordinates": indiv, "Time": f"{i // fps // 60:02d}:{i // fps % 60:02d}", "Keypoints": prev_keypoints, "Boundaries": boundaries}

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
        if prev_gray is None or curr_gray is None or prev_keypoints is None or len(prev_keypoints) == 0:
            return {}
        prev_points = np.array(list(prev_keypoints.values()), dtype=np.float32)
        if prev_points.ndim != 2 or prev_points.shape[0] == 0 or prev_points.shape[1] != 2:
            return {}
        try:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.lk_params)
        except Exception:
            return {}
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
            curr_x = np.clip(curr_x, 0, frame.shape[1] - 1)
            curr_y = np.clip(curr_y, 0, frame.shape[0] - 1)
            curr_x_min, curr_x_max = max(0, curr_x - 1), min(frame.shape[1], curr_x + 2)
            curr_y_min, curr_y_max = max(0, curr_y - 1), min(frame.shape[0], curr_y + 2)
            curr_grid = frame[curr_y_min:curr_y_max, curr_x_min:curr_x_max]
            curr_grid = cv2.cvtColor(curr_grid, cv2.COLOR_BGR2HSV)
            avg_hue_curr = np.mean(curr_grid[:, :, 0])

            prev_x, prev_y = point.astype(int)
            prev_x = np.clip(prev_x, 0, frame.shape[1] - 1)
            prev_y = np.clip(prev_y, 0, frame.shape[0] - 1)
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
        # res = {}
        # for i, x, y, score in keypoints:
        #     if score < self.keypoint_conf:
        #         continue
        #     label = INTERSECTION_TO_PITCH_POINTS[i]
        #     res[label] = (int(x * width), int(y * height))

        tmp = {}
        for i, x, y, score in keypoints:
            if score < self.keypoint_conf:
                continue
            label = INTERSECTION_TO_PITCH_POINTS[i]
            tmp[label] = (int(x * width), int(y * height), score, i)
        vals = list(tmp.values())
        coords = [x[:2] for x in vals]
        counts = Counter(coords)
        coords_to_label = {}
        # if multiple keypoints are assigned to the same point choose the highest scoring one
        for k, v in tmp.items():
            if counts[v[:2]] == 1:
                coords_to_label[v[:2]] = k
            else:
                if v[2] == max([x[2] for x in vals if x[:2] == v[:2]]):
                    coords_to_label[v[:2]] = k
        res = {coords_to_label[k]: k for k in coords_to_label}
        return res

    def calibrate_keypoints(self, frame: np.ndarray, keypoints: dict):
        """
        Calibrate keypoints by finding the brightest spot in a grid around the keypoint using HSV color space.
        If the original brightness is sufficient, no adjustment is made.
        :param frame: Input frame read using CV2 in BGR format
        :param keypoints: dictionary containing the keypoints where the key is the pitch point (str) and the value is the image coordinates

        :return: dictionary containing the calibrated keypoints where the key is the pitch point (str) and the value is the image coordinates
        """
        OFFSET = 3
        BRIGHTNESS_THRESHOLD = 150  # Define a threshold for brightness to decide if adjustment is needed
        new_keypoints = {}

        for key, (x, y) in keypoints.items():
            # Skip calibration for out-of-bounds points
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                new_keypoints[key] = (x, y)
                continue
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
        low_conf = min(self.detector_conf, 0.15)
        detector_pred = self.detector_model(frame, verbose=False, conf=low_conf)
        boxes = detector_pred[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        class_labels = boxes.cls.cpu().numpy().astype(int)

        # Track only players, goalkeepers
        res = {"Player": {}, "Goalkeeper": {}}
        try:
            tracks = self.tracker.update(np.hstack((coords, conf.reshape(-1, 1), class_labels.reshape(-1, 1))), frame)
        except Exception as e:
            raise e
        for track_item in tracks:
            x1, y1, x2, y2, id, curr_conf, class_idx, idx = track_item
            class_idx = int(class_idx)
            x1 = int(np.clip(x1, 0, frame.shape[1] - 1))
            y1 = int(np.clip(y1, 0, frame.shape[0] - 1))
            x2 = int(np.clip(x2, 0, frame.shape[1] - 1))
            y2 = int(np.clip(y2, 0, frame.shape[0] - 1))
            label_str = self.class_names.get(class_idx, None)
            if label_str not in res:  # Ignore staff members and referees
                continue
            if float(curr_conf) < self.detector_conf:
                continue
            res[label_str][int(id)] = {
                "BBox": [x1, y1, x2, y2],
                "Confidence": float(curr_conf),
                "Bottom_center": [int((x1 + x2) / 2), y2],
            }

        # Fallback: if tracking yields no players/goalkeepers but detections exist, use raw detections
        if (len(res["Player"]) == 0 and len(res["Goalkeeper"]) == 0) and coords.shape[0] > 0:
            for det_i in range(coords.shape[0]):
                cls = int(class_labels[det_i])
                label_str = self.class_names.get(cls, None)
                if label_str not in res:
                    continue
                x1, y1, x2, y2 = coords[det_i].astype(int)
                x1 = int(np.clip(x1, 0, frame.shape[1] - 1))
                y1 = int(np.clip(y1, 0, frame.shape[0] - 1))
                x2 = int(np.clip(x2, 0, frame.shape[1] - 1))
                y2 = int(np.clip(y2, 0, frame.shape[0] - 1))
                if float(conf[det_i]) < self.detector_conf:
                    continue
                res[label_str][det_i] = {
                    "BBox": [x1, y1, x2, y2],
                    "Confidence": float(conf[det_i]),
                    "Bottom_center": [int((x1 + x2) / 2), y2],
                }

        # Detect the ball
        if 2 in class_labels:
            indices = np.where(class_labels == 2)[0]
            for i, idx in enumerate(indices):
                box = coords[idx].astype(int)
                if "Ball" not in res:
                    res["Ball"] = {}
                if float(conf[idx]) < self.detector_conf:
                    continue
                res["Ball"][i] = {"BBox": box, "Confidence": float(conf[idx]), "Bottom_center": [int((box[0] + box[2]) / 2), box[3]]}
        return res
