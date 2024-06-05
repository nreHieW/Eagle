from ultralytics import YOLO
import cv2
import numpy as np
from ..utils.pitch import GROUND_TRUTH_POINTS, INTERSECTION_TO_PITCH_POINTS
from tqdm import tqdm


class CoordinateModel:

    def __init__(self):
        # TODO: Support GPU inference
        self.keypoint_model = YOLO("eagle/models/weights/keypoint_detector.onnx", task="pose", verbose=False)
        self.detector_model = YOLO("eagle/models/weights/detector_medium.onnx", task="detect", verbose=False)
        self.class_names = {0: "Player", 1: "Goalkeeper", 2: "Ball", 3: "Referee", 4: "Staff members"}

    def get_coordinates(self, frames: np.ndarray, fps: int, num_homography: int = 1, verbose: bool = True):
        """
        Get the coordinates of the players, goalkeepers and the ball
        :param frames: Input frames read using CV2 in BGR format
        :param fps: Frames per second of the video
        :param num_homography: Number of times per second to calculate the homography matrix
        :param verbose: Whether to show the progress bar

        :return: dictionary containing the image coordinates of players, goalkeepers and the ball
        """
        out = []
        for i, frame in tqdm(enumerate(frames), desc="Processing Frames", total=len(frames)) if verbose else enumerate(frames):
            if i % (fps // num_homography) == 0:
                _, _, homography_matrix = self.detect_keypoints(frame)
            objects = self.detect_objects(frame)
            res = {}
            for label, coords in objects.items():
                if len(coords) > 0:
                    coords = np.array([coords], dtype=np.float32)
                    transformed_coords = cv2.perspectiveTransform(coords, homography_matrix)[0]
                    res[label] = transformed_coords.tolist()
                else:
                    res[label] = []
            out.append(res)

        return out

    def detect_keypoints(self, frame: np.ndarray):
        """
        Detect keypoints and returns the homography matrix
        :param frame: Input frame read using CV2 in BGR format

        :return: tuple of img_pts, gt_pts and homography_matrix
        """
        keypoint_pred = self.keypoint_model(frame, verbose=False)
        keypoints = keypoint_pred[0].keypoints
        conf = keypoints.conf[0]
        points = keypoints.xy[0].cpu().numpy()
        img_pts = []
        gt_pts = []
        for i, p in enumerate(points):
            if conf[i] < 0.5:
                continue
            x, y = p.astype(int)
            img_pts.append((x, y))
            gt_pts.append(GROUND_TRUTH_POINTS[INTERSECTION_TO_PITCH_POINTS[i]])
        img_pts = np.array(img_pts)
        gt_pts = np.array(gt_pts)
        homography_matrix, _ = cv2.findHomography(img_pts, gt_pts)

        return img_pts, gt_pts, homography_matrix

    def detect_objects(self, frame: np.ndarray):
        """
        Detect objects in the frame and return the bounding boxes
        :param frame: Input frame read using CV2 in BGR format

        :return: dictionary containing the image coordinates of players, goalkeepers and the ball
        """
        detector_pred = self.detector_model(frame, verbose=False, conf=0.1)
        boxes = detector_pred[0].boxes
        coords = boxes.xyxy
        conf = boxes.conf
        class_labels = boxes.cls
        res = {"Player": [], "Goalkeeper": [], "Ball": []}
        for i, box in enumerate(coords):
            if conf[i] < 0.5:
                continue
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            label = class_labels[i].item()
            label = self.class_names[int(label)]
            if label in res:
                res[label].append(((x1 + x2) // 2, y2))  # Bottom center
        return res
