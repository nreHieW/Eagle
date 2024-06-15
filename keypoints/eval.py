from dataset import KeypointsDataset
import torch
import torch.nn as nn
import time
from metrics import *
import cv2
from ultralytics import YOLO
from hrnet import get_hrnet_model
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
from heatmap import get_keypoints_from_heatmap_batch_maxpool
from dataclasses import dataclass
import math
import json

DIAGONAL = (960**2 + 540**2) ** 0.5  # diagonal of the image


@dataclass
class ClassificationInput:

    u: int
    v: int
    probability: float
    class_id: int
    gt_u: int
    gt_v: int

    def l2_distance(self):
        if self.gt_u is None or self.gt_v is None or self.u is None or self.v is None:
            return float("inf")
        return math.sqrt((self.u - self.gt_u) ** 2 + (self.v - self.gt_v) ** 2)


class ClassificationMetric:
    def __init__(self, thresholds: List[int] = [2, 4, 8, 12]):
        super().__init__()
        self.data = {}

        for threshold in thresholds:
            self.data[f"true_positives_{threshold}"] = 0  # Keypoints that are correctly detected
            self.data[f"false_positives_{threshold}"] = 0  # Keypoints that are detected but not in the ground truth
            self.data[f"false_negatives_{threshold}"] = 0  # Keypoints that are in the ground truth but not detected

        self.prediction_conf_threshold = 0.01
        self.thresholds = thresholds
        self.pdj = []  # Percentage of Detected Joints

    def update(self, classification_inputs: List[ClassificationInput]):
        for x in classification_inputs:
            dist = x.l2_distance()
            if x.probability < self.prediction_conf_threshold:
                if x.gt_u is not None and x.gt_v is not None:
                    # False negative : Keypoint is in the ground truth but not detected
                    for threshold in self.thresholds:
                        self.data[f"false_negatives_{threshold}"] += 1
            else:
                # True positive : Keypoint is correctly detected
                for threshold in self.thresholds:
                    if dist < threshold:
                        self.data[f"true_positives_{threshold}"] += 1
                    else:
                        # False positive : Keypoint is detected but not in the ground truth
                        self.data[f"false_positives_{threshold}"] += 1

            # Percentage of Detected Joints
            if dist < DIAGONAL * 0.05:  # https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78
                self.pdj.append(1)
            else:
                self.pdj.append(0)

    def compute(self):
        res = {}
        for threshold in self.thresholds:
            true_positives = self.data[f"true_positives_{threshold}"]
            false_positives = self.data[f"false_positives_{threshold}"]
            false_negatives = self.data[f"false_negatives_{threshold}"]
            precision = true_positives / (true_positives + false_positives + 1e-5)
            recall = true_positives / (true_positives + false_negatives + 1e-5)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            res[f"precision_{threshold}"] = precision
            res[f"recall_{threshold}"] = recall
            res[f"f1_{threshold}"] = f1
            res[f"true_positives_{threshold}"] = true_positives
            res[f"false_positives_{threshold}"] = false_positives
            res[f"false_negatives_{threshold}"] = false_negatives
        res["pdj"] = sum(self.pdj) / len(self.pdj)
        return res


class InferenceModel(nn.Module):
    def __init__(self, n_heatmaps):
        super(InferenceModel, self).__init__()
        backbone = get_hrnet_model(False)
        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )
        self.unnormalized_model = nn.Sequential(
            backbone,
            head,
        )
        self.n_heatmaps = n_heatmaps

    def forward(self, x: torch.Tensor):
        """
        x shape must be of shape (N,3,H,W)
        returns tensor with shape (N, n_heatmaps, H,W)
        """
        return torch.sigmoid(self.forward_unnormalized(x))

    def forward_unnormalized(self, x: torch.Tensor):
        return self.unnormalized_model(x)


def main():
    res = {}
    test_dataset = KeypointsDataset(
        "test",
    )
    model = YOLO("/Users/weihern/Documents/Sports Analytics/Eagle/eagle/models/weights/keypoint_detector.onnx", task="pose", verbose=False)
    metric = KeypointAPMetrics([2, 4, 8, 12])
    classification_metric = ClassificationMetric([2, 4, 8, 12])
    yolo_times = []
    for i in tqdm(range(len(test_dataset))):
        # for i in range(10):
        images, keypoints = test_dataset[i]
        img_array = images
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        start = time.time()
        pred = model(img_array, verbose=False)[0].keypoints
        yolo_times.append(time.time() - start)
        conf = pred.conf[0]
        points = pred.xy[0].cpu().numpy()
        gt_keypoints = []
        detected_keypoints = []
        classification_inputs = []
        for i in range(len(keypoints)):
            gt = keypoints[i]
            detected_keypoints.append(
                DetectedKeypoint(
                    u=points[i][0],
                    v=points[i][1],
                    probability=conf[i],
                )
            )
            if len(gt) != 0:
                gt_keypoints.append(
                    Keypoint(
                        u=gt[0][0],
                        v=gt[0][1],
                    )
                )
                classification_inputs.append(
                    ClassificationInput(
                        u=points[i][0],
                        v=points[i][1],
                        gt_u=gt[0][0],
                        gt_v=gt[0][1],
                        probability=conf[i],
                        class_id=i,
                    )
                )
            else:
                classification_inputs.append(
                    ClassificationInput(
                        u=points[i][0],
                        v=points[i][1],
                        gt_u=None,
                        gt_v=None,
                        probability=conf[i],
                        class_id=i,
                    )
                )
        metric.update(detected_keypoints, gt_keypoints)
        classification_metric.update(classification_inputs)
    res["YOLO"] = {
        "metrics": metric.compute(),
        "time": sum(yolo_times) / len(yolo_times),
        "classification": classification_metric.compute(),
    }

    hrnet = InferenceModel(57)
    hrnet.load_state_dict(torch.load("/Users/weihern/Documents/Sports Analytics/Eagle/eagle/models/weights/keypoints_main.pth"))
    hrnet.eval().to("mps")
    test_dataset = KeypointsDataset(
        "test",
        transform=A.Compose(
            # [A.Resize(cfg.train_size[0], cfg.train_size[1]), A.Normalize(), ToTensorV2()],
            [A.Normalize(), ToTensorV2()],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        ),
    )
    metric = KeypointAPMetrics([2, 4, 8, 12])
    classification_metric = ClassificationMetric([2, 4, 8, 12])
    hrnet_times = []
    classification_inputs = []
    for i in tqdm(range(len(test_dataset))):
        # for i in range(10):
        images, keypoints = test_dataset[i]
        x = images.unsqueeze(0).to("mps")
        start = time.time()
        out = hrnet(x)
        hrnet_times.append(time.time() - start)
        vals, scores = get_keypoints_from_heatmap_batch_maxpool(out, return_scores=True)
        gt_keypoints = []
        detected_keypoints = []
        for i in range(len(keypoints)):
            gt = keypoints[i]
            if len(gt) != 0:
                gt_keypoints.append(
                    Keypoint(
                        u=gt[0][0],
                        v=gt[0][1],
                    )
                )
            pred = vals[0][i]
            score = scores[0][i]
            if len(pred) == 0:

                if len(gt) != 0:
                    classification_inputs.append(
                        ClassificationInput(
                            u=None,
                            v=None,
                            gt_u=gt[0][0],
                            gt_v=gt[0][1],
                            probability=0,
                            class_id=i,
                        )
                    )
                else:
                    classification_inputs.append(
                        ClassificationInput(
                            u=None,
                            v=None,
                            gt_u=None,
                            gt_v=None,
                            probability=0,
                            class_id=i,
                        )
                    )
                continue  # No keypoints detected
            pred = pred[0]
            u, v = pred
            score = max(score)
            detected_keypoints.append(
                DetectedKeypoint(
                    u=u,
                    v=v,
                    probability=score,
                )
            )

            if len(gt) != 0:
                classification_inputs.append(
                    ClassificationInput(
                        u=u,
                        v=v,
                        gt_u=gt[0][0],
                        gt_v=gt[0][1],
                        probability=score,
                        class_id=i,
                    )
                )
            else:
                classification_inputs.append(
                    ClassificationInput(
                        u=u,
                        v=v,
                        gt_u=None,
                        gt_v=None,
                        probability=score,
                        class_id=i,
                    )
                )
        metric.update(detected_keypoints, gt_keypoints)
        classification_metric.update(classification_inputs)
    res["HRNet"] = {
        "metrics": metric.compute(),
        "time": sum(hrnet_times) / len(hrnet_times),
        "classification": classification_metric.compute(),
    }
    for k, v in res.items():
        print(f"{k}: {v}")

    with open("results.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
