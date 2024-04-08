import numpy as np
import cv2
import torch
from enum import Enum
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Config:
    backbone: str = "resnet50"
    num_keypoints: int = 57
    batch_size: int = 8
    val_batch_size: int = 32
    num_workers: int = 1
    lr: float = 0.0001
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    log_batch_interval: int = 100
    train_size: Tuple[int, int] = (540, 960)  # 540, 960
    pred_size: Tuple[int, int] = (135, 240)  # 540, 960
    torch_dtype: str = "bf16"
    sigma: int = 3
    pretrained: bool = True
    heatmap: bool = False


def show_normalized_keypoints(item, verbose: bool = True):
    kp = item["normalized_keypoints"]
    img = cv2.cvtColor(np.array(item["image"]), cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    for i, point in enumerate(kp):
        if point is not None:
            point = (point[0] * width, point[1] * height)
            point = (int(point[0]), int(point[1]))
            img = cv2.circle(img, point, 10, (255, 0, 0), 2)
            if verbose:
                label_position = (point[0] + 10, point[1])
                img = cv2.putText(img, str(i), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img


class Prediction(Enum):
    TP = 1
    FP = 2
    FN = 3
    TN = 4
    WRONG = 5


# https://github.com/tlpss/keypoint-detection/blob/3cbca831b972269b95113cb0fad0e79bf391acff/keypoint_detection/models/metrics.py
def l2_distance(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2))


@torch.no_grad()
def get_metrics(preds, targets, dist_threshold: float = 0.01, confidence_threshold: float = 50):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.reshape(57, -1)
    targets = targets.reshape(57, 2)
    out = []
    dists = []

    # Part 1: Get Classification
    for p_i, t_i in zip(preds, targets):
        if len(p_i) == 2:
            x, y = p_i
            c = 100
        elif len(p_i) == 3:
            x, y, c = p_i
        if c < confidence_threshold:
            p_i = torch.tensor([-1, -1])
        else:
            p_i = torch.tensor([x, y])
        if (p_i < 0).all():
            if (t_i < 0).all():
                out.append(Prediction.TN)
            else:
                out.append(Prediction.FN)
        else:
            if (t_i < 0).all():
                out.append(Prediction.FP)
            else:
                d = l2_distance(p_i, t_i)
                dists.append(d)
                if d <= dist_threshold:
                    out.append(Prediction.TP)
                else:
                    out.append(Prediction.WRONG)

    # Part 2 get precision and recall and map
    tp = len([x for x in out if x == Prediction.TP])
    fp = len([x for x in out if x == Prediction.FP])
    fn = len([x for x in out if x == Prediction.FN])
    tn = len([x for x in out if x == Prediction.TN])
    wrong = len([x for x in out if x == Prediction.WRONG])

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "avg_distance": sum(dists) / len(dists) if len(dists) > 0 else 0,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "wrong": wrong,
    }
