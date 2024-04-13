from datasets import load_dataset
import albumentations as A

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import random

from pitch import *


class KeypointsDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        img_size=(540, 960),
        pred_size=(540, 960),
        sigma=3,
        hr_flip: float = 0.0,
        use_calibrated: bool = False,
        num_keypoints: int = 57,
        is_rcnn: bool = False,
    ):
        self.dataset = load_dataset("nreHieW/SoccerNet_Field_Keypoints", split=split, cache_dir="data")
        self.transform = transform
        self.pred_size = pred_size
        self.sigma = sigma
        self.height, self.width = img_size
        self.image_size = img_size
        self.hr_flip = hr_flip
        self.kp_name = "calibrated_keypoints" if use_calibrated else "keypoints"
        self.num_keypoints = num_keypoints
        self.is_rcnn = is_rcnn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        image = np.array(image)
        keypoints = sample[self.kp_name][: self.num_keypoints]

        if self.hr_flip > 0 and random.random() < self.hr_flip:
            # Horizontal flip
            image, keypoints = self.horizontal_flip(image, keypoints)
        present_class_labels = [i for i, kp in enumerate(keypoints) if kp is not None]
        present_keypoints = torch.stack([torch.tensor(x) for x in keypoints if x is not None])

        if self.transform:
            transformed = self.transform(image=image, keypoints=present_keypoints, class_labels=present_class_labels)
            image = transformed["image"]
            transformed_keypoints = torch.tensor(transformed["keypoints"])
            visible_labels = transformed["class_labels"]
            keypoints = torch.full((self.num_keypoints, 2), -1.0)
            keypoints[visible_labels] = transformed_keypoints
        else:
            keypoints = torch.full((self.num_keypoints, 2), -1.0)
            keypoints[present_class_labels] = transformed_keypoints

        mask = (keypoints != -1).all(-1).int()
        keypoints = (keypoints + 1) * mask.unsqueeze(-1) - 1
        heatmaps = self.create_heatmaps(keypoints) * mask.unsqueeze(-1).unsqueeze(-1)
        normalized_keypoints = keypoints / torch.tensor([self.width, self.height])
        normalized_keypoints = torch.where(normalized_keypoints == 0, torch.tensor(-1), normalized_keypoints)
        if self.is_rcnn:
            inv_mask = 1 - mask
            x_min = torch.min(keypoints[:, 0])
            y_min = torch.min(keypoints[:, 1])
            x_max = torch.max(keypoints[:, 0])
            y_max = torch.max(keypoints[:, 1])
            return {
                "image": image,
                "keypoints": torch.cat([keypoints, inv_mask.unsqueeze(-1)], dim=-1),
                "normalized_keypoints": torch.cat([normalized_keypoints, inv_mask.unsqueeze(-1)], dim=-1),
                "boxes": torch.tensor([[x_min, y_min, x_max, y_max]]),
                "mask": mask,
                "labels": torch.tensor([1]),
            }
        else:
            return {
                "image": image,
                "keypoints": keypoints,
                "normalized_keypoints": normalized_keypoints,
                "heatmaps": heatmaps,
                "mask": mask,
                "test": self._generate_target(keypoints) * mask.unsqueeze(-1).unsqueeze(-1),
            }

    # https://github.com/NikolasEnt/soccernet-calibration-sportlight/blob/8255f5044bc7f2ef4f77e9c1dc67cf0861045290/src/models/hrnet/loss.py
    def gaussian(self, x, mu: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.exp(-(torch.div(x - mu.unsqueeze(-1), sigma) ** 2) / 2.0)

    def create_heatmaps(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Create Gaussian distribution heatmaps for keypoints.

        Each heatmap is drawn on an individual channel.

        Args:
            keypoints (torch.Tensor): An array of N points, each point is (x, y).
                Expected shape: (N, 2).

        Returns:
            torch.Tensor: Resulted Gaussian heatmaps: (N, H, W).
        """
        h, w = self.pred_size
        device = keypoints.device
        x = keypoints[:, 0] / self.width * w
        y = keypoints[:, 1] / self.height * h

        x_range = torch.arange(0, w, device=device, dtype=torch.float32)
        y_range = torch.arange(0, h, device=device, dtype=torch.float32)
        gauss_x = self.gaussian(x_range, x, self.sigma)
        gauss_y = self.gaussian(y_range, y, self.sigma)
        heatmaps = torch.einsum("NW, NH -> NHW", gauss_x, gauss_y)

        return heatmaps

    def _generate_target(self, keypoints):
        target = np.zeros((self.num_keypoints, self.pred_size[0], self.pred_size[1]), dtype=np.float32)

        tmp_size = self.sigma * 3

        for cls_id in range(self.num_keypoints):
            curr = keypoints[cls_id]
            # curr_x, curr_y = curr
            # curr = [curr_x / self.width * self.pred_size[1], curr_y / self.height * self.pred_size[0]]
            feat_stride = np.asarray(self.image_size) / np.asarray(self.pred_size)
            mu_x = int(curr[0] / feat_stride[0] + 0.5)
            mu_y = int(curr[1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.pred_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.pred_size[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.pred_size[1])
            img_y = max(0, ul[1]), min(br[1], self.pred_size[0])
            target[cls_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[g_y[0] : g_y[1], g_x[0] : g_x[1]]

        return torch.tensor(target)

    def horizontal_flip(self, image, keypoints):
        image = cv2.flip(image, 1)

        flipped_keypoints = [None] * len(keypoints)
        behind_goal = self.is_behind_goal(keypoints)
        if behind_goal:
            mapping_fn = self.map_behind_goal
        else:
            mapping_fn = self.map_side
        for i, point in enumerate(keypoints):
            if point is None:
                continue
            old_class = INTERSECTION_TO_PITCH_POINTS[i]
            new_class = mapping_fn(old_class)
            flipped_keypoints[PITCH_POINTS_TO_INTERSECTION[new_class]] = [self.width - point[0], point[1]]
        return image, flipped_keypoints

    def map_side(self, input_point):
        mapping = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL", "L": "R", "R": "L", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return "_".join(mapping.get(part, part) for part in input_point.split("_"))

    def map_behind_goal(self, input_point):
        mapping = {
            "T": "B",
            "B": "T",
            "TL": "BL" if "POST" not in input_point else "TR",
            "BL": "TL" if "POST" not in input_point else "BR",
            "TR": "BR" if "POST" not in input_point else "TL",
            "BR": "TR" if "POST" not in input_point else "BL",
        }

        return "_".join(
            [
                mapping.get(part, part)
                for part in input_point.split("_")
                if "PENALTY_MARK" not in input_point or part not in mapping
            ]
        )

    def is_behind_goal(self, keypoints):
        n_total = 0
        n_horizontal = 0
        for p1_idx, p2_idx in PERP_LINES:
            if keypoints[p1_idx] is not None and keypoints[p2_idx] is not None:
                n_total += 1

                p1 = keypoints[p1_idx]
                p2 = keypoints[p2_idx]
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])

                if dy < 1.0 or dx / dy > 10:
                    n_horizontal += 1
        return n_horizontal > 0 and n_horizontal / n_total > 0.5
