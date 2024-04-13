import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

from typing import Tuple
from hrnet import get_hrnet_model
import pytorch_lightning as pl
from utils import Config, get_metrics

# TODO: Loss function pred * mask
# TODO: Loss function exp/sigmoid
# TODO: Loss function pixel wise cross entropy


def get_model(cfg: Config):
    if cfg.heatmap:
        return KeypointHeatmapModel(cfg)
    else:
        return KeypointRegressorModel(cfg)


def heatmap_to_preds(heatmaps, img_size: Tuple[int, int] = (540, 960)):
    img_h, img_w = img_size
    _, heatmap_height, heatmap_width = heatmaps.shape
    heatmaps = torch.exp(heatmaps)
    x_prob, x = torch.max(torch.max(heatmaps, dim=1)[0], dim=1)
    y_prob, y = torch.max(torch.max(heatmaps, dim=2)[0], dim=1)
    conf = torch.min(x_prob, y_prob)
    x = x * img_w / heatmap_width
    y = y * img_h / heatmap_height

    predictions = torch.stack([x, y, conf], dim=-1)
    return predictions


class RCNN(pl.LightningModule):
    def __init__(self, cfg: Config):
        anchor_generator = AnchorGenerator(
            sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)
        )
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=cfg.pretrained,
            num_keypoints=cfg.num_keypoints,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
        )

        self.cfg = cfg
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "avg_distance": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "wrong": 0,
        }

    def training_step(self, batch, batch_idx):
        images = [x["image"] for x in batch]
        losses = self.model(images, batch)
        loss = sum(loss for loss in losses.values())
        if batch_idx % self.cfg.log_batch_interval == 0:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = []
        images = [x["image"] for x in batch]
        outputs = self.model(images)
        outputs = outputs[0]["keypoints"][0]
        keypoints = [x["keypoints"] for x in batch]
        for output, keypoint in zip(outputs, keypoints):
            preds = heatmap_to_preds(output, img_size=self.cfg.train_size)
            metrics.append(get_metrics(preds, keypoint))

        for metric in metrics:
            for key, value in metric.items():
                self.metrics[key] += value

        return


class KeypointHeatmapModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.backbone == "hrnet":
            self.model = get_hrnet_model(
                cfg.train_size, cfg.pred_size, num_classes=cfg.num_keypoints, pretrained=cfg.pretrained
            )
        else:
            self.model = KeypointHeatmap(cfg.backbone, cfg.num_keypoints, cfg.pretrained)
        self.cfg = cfg
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "avg_distance": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "wrong": 0,
            "loss": 0,
        }

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        heatmaps = batch["heatmaps"]
        outputs = self.model(images)
        outputs = torch.exp(outputs)
        loss = F.mse_loss(outputs, heatmaps)
        if batch_idx % self.cfg.log_batch_interval == 0:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = []
        images = batch["image"]
        heatmaps = batch["heatmaps"]
        keypoints = batch["keypoints"]
        outputs = self.model(images)
        outputs = torch.exp(outputs)
        loss = F.mse_loss(outputs, heatmaps)
        for output, keypoint in zip(outputs, keypoints):
            preds = heatmap_to_preds(output, img_size=self.cfg.train_size)
            metrics.append(get_metrics(preds, keypoint))

        for metric in metrics:
            for key, value in metric.items():
                self.metrics[key] += value

        self.metrics["loss"] += loss.item()
        return loss

    def validation_epoch_end(self, outputs):
        for key in self.metrics:
            if key not in ["tp", "fp", "fn", "tn", "wrong"]:
                self.metrics[key] /= len(outputs)
        print(self.metrics)
        self.log_dict(self.metrics)
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "avg_distance": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "wrong": 0,
            "loss": 0,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.betas
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


class KeypointRegressorModel(pl.LightningModule):

    def __init__(self, cfg: Config):
        super().__init__()
        self.model = KeypointRegressor(cfg.backbone, cfg.num_keypoints, cfg.pretrained)
        self.cfg = cfg
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "avg_distance": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "wrong": 0,
        }

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        keypoints = batch["normalized_keypoints"]
        keypoints = keypoints.view(-1, 57 * 2)
        outputs = self.model(images)
        loss = F.mse_loss(outputs, keypoints)
        if batch_idx % self.cfg.log_batch_interval == 0:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = []
        images = batch["image"]
        keypoints = batch["normalized_keypoints"]
        keypoints = keypoints.view(-1, 57 * 2)
        outputs = self.model(images)
        loss = F.mse_loss(outputs, keypoints)
        for output, keypoint in zip(outputs, keypoints):
            metrics.append(get_metrics(output, keypoint))

        for metric in metrics:
            for key, value in metric.items():
                self.metrics[key] += value
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.betas
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def validation_epoch_end(self, outputs):
        for key in self.metrics:
            if key not in ["tp", "fp", "fn", "tn", "wrong"]:
                self.metrics[key] /= len(outputs)

        print(self.metrics)
        self.log_dict(self.metrics)
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "avg_distance": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "wrong": 0,
        }


class KeypointRegressor(nn.Module):
    def __init__(self, backbone_name: str, num_keypoints: int = 57, use_pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=use_pretrained, num_classes=num_keypoints * 2)

    def forward(self, x):
        out = self.backbone(x)
        return out


class KeypointHeatmap(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_keypoints: int = 57,
        use_pretrained: bool = True,
        num_deconv_layers: int = 3,
        deconv_channels: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=use_pretrained, features_only=True)
        deconv = []
        channels = self.backbone.feature_info[-1]["num_chs"]
        for _ in range(num_deconv_layers):
            deconv.append(
                nn.ConvTranspose2d(
                    in_channels=channels, out_channels=deconv_channels, kernel_size=4, stride=2, padding=1
                )
            )
            channels = 256
        self.deconv = nn.Sequential(*deconv)
        self.norm = nn.BatchNorm2d(deconv_channels)
        self.head = nn.Conv2d(
            in_channels=channels,
            out_channels=num_keypoints,
            kernel_size=(4, 3),
            padding=1,
        )

    def forward(self, x):
        out = self.backbone(x)[-1]
        out = self.deconv(out)
        out = self.norm(out)
        out = self.head(out)
        out = F.log_softmax(out, dim=1)
        return out
