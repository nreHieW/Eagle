import pytorch_lightning as pl
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset import KeypointsDataset
from model import KeypointDetector
from hrnet import get_hrnet_model
from utils import RelativeEarlyStopping, get_config


def test_architecture(model, cfg):
    model.train()
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} requires grad is False"

    x = torch.randn(2, 3, cfg.train_size[0], cfg.train_size[1])
    y = model(x)
    loss = y.mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} grad is None"

    model.zero_grad()


def main():
    cfg = get_config()
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(540, 960, scale=(0.8, 1), p=0.5),
            A.Downscale(scale_min=0.35, scale_max=0.9, p=0.5, interpolation=cv2.INTER_LINEAR),
            A.GaussNoise(p=0.7),
            A.MotionBlur(p=0.4),
            A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.2, p=0.5),
            A.ColorJitter(p=0.4),
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    valid_transform = A.Compose(
        [A.Normalize(), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    train_dataset = KeypointsDataset(
        "train[:1%]",
        transform=train_transform,
        use_calibrated=cfg.use_calibrated,
    )
    valid_dataset = KeypointsDataset(
        "val[:1%]",
        transform=valid_transform,
        use_calibrated=cfg.use_calibrated,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=KeypointsDataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        collate_fn=KeypointsDataset.collate_fn,
    )
    backbone = get_hrnet_model(True)
    num_steps = len(train_loader) * cfg.num_epochs
    model = KeypointDetector(
        heatmap_sigma=cfg.heatmap_sigma,
        maximal_gt_keypoint_pixel_distances="2 4",
        minimal_keypoint_extraction_pixel_distance=1,
        learning_rate=cfg.lr,
        num_steps=num_steps,
        backbone=backbone,
        keypoint_channel_configuration=[[str(x)] for x in range(57)],  # every keypoint is a separate channel
        ap_epoch_start=1,
        ap_epoch_freq=2,
        max_keypoints=20,
    )

    wandb.init(
        name=cfg.run_name,
        project=cfg.project_name,
    )
    early_stopping = RelativeEarlyStopping(
        monitor="validation/epoch_loss",
        patience=int(0.125 * cfg.num_epochs),
        min_relative_delta=0.01,
        # verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(max_epochs=cfg.num_epochs, precision=cfg.precision, logger=WandbLogger(), callbacks=[early_stopping, LearningRateMonitor(logging_interval="epoch")])
    trainer.fit(model, train_loader, valid_loader)

    torch.save(model.state_dict(), "keypoints.pth")


if __name__ == "__main__":
    main()
