import pytorch_lightning as pl
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
import wandb
from pytorch_lightning.loggers import WandbLogger

from dataset import KeypointsDataset
from model import KeypointDetector
from hrnet import get_hrnet_model


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
    train_transform = A.Compose(
        [
            # A.Resize(cfg.train_size[0], cfg.train_size[1]),
            # A.Downscale(scale_min=0.35, scale_max=0.9, p=0.5, interpolation=cv2.INTER_LINEAR),
            # A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            # A.GaussNoise(p=0.7),
            # A.MotionBlur(p=0.4),
            # A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.2, p=0.5),
            # A.HueSaturationValue(p=0.1),
            # A.ColorJitter(p=0.4),
            # A.ShiftScaleRotate(
            #     shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT
            # ),
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    valid_transform = A.Compose(
        # [A.Resize(cfg.train_size[0], cfg.train_size[1]), A.Normalize(), ToTensorV2()],
        [A.Normalize(), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    train_dataset = KeypointsDataset(
        "train[:1%]",
        transform=train_transform,
    )
    valid_dataset = KeypointsDataset(
        "val[:1%]",
        transform=valid_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=KeypointsDataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=KeypointsDataset.collate_fn,
    )
    backbone = get_hrnet_model(True)
    model = KeypointDetector(
        heatmap_sigma=3,
        maximal_gt_keypoint_pixel_distances="2 4",
        minimal_keypoint_extraction_pixel_distance=1,
        learning_rate=3e-4,
        backbone=backbone,
        keypoint_channel_configuration=[[str(x)] for x in range(57)],  # every keypoint is a separate channel
        ap_epoch_start=1,
        ap_epoch_freq=2,
        lr_scheduler_relative_threshold=0.0,
        max_keypoints=20,
    )

    wandb.init(
        name="test",
        project="Eagle",
    )
    trainer = pl.Trainer(max_epochs=100, precision="16", logger=WandbLogger())
    trainer.fit(model, train_loader, valid_loader)

    torch.save(model.state_dict(), "keypoints.pth")


if __name__ == "__main__":
    main()
