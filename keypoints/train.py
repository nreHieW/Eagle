import pytorch_lightning as pl
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import cv2

from utils import Config
from dataset import KeypointsDataset
from models import get_model


def get_config() -> Config:
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="hrnet")
    parser.add_argument("--num_keypoints", type=int, default=57)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--log_batch_interval", type=int, default=100)
    parser.add_argument("--train_size", type=int, nargs=2, default=(540, 960))
    parser.add_argument("--pred_size", type=int, nargs=2, default=(135, 240))
    parser.add_argument("--torch_dtype", type=str, default="bf16")
    parser.add_argument("--sigma", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--heatmap", action="store_true")

    args = parser.parse_args()
    return Config(**vars(args))


def main():
    cfg = get_config()
    train_transform = A.Compose(
        [
            A.Resize(cfg.train_size[0], cfg.train_size[1]),
            A.Downscale(scale_min=0.35, scale_max=0.9, p=0.5, interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.GaussNoise(p=0.7),
            A.MotionBlur(p=0.4),
            A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.2, p=0.5),
            A.HueSaturationValue(p=0.1),
            A.ColorJitter(p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    valid_transform = A.Compose(
        [A.Resize(cfg.train_size[0], cfg.train_size[1]), A.Normalize(), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    train_dataset = KeypointsDataset(
        "train[:10%]",
        transform=train_transform,
        img_size=cfg.train_size,
        pred_size=cfg.pred_size,
        hr_flip=0.5,
        sigma=cfg.sigma,
    )
    valid_dataset = KeypointsDataset(
        "val[:10%]",
        transform=valid_transform,
        img_size=cfg.train_size,
        pred_size=cfg.pred_size,
        hr_flip=0.5,
        sigma=cfg.sigma,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = get_model(cfg)
    if cfg.torch_dtype == "bf16":
        lightning_dtype = "bf16"
    elif cfg.torch_dtype == "fp32":
        lightning_dtype = 32
    elif cfg.torch_dtype == "fp16":
        lightning_dtype = 16
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs, precision=lightning_dtype)
    trainer.fit(model, train_loader, valid_loader)

    torch.save(model.state_dict(), f"keypoints_{cfg.backbone}.pth")


if __name__ == "__main__":
    main()
