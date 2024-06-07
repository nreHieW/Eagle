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
        pred_size=(135, 240),
        use_calibrated: bool = False,
        num_keypoints: int = 57,
    ):
        self.dataset = load_dataset("nreHieW/SoccerNet_Field_Keypoints", split=split, cache_dir="data")
        self.transform = transform
        self.pred_size = pred_size
        self.height, self.width = img_size
        self.image_size = img_size
        self.kp_name = "calibrated_keypoints" if use_calibrated else "keypoints"
        self.num_keypoints = num_keypoints

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        image = np.array(image)
        keypoints = sample[self.kp_name][: self.num_keypoints]

        present_class_labels = [i for i, kp in enumerate(keypoints) if kp is not None]
        present_keypoints = [kp for kp in keypoints if kp is not None]

        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=present_keypoints,
                class_labels=present_class_labels,
            )
            image = transformed["image"]
            transformed_keypoints = [kp for kp in transformed["keypoints"]]
            visible_labels = transformed["class_labels"]
            keypoints = [([] if i not in visible_labels else transformed_keypoints[visible_labels.index(i)]) for i in range(self.num_keypoints)]
        else:
            keypoints = [([] if i not in present_class_labels else present_keypoints[present_class_labels.index(i)]) for i in range(self.num_keypoints)]

        return image, [[x] if len(x) > 0 else [] for x in keypoints]  # force each keypoint to be a separate channel

    @staticmethod
    def collate_fn(data):
        """custom collate function for use with the torch dataloader

        Note that it could have been more efficient to padd for each channel separately, but it's not worth the trouble as even
        for 100 channels with each 100 occurances the padded data size is still < 1kB..

        Args:
            data: list of tuples (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list(? keypoints ))

        Returns:
            (images, keypoints); Images as a torch tensor Nx3xHxW,
            keypoints is a nested list of lists. where each item is a tensor (K,2) with K the number of keypoints
            for that channel and that sample:

                List(List(Tensor(K,2))) -> C x N x Tensor(max_keypoints_for_any_channel_in_batch x 2)

        Note there is no padding, as all values need to be unpacked again in the detector to create all the heatmaps,
        unlike e.g. NLP where you directly feed the padded sequences to the network.
        """
        images, keypoints = zip(*data)
        # convert the list of keypoints to a 2D tensor
        keypoints = [[torch.tensor(x) for x in y] for y in keypoints]
        # reorder to have the different keypoint channels as  first dimension
        # C x N x K x 2 , K = variable number of keypoints for each (N,C)
        reordered_keypoints = [[keypoints[i][j] for i in range(len(keypoints))] for j in range(len(keypoints[0]))]

        images = torch.stack(images)

        return images, reordered_keypoints


if __name__ == "__main__":
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    train_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    train_dataset = KeypointsDataset(
        "train[:1%]",
        transform=train_transform,
    )
    print(len(train_dataset))

    item = train_dataset[0]
    image, keypoints = item
    print(keypoints)
    print("\n\nDataloader\n\n")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=KeypointsDataset.collate_fn,
    )
    for i, (images, keypoints) in enumerate(train_loader):
        # print(images.shape)
        # print(keypoints)
        # print(keypoints[0])
        # print(keypoints[0].shape)
        break
