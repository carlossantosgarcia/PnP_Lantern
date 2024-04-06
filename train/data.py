import os

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class ImageNet(Dataset):
    def __init__(self, root, noise_std):
        super().__init__()
        self.root = root
        self.files = os.listdir(root)
        self.length = len(self.files)
        self.idx_to_path = dict(zip(range(self.length), self.files))
        self.noise_std = noise_std
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                # transforms.Normalize((0.445), (0.269)),
                transforms.RandomCrop(size=(64, 64)),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.idx_to_path[idx])
        data = Image.open(path).convert("RGB")
        data = self.transforms(data)
        eps = (self.noise_std / 255) * torch.randn_like(data)
        normalized_noise = eps
        noisy = data + normalized_noise
        return noisy, data


def create_dataloaders(
    dataset_name: str, root: str, batch_size: int, noise_std: float
):

    if dataset_name.lower() == "imagenet":
        dataset = ImageNet(root=root, noise_std=noise_std)
        train_length, val_length = int(0.65 * dataset.__len__()), int(
            0.15 * dataset.__len__()
        )
        test_length = dataset.__len__() - train_length - val_length
    else:
        raise NotImplementedError

    train_ds, val_ds, test_ds = random_split(
        dataset=dataset,
        lengths=[train_length, val_length, test_length],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    test_dataloader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, val_dataloader, test_dataloader


class ImageNetDataModule(LightningDataModule):
    def __init__(self, dataset_name, root, batch_size, noise_std):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.dataloaders = create_dataloaders(
            dataset_name=dataset_name,
            root=root,
            batch_size=batch_size,
            noise_std=noise_std,
        )

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]

    def test_dataloader(self):
        return self.dataloaders[2]
