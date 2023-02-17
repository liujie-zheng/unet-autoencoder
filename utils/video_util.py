import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms


class RouenVideo(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 8518

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image, 0

class BearVideo(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 4753

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image, 0

def show_sample_frame(dataloader):
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img.detach().cpu().permute(1, 2, 0), cmap="gray")
    plt.axis(False)
    plt.show()

def get_dataloader(batch_size):
    train_dataset = RouenVideo(
        root= "data/frames_rouen",
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32)
        ])
    )
    test_dataset = BearVideo(
        root= "data/frames_bear",
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32)
        ])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataset = RouenVideo(
        root= "../data/frames_rouen",
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
        ])
    )
    train_dataloader = DataLoader(train_dataset, 64)
    show_sample_frame(train_dataloader)