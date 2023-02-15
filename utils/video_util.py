import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import transforms


# path = "../data/rouen.mp4"
# vframe, _, _ = torchvision.io.read_video(path)
# print(vframe.shape)

class RouenVideo(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 2000

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
        return 1000

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
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.axis(False)
    plt.show()

def get_dataloader(batch_size):
    train_dataset = RouenVideo(
        root = "data/frames_rouen",
        transform = transforms.Compose([
            # transforms.Resize(360),
            transforms.ConvertImageDtype(torch.float32)])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = BearVideo(
        root = "data/frames_bear",
        transform = transforms.Compose([
            # transforms.Resize(360),
            transforms.ConvertImageDtype(torch.float32)])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader