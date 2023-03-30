import os

import torch
import torchvision.io
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms

import unet_model

class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image, 0

class TestDataset(Dataset):
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
    plt.imshow(img.detach().cpu().permute(1, 2, 0), cmap="gray")
    plt.axis(False)
    plt.show()

dataset_info = ["data/frames_bear",
                "data/frames_rouen",
                "data/frames_car",
                "data/frames_faces",
                "frames_vfx",
                "frames_cartoon",
                "frames_game",
                "frames_sports"
                ]
def get_dataloader(batch_size, train_idx, test_idx):
    train_dataset = TrainDataset(
        root= dataset_info[train_idx],
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(180)
        ])
    )
    test_dataset = TestDataset(
        root= dataset_info[test_idx],
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(180)
        ])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader

def show_test_frames(dataloader):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = unet_model.UNet().to(device)
    model.load_state_dict(torch.load("../weights/video_epoch3_adam"))
    model.eval()
    # output = torch.empty([0, 3, 720, 1280], dtype=torch.uint8)
    for X, y in test_dataloader:
        X = X.to(device)
        pred = model(X)
        pred = transforms.ConvertImageDtype(torch.uint8)(pred)
        # output = torch.cat((output, pred.cpu()), 0)
        for i in range(0, len(X)):
            plt.imshow(pred[i].cpu().detach().numpy().transpose(1, 2, 0), cmap="gray")
            plt.axis(False)
            plt.show()
    # output = output.permute(0,2,3,1)
    # print(output.shape)
    # torchvision.io.write_video("../output/bear_output.mp4", output, 25)




if __name__ == "__main__":
    # train_dataset = RouenVideo(
    #     root= "../data/frames_bear",
    #     transform=transforms.Compose([
    #         transforms.ConvertImageDtype(torch.float32),
    #     ])
    # )
    # train_dataloader = DataLoader(train_dataset, 64)
    # show_sample_frame(train_dataloader)

    test_dataset = TestDataset(
        root= "../data/frames_bear",
        transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(90)
        ])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    show_sample_frame(test_dataloader)