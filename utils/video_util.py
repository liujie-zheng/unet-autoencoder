import os

import torch
import torchvision.io
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch.nn.functional as F

import unet_model
import unet_model_smaller


class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        image = image.float() / 255.0 * 2 - 1
        if self.transform:
            image = self.transform(image)
        return image, image

class TrainDatasetPerturbed(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        image = image.float() / 255.0 * 2 - 1
        if self.transform:
            image = self.transform(image)
        image_p = random_perturb(image)
        return image_p, image

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        image = image.float() / 255.0 * 2 - 1
        if self.transform:
            image = self.transform(image)
        return image, image

class TestDatasetPerturbed(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "frame_{:05d}.jpg".format(idx))
        image = read_image(image_path)
        image = image.float() / 255.0 * 2 - 1
        if self.transform:
            image = self.transform(image)
        image_p = random_perturb(image)
        return image_p, image

def random_perturb(image, perturb_factor=0.004):
    channels, height, width = image.shape

    # Create a normalized grid (range: -1 to 1) with the same size as the input image
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
    grid = torch.stack((x, y), dim=-1).unsqueeze(0)

    # Create random perturbations
    random_perturbations = torch.randn_like(grid) * perturb_factor

    # Add the random perturbations to the grid
    perturbed_grid = grid + random_perturbations

    # Apply the perturbed grid to the image using grid_sample
    perturbed_image = F.grid_sample(image.unsqueeze(0), perturbed_grid, padding_mode='border').squeeze(0)

    return perturbed_image

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
                "data/frames_vfx",
                "data/frames_cartoon",
                "data/frames_game",
                "data/frames_sports"
                ]
def get_dataloader(batch_size, train_idx = 3, test_idx = 4, train_perturbed = False, test_perturbed = False):
    if train_perturbed:
        train_dataset = TrainDatasetPerturbed(
            root= dataset_info[train_idx],
            transform=transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(180)
            ])
        )
    else:
        train_dataset = TrainDataset(
            root= dataset_info[train_idx],
            transform=transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(180)
            ])
        )

    if test_perturbed:
        test_dataset = TestDatasetPerturbed(
            root= dataset_info[test_idx],
            transform=transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(180)
            ])
        )
    else:
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
    model = unet_model_smaller.UNet().to(device)
    model.load_state_dict(torch.load("../weights/180_epoch10_adam_clean"))
    model.eval()
    # output = torch.empty([0, 3, 720, 1280], dtype=torch.uint8)
    for X, y in dataloader:
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
    #
    # test_dataset = TestDataset(
    #     root= "../data/frames_bear",
    #     transform=transforms.Compose([
    #         transforms.ConvertImageDtype(torch.float32),
    #         transforms.Resize(90)
    #     ])
    # )
    # test_dataloader = DataLoader(test_dataset, batch_size=8)
    # show_sample_frame(test_dataloader)
    train_dataloader, test_dataloader = get_dataloader(32, 3, 4, False, True)
    show_test_frames(test_dataloader)



