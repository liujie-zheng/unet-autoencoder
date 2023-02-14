import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image


# path = "../data/rouen.mp4"
# vframe, _, _ = torchvision.io.read_video(path)
# print(vframe.shape)

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

def show_sample_frame(dataloader):
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.axis(False)
    plt.show()

if __name__ == "__main__":
    dataset = RouenVideo("../data/frames_rouen")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    show_sample_frame(dataloader)