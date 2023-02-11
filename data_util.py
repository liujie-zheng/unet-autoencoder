import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def get_dataloader(batch_size):
    training_data = datasets.Caltech101(
        root="data",
        target_type='category',
        download=True,
        transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(256), transforms.ToTensor()])
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)

def print_sample_img(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()

def show_data(dataset, idx):
    img, label = dataset[idx]
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()
    plt.axis("off")
    print(f"Label: {label}")

print(show_data(train_dataloader.dataset, 2323))
print_sample_img(train_dataloader.dataset)