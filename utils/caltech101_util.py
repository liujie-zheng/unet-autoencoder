import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# return train_dataloader and test_dataloader
def get_dataloader(batch_size):
    dataset = datasets.Caltech101(
        root="data",
        target_type='category',
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            # raise grayscale images to 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)])
    )
    # split into train and test datas
    train_list = list(range(0, 6400))
    test_list = list(range(6400, 8677))
    train_data = torch.utils.data.Subset(dataset, train_list)
    test_data = torch.utils.data.Subset(dataset, test_list)
    return DataLoader(train_data, batch_size=batch_size), \
        DataLoader(test_data, batch_size=batch_size)

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

def show_img(dataset, idx):
    img, label = dataset[idx]
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()
    plt.axis("off")
    print(f"Label: {label}")

if __name__ == "__main__":
    get_dataloader(16)
    print('Done')