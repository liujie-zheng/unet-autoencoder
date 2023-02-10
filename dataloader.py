import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt

training_data = datasets.Caltech101(
    root="data",
    target_type={'elephant', 'elephant'},
    download=True,
    transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(256), transforms.ToTensor()])
)