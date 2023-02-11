import torch
from torchvision import transforms
from PIL import Image, ImageOps

def load_image(path):
    img = Image.open(path)
    # resize to 256 * 256
    img = transforms.Resize(128)(img)
    img = transforms.RandomCrop(128)(img)
    # img = ImageOps.grayscale(img)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor

def export_image(img_tensor, path):
    img_tensor = torch.squeeze(img_tensor, 0)
    # print(img_tensor.shape)
    img = transforms.ToPILImage()(img_tensor)
    img.save(path)