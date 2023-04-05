import torch
from torch import nn

import unet_model_smaller
import unet_train
from utils import video_util


def train_clean(epochs=15, save_weights=True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_dataloader, test_dataloader = video_util.get_dataloader(32, 7, 6, False, False)
    model = unet_model_smaller.UNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        unet_train.train(train_dataloader, model, loss_fn, optimizer)
        # unet_train.eval(test_dataloader, model, loss_fn)
        # save weights
        if save_weights:
            save_path = f"weights/clean_trial3/epoch{t + 1}"
            torch.save(model.state_dict(), save_path)
            print("Weights saved at", save_path)
    print("Done!")

def train_perturbed(epochs=15, save_weights=True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_dataloader, test_dataloader = video_util.get_dataloader(32, 7, 6, True, False)
    model = unet_model_smaller.UNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        unet_train.train(train_dataloader, model, loss_fn, optimizer)
        # unet_train.eval(test_dataloader, model, loss_fn)
        # save weights
        if save_weights:
            save_path = f"weights/perturbed_trial3/epoch{t + 1}"
            torch.save(model.state_dict(), save_path)
            print("Weights saved at", save_path)
    print("Done!")

def test_clean():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_dataloader, test_dataloader = video_util.get_dataloader(32, 7, 6, False, True)
    model = unet_model_smaller.UNet().to(device)
    print(f"loading weights trained with clean dataset")
    model.load_state_dict(torch.load("./weights/clean_trial3/epoch15"))
    loss_fn = nn.MSELoss()
    print(f"testing on perturbed dataset")
    unet_train.eval(test_dataloader, model, loss_fn)
    print("Done!")


def test_perturbed():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_dataloader, test_dataloader = video_util.get_dataloader(32, 7, 6, False, True)
    model = unet_model_smaller.UNet().to(device)
    print(f"loading weights trained with perturbed dataset")
    model.load_state_dict(torch.load("./weights/perturbed_trial3/epoch15"))
    loss_fn = nn.MSELoss()
    print(f"testing on perturbed dataset")
    unet_train.eval(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    # train_clean()
    # train_perturbed()
    test_clean()
    test_perturbed()

    # trial 1
    # train dataset: 3 faces
    # test dataset: 5 cartoon
    # avg psnr: 24.038799 (clean)
    # avg psnr: 23.908739 (perturbed)

    # trial 2
    # train dataset: 1 rouen
    # test dataset: 3 faces
    # avg psnr: 30.528225 (clean)
    # avg psnr: 31.409842 (perturbed)

    # trial 3
    # train dataset: 7 sports
    # test dataset: 6 game
    # avg psnr: 26.391949 (clean)
    # avg psnr: 26.211687 (perturbed)