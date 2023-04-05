import random

import numpy as np

import unet_model
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt

import unet_model_smaller
from utils import caltech101_util
from utils import video_util


def train(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        # Compute prediction error
        pred = model(X)
        y = y.to(device)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Show progress
        if batch % 20 == 0:
            figure = plt.figure(figsize=(4, 2))
            rand_idx = random.randint(0, len(X) - 1)
            input_img = X[rand_idx]
            gt_img = y[rand_idx]
            output_img = model(X)[rand_idx]

            # imshow
            input_img = (input_img + 1) / 2
            output_img = (output_img + 1) / 2
            gt_img = (gt_img + 1) / 2
            figure.add_subplot(1, 2, 1)

            plt.imshow(input_img.cpu().detach().numpy().transpose(1, 2, 0))
            plt.axis("off")
            plt.title("train_input")
            figure.add_subplot(1, 2, 2)

            plt.imshow(np.clip(output_img.cpu().detach().numpy().transpose(1, 2, 0), 0, 1))
            plt.axis("off")
            plt.title("train_output")
            # calculate PSNR
            psnr = PeakSignalNoiseRatio()
            p = psnr(output_img.cpu(), gt_img.cpu())

            # show imgs and loss
            plt.show()
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  psnr: {p:>7f} [{current:>5d}/{size:>5d}]")

# avoiding the use of test as function name
def eval(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            y = y.to(device)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches

    # Show test results
    figure = plt.figure(figsize=(4, 2))
    rand_idx = random.randint(0, len(X) - 1)

    input_img = X[rand_idx]
    gt_img = y[rand_idx]
    output_img = pred[rand_idx]

    # calculate PSNR
    psnr = PeakSignalNoiseRatio()
    p = psnr(output_img.cpu(), gt_img.cpu())

    # avg PSNR
    p_total = 0
    for i in range(len(X)):
        input_img = X[i]
        gt_img = y[i]
        output_img = pred[i]
        p_total += psnr(output_img.cpu(), gt_img.cpu())
    p_avg = p_total / len(X)


    input_img = (input_img + 1) / 2
    output_img = (output_img + 1) / 2
    gt_img = (gt_img + 1) / 2

    figure.add_subplot(1, 2, 1)
    plt.imshow(input_img.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title("test_input")
    # output img
    figure.add_subplot(1, 2, 2)
    plt.imshow(np.clip(output_img.cpu().detach().numpy().transpose(1, 2, 0), 0, 1).astype(np.float32), cmap="gray")
    plt.axis("off")
    plt.title("test_output")
    # show imgs and loss
    plt.show()
    print(f"test size: {size}, test avg loss: {test_loss:>8f}, psnr: {p:>7f}, avg psnr: {p_avg:>7f} \n")

# train and test a video
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #permutations = [(x, y) for (i, x) in enumerate(range(4)) for (j, y) in enumerate(range(4)) if i != j]
    #for p in permutations:
    train_dataloader, test_dataloader = video_util.get_dataloader(32, 3, 4, False, True)
    model = unet_model_smaller.UNet().to(device)
    model.load_state_dict(torch.load("./weights/180_epoch10_adam_perturbed"))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    #print(f"training on {video_util.dataset_info[p[0]]}, testing on {video_util.dataset_info[p[1]]}")
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # train(train_dataloader, model, loss_fn, optimizer)
        eval(test_dataloader, model, loss_fn)
        break;


    print("Done!")
    # save weights
    # save_path = f"weights/180_epoch{t + 1}_adam_clean"
    # torch.save(model.state_dict(), save_path)
    # print("Weights saved at", save_path)



#  train and save caltech101
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#     train_dataloader, test_dataloader = caltech101_util.get_dataloader(64)
#     model = unet_model.UNet().to(device)
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     epochs = 20
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train(train_dataloader, model, loss_fn, optimizer)
#         eval(test_dataloader, model, loss_fn)
#     print("Done!")
#
#     # save weights
#     save_path = "weights/256_epoch20_adam"
#     torch.save(model.state_dict(), save_path)
#     print("Weights saved at", save_path)



# train a single image
# def train(image, model):
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#
#     # Compute prediction error
#     pred = model(image)
#     loss = loss_fn(pred, image)
#     # print('loss =', loss)
#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#     image = image_util.load_image('./data/test_pic_2.jpg')
#     model = unet_model.UNet()
#
#     epochs = 10000
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train(image, model)
#     print("Done!")
#
#     res = model(image)
#     # print(res.shape)
#     image_util.export_image(res, './data/test_save_2.jpg')
