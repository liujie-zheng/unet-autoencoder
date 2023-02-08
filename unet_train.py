import unet_model
import unet_model_legacy
import torch
import image_util
import torch.nn as nn

import matplotlib.pyplot as plt

def train(image, model):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Compute prediction error
    pred = model(image)
    loss = loss_fn(pred, image)
    print('loss =', loss)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    image = image_util.load_image('./data/test_pic.jpg')
    model = unet_model.UNet()

    epochs = 10000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(image, model)
    print("Done!")

    res = model(image)
    print(res.shape)
    image_util.export_image(res, './data/test_save.jpg')
