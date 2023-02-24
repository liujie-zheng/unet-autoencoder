import torch
import torch.nn as nn

from utils import image_util
import unet_model


def conv_and_relu(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True)
    )
    return conv



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = conv_and_relu(3, 16)
        self.conv_2 = conv_and_relu(16, 32)
        self.conv_3 = conv_and_relu(32, 64)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_4 = conv_and_relu(64, 32)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv_5 = conv_and_relu(32, 16)

        self.out = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)

    def forward(self, image):
        # encoder
        x1 = self.conv_1(image)  #
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_2(x2)  #
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_3(x4)

        # decoder
        x = self.up_trans_1(x5)
        x = self.conv_4(torch.cat([x, x3], 1))
        x = self.up_trans_2(x)
        x = self.conv_5(torch.cat([x, x1], 1))

        x = self.out(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    image = torch.rand((1, 3, 32, 32))
    model = UNet()
    res = model.forward(image)
    print(res.shape)
