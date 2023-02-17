import torch
import torch.nn

def double_conv_3d(in_c, out_c):
    conv = nn.Sequencial(
        nn.conv3D(in_c, out_c, kernel_size = (3, 3, 1), stride = (1, 1, 1)),
        nn.ReLU(inplace = True),
        nn.conv3D(out_c, out_c, kernel_size = (3, 3, 1), stride = (1, 1, 1)),
        nn.ReLU(inplace = True)
    )
    return conv

def crop_img_seq(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool(kernel_size = (2, 2, 1), stride = (2, 2, 1))
        self.down_conv_1 = double_conv_3d(1*f, 64*f)
        self.down_conv_2 = double_conv_3d(64*f, 128*f)
        self.down_conv_3 = double_conv_3d(128*f, 256*f)
        self.down_conv_4 = double_conv_3d(256*f, 512*f)
        self.down_conv_5 = double_conv_3d(512*f, 1024*f)

        self.up_trans_1 = nn.ConvTranspose3d(in_channels = 1024*f, out_channels = 512*f, kernel_size = (2, 2, 1), stride = (2, 2, 1))
        self.up_conv_1 = double_conv_3d(1024*f, 512*f)
        self.up_trans_2 = nn.ConvTranspose3d(in_channels = 512*f, out_channels = 256*f, kernel_size = (2, 2, 1), stride = (2, 2, 1))
        self.up_conv_2 = double_conv_3d(512*f, 256*f)
        self.up_trans_3 = nn.ConvTranspose3d(in_channels = 256*f, out_channels = 128*f, kernel_size = (2, 2, 1), stride = (2, 2, 1))
        self.up_conv_3 = double_conv_3d(256*f, 128*f)
        self.up_trans_4 = nn.ConvTranspose3d(in_channels = 128*f, out_channels = 64*f, kernel_size = (2, 2, 1), stride = (2, 2, 1))
        self.up_conv_4 = double_conv_3d(128*f, 64*f)

        self.out = nn.conv2d(in_channels = 64*f, out_channels = 2*f, kernel_size = (1, 1, 1))
    
    def forward(self, image_seq):
        # encoder
        x1 = self.down_conv_1(image_seq) #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # decoder

        x = self.up_trans_1(x)
        y = crop_img_seq(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img_seq(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img_seq(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img_seq(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)
        return x


if __name__ == "__main__":
    f = 100
    image_seq = torch.rand((1, f, 572, 572))
    model = UNet()
    print(model(image_seq))