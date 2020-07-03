import math
import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()


        self.conv1 = torch.nn.Sequential(
            #第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv3 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )


        self.conv4 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv5 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )



        self.conv6 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            #nn.ReLU(inplace=True),

        )

        # self.conv7 = torch.nn.Sequential(
        #     第一层in_channel=1,kenel_size=9,out_channel=64
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),

        # )
        self.conv11 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv22 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv33 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv44 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv55 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv66 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            #nn.ReLU(inplace=True),

        )

        self.conv111 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.conv222 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv333 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv444 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv555 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv666 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),

        )

        self.shortcut = nn.Sequential()


    def forward(self, x):

        out1 = self.conv1(x)
        out1 = torch.cat((self.conv2(out1), self.conv3(out1)), 1)
        out1 = torch.cat((self.conv4(out1), self.conv5(out1)), 1)
        out1 = self.conv6(out1)
        out1 = out1 + self.shortcut(x)
        out1 = F.relu(out1)

        input2 = torch.cat((out1, self.shortcut(x)), 1)

        out2 = self.conv11(input2)
        out2 = torch.cat((self.conv22(out2), self.conv33(out2)), 1)
        out2 = torch.cat((self.conv44(out2), self.conv55(out2)), 1)
        out2 = self.conv66(out2)
        out2 = out2 + self.shortcut(x)
        out2 = F.relu(out2)

        input3 = torch.cat((out1,out2, self.shortcut(x)), 1)

        out3 = self.conv111(input3)
        out3 = torch.cat((self.conv222(out3), self.conv333(out3)), 1)
        out3 = torch.cat((self.conv444(out3), self.conv555(out3)), 1)
        out3 = self.conv666(out3)
        out3 = out3 + self.shortcut(x)
        out = F.relu(out3)

        return out



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        print(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
