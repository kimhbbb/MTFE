import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, nin, nout, nmid = None): # nin: 12, nout: 16
        super().__init__()
        if not nmid:
            nmid = nout
        self.double_conv = nn.Sequential(
            nn.Conv2d(nin, nmid, kernel_size=3, padding=1),
            nn.BatchNorm2d(nmid),
            nn.ReLU(inplace=True),
            nn.Conv2d(nmid, nout, kernel_size=3, padding=1),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): # input : [16, 12, H, W]
        # print("-----------------------------------")
        # print(f"x input shape : {x.shape}")
        # print("-----------------------------------")
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(nin, nout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, nin, nout, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(nin, nout, nin // 2)
        else:
            self.up = nn.ConvTranspose2d(nin , nin // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(nin, nout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, nin, nout, bilinear=True):
        super().__init__()

        self.in_conv = DoubleConv(nin, 16) # 12, 16
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.out_conv = OutConv(16, nout)

    def forward(self, x): 
        x1 = self.in_conv(x) # input : [16, 12, H, W]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits