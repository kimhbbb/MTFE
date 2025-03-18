import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from UNet import UNet

class IFEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        expansion = 4
        
        self.stage1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.stage1_bn = nn.BatchNorm2d(6)
        self.stage1_af = nn.ReLU()

        self.stage2 = SFC_module(6, 12, expansion, True)
        self.stage3 = SFC_module(12, 24, expansion)
        self.stage4 = SFC_module(24, 48, expansion)
        self.stage5 = SFC_module(48, 96, expansion)
        self.stage6 = SFC_module(96, 192, expansion)
        self.stage7 = SFC_module(192, 384, expansion)
        self.stage8 = SFC_module(384, 768, expansion)
        
        self.stage9 = nn.AdaptiveAvgPool2d(1)

    def forward(self, img): # [16, 3, 256, 256]
        img = self.stage1(img) # [16, 6, 256, 256]
        img = self.stage1_bn(img) # [16, 6, 256, 256]
        img = self.stage1_af(img) # [16, 6, 256, 256]

        img = self.stage2(img) # [16, 12, 256, 256]
        img = self.stage3(img) # [16, 24, 128, 128]
        img = self.stage4(img) # [16, 48, 64, 64]
        img = self.stage5(img) # [16, 96, 32, 32]
        img = self.stage6(img) # [16, 192, 16, 16]
        img = self.stage7(img) # [16, 384, 8, 8]
        img = self.stage8(img) # [16, 768, 4, 4]
        img = self.stage9(img) # [16, 768, 1, 1]

        img = img.squeeze(2).squeeze(2) # [16, 768]
        img = img.unsqueeze(1).unsqueeze(3) # [16, 1, 768, 1]

        return img

class SFC_module(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stage1=False):
        super().__init__()

        expansion_channels = int(in_channels * expansion)

        if stage1 == True:
            self.se_conv =  nn.Conv2d(in_channels, expansion_channels, 3, 1, 1, groups=in_channels)
        else:
            self.se_conv = nn.Conv2d(in_channels, expansion_channels, 3, 2, 1, groups=in_channels)
        self.se_bn = nn.BatchNorm2d(expansion_channels)
        self.se_relu = nn.ReLU()

        self.hdf_conv = nn.Conv2d(expansion_channels, expansion_channels, 3, 1, 1, groups=in_channels)
        self.hdf_bn = nn.BatchNorm2d(expansion_channels)
        self.hdf_relu = nn.ReLU()

        self.comp_conv = nn.Conv2d(expansion_channels, out_channels, 1, 1, groups=in_channels)
        self.comp_bn = nn.BatchNorm2d(out_channels)

        self.pw_conv = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU()

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hdf_conv(x)
        x = self.hdf_bn(x)
        x = self.hdf_relu(x)
        x = self.comp_conv(x)
        x = self.comp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        
        return x

class HFFBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 2, 1)
        self.conv2 = nn.Conv2d(2, 1, 1)

        self.FC = nn.Linear(768, 768)
        self.BN = nn.BatchNorm1d(768)
        self.Sig = nn.Sigmoid()

    def forward(self, img, hist): # # [16, 1, 768, 1], # [16, 1, 768, 1]

        y = torch.cat([img, hist], dim=1) # [16, 2, 768, 1]
        y = self.conv1(y) # [16, 2, 768, 1]
        y = self.conv2(y) # [16, 1, 768, 1]
        y = y.squeeze(3)
        y = y.squeeze(1) # [16, 768]
        y = self.FC(y) # [16, 768]
        y = self.BN(y) # [16, 768]
        y = self.Sig(y) # [16, 768]

        img = img.squeeze(3).squeeze(1) # [16, 768]

        y = y * img + img # [16, 768]

        y = torch.relu(y) # [16, 768]

        return y       

class Hist_net(nn.Module):
    def __init__(self):
        super().__init__()
        extenstion = 4
        
        self.stage1 = SHFC_module(3, extenstion)
        self.stage2 = SHFC_module(3, extenstion)
        self.stage3 = SHFC_module(3, extenstion)
        self.stage4 = SHFC_module(3, extenstion)

    def forward(self, img): # [16, 3, 256, 256]
        hist_features = self.calc_hist(img) # [16, 3, 256]

        y = self.stage1(hist_features)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y) # [16, 3, 256]
        y = y.flatten(1) # [16, 768]

        y = y.unsqueeze(1).unsqueeze(3) # [16, 1, 768, 1]

        return y

    def calc_hist(self, img): # img : [16, 3, 256, 256]
        # img = img.squeeze(0) 
        hist_list = []

        for i in range(img.shape[0]): # 16번 반복
            channel_hist_list = []
            for c in range(img.shape[1]):
                channel_tensor = img[i, c, ...] # [256, 256]
                hist = torch.histc(channel_tensor, bins=256, min=0, max=255) # [256]
                channel_hist_list.append(hist)        
            hist_list.append(torch.stack(channel_hist_list)) # [3, 256]

        return torch.stack(hist_list) # [16, 3, 256]
    
class SHFC_module(nn.Module):
    def __init__(self, in_channels, expansion = 4):
        super().__init__()

        expansion_channels = int(in_channels * expansion)

        self.se_conv =  nn.Conv1d(in_channels, expansion_channels, 3, 1, 1, groups=in_channels) # 3개의 그룹으로 나눔.
        self.se_bn = nn.BatchNorm1d(expansion_channels)
        self.se_relu = nn.ReLU()

        self.hdf_conv = nn.Conv1d(expansion_channels, expansion_channels, 3, 1, 1, groups=in_channels)
        self.hdf_bn = nn.BatchNorm1d(expansion_channels)
        self.hdf_relu = nn.ReLU()

        self.comp_conv = nn.Conv1d(expansion_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.comp_bn = nn.BatchNorm1d(in_channels)

        self.pw_conv = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.pw_bn = nn.BatchNorm1d(in_channels)
        self.pw_relu = nn.ReLU()

    def forward(self, x): # [16, 3, 256] : (N, C, L)
        x = self.se_conv(x) # [16, 3, 256] -> # [16, 12, 256]
        x = self.se_bn(x) # [16, 12, 256]
        x = self.se_relu(x) # [16, 12, 256]
        x = self.hdf_conv(x) # [16, 12, 256] -> [16, 12, 256]
        x = self.hdf_bn(x) # [16, 12, 256]
        x = self.hdf_relu(x) # [16, 12, 256]
        x = self.comp_conv(x) # [16, 12, 256] -> # [16, 3, 256]
        x = self.comp_bn(x) # [16, 3, 256]
        x = self.pw_conv(x) # [16, 3, 256]
        x = self.pw_bn(x) # [16, 3, 256]
        x = self.pw_relu(x) # [16, 3, 256]

        return x # [16, 3, 256]

class TFGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.FC11 = nn.Linear(768, 768)
        self.FC12 = nn.Linear(768, 768)
        self.FC13 = nn.Linear(768, 768)
        self.FC21 = nn.Linear(768, 768)
        self.FC22 = nn.Linear(768, 768)
        self.FC23 = nn.Linear(768, 768)
        self.FC31 = nn.Linear(768, 768)
        self.FC32 = nn.Linear(768, 768)
        self.FC33 = nn.Linear(768, 768)

    def forward(self, y): # [16, 768]
        y1 = self.FC11(y)
        y1 = self.FC12(y1)
        y1 = self.FC13(y1) # [16, 768]

        y2 = self.FC21(y)
        y2 = self.FC22(y2)
        y2 = self.FC23(y2) # [16, 768]
        
        y3 = self.FC31(y)
        y3 = self.FC32(y3)
        y3 = self.FC33(y3) # [16, 768]
        
        y1 = y1.reshape(y1.shape[0], 3, 256) # [16, 3, 256]
        y2 = y2.reshape(y2.shape[0], 3, 256) # [16, 3, 256]
        y3 = y3.reshape(y3.shape[0], 3, 256) # [16, 3, 256]

        tf1 = torch.sigmoid(y1) 
        tf2 = torch.sigmoid(y2) 
        tf3 = torch.sigmoid(y3) 

        return tf1, tf2, tf3 # [16, 3, 256]

class IntensityTransformation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, tf1, tf2, tf3): # [16, 3, H, W], [16, 3, 256], [16, 3, 256], [16, 3, 256]
        img = torch.round(255 * img) # 일단 이미지 스케일링 해주어야 함.
        img = img.type(torch.LongTensor) # 정수 텐서로 변환
        img = img.cuda()

        tf1 = tf1.unsqueeze(3).cuda() # [16, 3, 256, 1]
        tf2 = tf2.unsqueeze(3).cuda() # [16, 3, 256, 1]
        tf3 = tf3.unsqueeze(3).cuda() # [16, 3, 256, 1]
        
        min_w = img.size(3) # 이미지의 width로, transform은 w가 더 커야함. 그래야 gather 가능.
        temp = 1
        iter_n = 0

        intensity_list = []

        while min_w > temp:
            temp *= 2 # 2배씩 키워도 gather로 나중에 잘림.
            iter_n += 1
        
        for _ in range(iter_n): # [16, 3, 256, X]
            tf1 = torch.cat([tf1, tf1], dim=3)
            tf2 = torch.cat([tf2, tf2], dim=3)
            tf3 = torch.cat([tf3, tf3], dim=3)
        
        img = torch.split(img, 1, dim=1) # [16, 1, H, W]
        tf1 = torch.split(tf1, 1, dim=1) # [16, 1, 256, X] * 3
        tf2 = torch.split(tf2, 1, dim=1) # [16, 1, 256, X] * 3
        tf3 = torch.split(tf3, 1, dim=1) # [16, 1, 256, X] * 3

        for transform in [tf1, tf2, tf3]:
            x = torch.gather(transform[0], dim=2, index=img[0])
            y = torch.gather(transform[1], dim=2, index=img[1])
            z = torch.gather(transform[2], dim=2, index=img[2])

            intensity = torch.cat([x, y, z], dim=1)
            intensity_list.append(intensity)

        return intensity_list # [16, 3, H, W] * 3

class HMTF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ife = IFEBlock()

        self.hist_net = Hist_net()
        self.hff = HFFBlock()

        self.tfg = TFGBlock()

        self.intensity_trans = IntensityTransformation()

    def forward(self, img):
        downsampled_img = F.interpolate(img, 256) # out : [16, 3, 256, 256]

        ife_img = self.ife(downsampled_img) # out : # [16, 1, 768, 1]
        hist = self.hist_net(downsampled_img) # out : # [16, 1, 768, 1]

        hff_img = self.hff(ife_img, hist) # out : [16, 768]
        tf1, tf2, tf3 = self.tfg(hff_img)
        intensity_list = self.intensity_trans(img, tf1, tf2, tf3)

        return intensity_list # xy1, xy2, xy3
    
class W_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_net = UNet(12, 3)

    def forward(self, img):
        # print("-----------------------------------")
        # print(f"img input shape : {img.shape}")
        # print("-----------------------------------")
        W = self.u_net(img) # input : [16, 12, H, W]

        return W

class IMG_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hmtf_net = HMTF_Net()
        self.w_net = W_Net()

    def forward(self, low):
        intensity_list = self.hmtf_net(low) # xy1, xy2, xy3

        intensity_tensor = torch.cat(intensity_list, dim=1)
        # print("-----------------------------------")
        # print(f"cat input shape : {torch.cat((low, intensity_tensor), dim=1).shape}")
        # print("-----------------------------------")
        w = self.w_net(torch.cat((low, intensity_tensor), dim=1)) # input : [16, 12, H, W]
        w = torch.sigmoid(w)
        w1, w2, w3 = torch.chunk(w, 3, dim=1) # [16, 1, H, W] * 3

        w1 = w1 / (w1 + w2 + w3)
        w2 = w2 / (w1 + w2 + w3)
        w3 = w3 / (w1 + w2 + w3)

        xy = w1 * intensity_list[0] + w2 * intensity_list[1] + w3 * intensity_list[2]

        return xy, (w1, w2, w3)