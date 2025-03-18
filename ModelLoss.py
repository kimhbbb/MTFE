import torch
import torch.nn as nn




class entropy_loss(nn.Module):
    def __init___(self):
        super().__init__()

    def forward(self):
        
        


        pass


class total_variation_loss(nn.Module):
    def __init__(self, l_tv = 1e-4):
        super().__init__()
        self.l_tv = l_tv

    def forward(self, w):
        print("-----------------------------------")
        print(f"w shape: {w.shape}")
        print("-----------------------------------")

        w1, w2, w3 = w
        w = torch.cat((w1,w2,w3),dim=1)

        loss_tv = 0
        batch_size = w.size()[0]
        h_x = w.size()[2]
        w_x = w.size()[3]
        count_h = (w.size()[2] - 1) * w.size()[3]  # 전체 픽셀 개수 중 세로 방향
        ount_w = w.size()[2] * (w.size()[3] - 1)  # 전체 픽셀 개수 중 가로 방향
        
        #################################################################################
        # 여기부터 다시 해야함.                                                           #
        #################################################################################
        loss_tv = self.l_tv * h_tv.s



        return loss_tv