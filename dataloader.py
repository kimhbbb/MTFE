import torch
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import numpy as np
import random


class input_loader(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        print(f"Here : {os.path.join(image_path, 'input')}")
        self.in_files = self.list_files(os.path.join(image_path, 'input'))

    def data_augment(self, inp, gt):
        a = random.randint(1, 4)
        if a == 1:
            return inp, gt
        elif a == 2:
            return inp.rotate(180, expand=True), gt.rotate(180, expand=True)
        elif a == 3:
            return ImageOps.flip(inp.rotate(180, expand=True)), ImageOps.flip(gt.rotate(180, expand=True))
        else:
            return ImageOps.flip(inp), ImageOps.flip(gt)

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files

    def __getitem__(self, index): # index는 DataLoader에서 내부적으로 생성해줌. 기본적으로 range(len(dataset))의 인덱스를 사용하여, 각 배치마다 batch_size 개수만큼의 index를 가져옴
        fname = os.path.split(self.in_files[index])[-1] # (index에 해당하는 이미지의 경로)에서 [-1]을 이용하여 파일명만 가져옴 # os.path.split(path)는 파일 경로를 디렉토리 부분과 파일 이름 부분으로 분리함.
        data_low = Image.open(self.in_files[index]) # 저해상도 이미지 경로의 해당 이미지를 불러오는 코드.
        data_gt = Image.open(os.path.join(self.image_path, 'gt', fname)) # gt 폴더에 따로 담긴 동일한 이름의 정답(고해상도) 이미지를 불러옴.

        low = np.asarray(data_low) # Numpy 배열로 변환 (H, W, 3)


        #########################################################################################
        # 히스토그램이 원래 이미지에서 뽑히는 거 아니지 않나?                                       #
        # 다운 샘플에서 뽑는 거 아님?                                                             #
        # data_hist = np.zeros((3, 256))  # 3 채널 x 256 개의 bin (히스토그램 저장할 공간)
        # for i in range(3): 
        #     S = low[..., i] # R, G, B 순으로 특정 채널의 픽셀 데이터 가져오기. # 모든 행 H, 열 W, 마지막 축에서 i번째 요소만 선택.
        #     data_hist[i, ...], _ = np.histogram(S.flatten(), 256, [0, 255]) # data, bin, range # hist((256,)), bins를 반환함.
        #     data_hist[i, ...] = data_hist[i, ...] / np.sum(data_hist[i, ...]) # 각 채널의 히스토그램 데이터 정규화
        #                                                                                       #
        #########################################################################################
        
        data_input, data_gt = self.data_augment(data_low, data_gt)

        data_input = (np.asarray(data_input)/255.0)
        data_gt = (np.array(data_gt)/255.0)

        data_input = torch.from_numpy(data_input).float()
        data_gt = torch.from_numpy(data_gt).float()
        # data_hist = torch.from_numpy(data_hist).float()


        return data_input.permute(2,0,1), data_gt.permute(2,0,1)

    def __len__(self):
        return len(self.in_files)