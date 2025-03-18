import numpy as np
import argparse
import torch
from Model import IMG_Net
from Model import Hist_net
import dataloader
from torch.utils.data import DataLoader
import ModelLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import sys
import os
import shutil
import time
import glob
import torchvision
from matplotlib import pyplot as plt
import math
import fnmatch
import cv2


writer = SummaryWriter()
GPU_NUM = 0

def read_img(file_name):
    return cv2.imread(file_name)

def findFile(filePath, fileName):
    ans = None
    for f_name in os.listdir(filePath):
        fileName = os.path.splitext(fileName)[0]
        if fnmatch.fnmatch(f_name, fileName + ".*"):
            ans = f_name
    if ans == None:
        print("There is no Ground-Truth matched with input file")

    return ans

def calc_PSNR(image_path, gt_path):
    gt_file = gt_path + str(findFile(gt_path, os.path.basename(image_path)))
    s = read_img(image_path) / 255.0
    r = read_img(gt_file) / 255.0
    mse = np.mean(np.square(s - r))
    psnr = 10 * math.log10(1 / mse)

    return psnr

def eval(model, model_path, save_plot = False): 
    originDirPath = 'data/test_data'

    dir_list = os.listdir(originDirPath)
    sum_psnr = 0
    n_of_files = 0

    for dir_name in dir_list:
        file_list = glob.glob(originDirPath + dir_name + "/*")
        n_of_files = len(file_list)
        for image in file_list:
            img_lowlight = Image.open(image)
            img_lowlight = np.asarray(img_lowlight) / 255.0
            img_lowlight = torch.from_numpy(img_lowlight).float()
            img_lowlight = img_lowlight.permute(2, 0, 1)
            img_lowlight = img_lowlight.cuda().unsqueeze(0)

            hist = Hist_net.calc_hist(image).cuda().unsqueeze(0)

            with torch.no_grad(): # 그래디언트 연산 비활성화
                enhanced_img, vec, wm, xy = model(img_lowlight, hist)

            result_path = image.replace('test_data', 'analysis/result')
            plot_path = image.replace('test_data', 'analysis/test_plots')
            wm_path1 = image.replace('test_data/LOL', 'analysis/test_weightmap1')
            wm_path2 = image.replace('test_data/LOL', 'analysis/test_weightmap2')
            wm_path3 = image.replace('test_data/LOL', 'analysis/test_weightmap3')
            xy_path1 = image.replace('test_data/LOL', 'analysis/test_output1')
            xy_path2 = image.replace('test_data/LOL', 'analysis/test_output2')
            xy_path3 = image.replace('test_data/LOL', 'analysis/test_output3')

            if not os.path.exists(result_path.replace('/' + result_path.split("/")[-1], '')):
                os.makedirs(result_path.replace('/' + result_path.split("/")[-1], ''))
            if not os.path.exists(plot_path.replace('/' + plot_path.split("/")[-1], '')):
                os.makedirs(plot_path.replace('/' + plot_path.split("/")[-1], ''))
            if not os.path.exists(wm_path1.replace('/' + wm_path1.split("/")[-1], '')):
                os.makedirs(wm_path1.replace('/' + wm_path1.split("/")[-1], ''))
            if not os.path.exists(wm_path2.replace('/' + wm_path2.split("/")[-1], '')):
                os.makedirs(wm_path2.replace('/' + wm_path2.split("/")[-1], ''))
            if not os.path.exists(wm_path3.replace('/' + wm_path3.split("/")[-1], '')):
                os.makedirs(wm_path3.replace('/' + wm_path3.split("/")[-1], ''))
            if not os.path.exists(xy_path1.replace('/' + xy_path1.split("/")[-1], '')):
                os.makedirs(xy_path1.replace('/' + xy_path1.split("/")[-1], ''))
            if not os.path.exists(xy_path2.replace('/' + xy_path2.split("/")[-1], '')):
                os.makedirs(xy_path2.replace('/' + xy_path2.split("/")[-1], ''))
            if not os.path.exists(xy_path3.replace('/' + xy_path3.split("/")[-1], '')):
                os.makedirs(xy_path3.replace('/' + xy_path3.split("/")[-1], ''))

            torchvision.utils.save_image(enhanced_img, result_path)
            torchvision.utils.save_image(wm[0], wm_path1)
            torchvision.utils.save_image(wm[1], wm_path2)
            torchvision.utils.save_image(wm[2], wm_path3)
            torchvision.utils.save_image(xy[0], xy_path1)
            torchvision.utils.save_image(xy[1], xy_path2)
            torchvision.utils.save_image(xy[2], xy_path3)

            if save_plot == True:
                if not os.path.exists(plot_path.replace('/' + plot_path.split("/")[-1], '')):
                    os.makedirs(plot_path.replace('/' + plot_path.split("/")[-1], ''))

                vec1 = vec[0]
                vec2 = vec[1]
                vec3 = vec[2]

                (fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
                vec1 = vec1.squeeze(0)
                # vec1 = vec1 * 0.5 + 0.5
                vec1 = vec1.cpu().detach().numpy()
                vec1 = vec1 * 255
                r1 = vec1[0, ...]
                g1 = vec1[1, ...]
                b1 = vec1[2, ...]
                axs[0][0].plot(r1, color='r')
                axs[0][1].plot(g1, color='g')
                axs[0][2].plot(b1, color='b')

                vec2 = vec2.squeeze(0)
                # vec2 = vec2 * 0.5 + 0.5
                vec2 = vec2.cpu().detach().numpy()
                vec2 = vec2 * 255
                r2 = vec2[0, ...]
                g2 = vec2[1, ...]
                b2 = vec2[2, ...]
                axs[1][0].plot(r2, color='r')
                axs[1][1].plot(g2, color='g')
                axs[1][2].plot(b2, color='b')

                vec3 = vec3.squeeze(0)
                # vec3 = vec3 * 0.5 + 0.5
                vec3 = vec3.cpu().detach().numpy()
                vec3 = vec3 * 255
                r3 = vec3[0, ...]
                g3 = vec3[1, ...]
                b3 = vec3[2, ...]
                axs[2][0].plot(r3, color='r')
                axs[2][1].plot(g3, color='g')
                axs[2][2].plot(b3, color='b')

                axs[0][0].set_xlim([0, 255])
                axs[0][0].set_ylim([0, 255])
                axs[0][1].set_xlim([0, 255])
                axs[0][1].set_ylim([0, 255])
                axs[0][2].set_xlim([0, 255])
                axs[0][2].set_ylim([0, 255])
                axs[1][0].set_xlim([0, 255])
                axs[1][0].set_ylim([0, 255])
                axs[1][1].set_xlim([0, 255])
                axs[1][1].set_ylim([0, 255])
                axs[1][2].set_xlim([0, 255])
                axs[1][2].set_ylim([0, 255])
                axs[2][0].set_xlim([0, 255])
                axs[2][0].set_ylim([0, 255])
                axs[2][1].set_xlim([0, 255])
                axs[2][1].set_ylim([0, 255])
                axs[2][2].set_xlim([0, 255])
                axs[2][2].set_ylim([0, 255])

                plt.tight_layout()
                plt.draw()
                plt.savefig(plot_path)

        sum_psnr += calc_PSNR(result_path, './data/test_gt/')
    avg_psnr = sum_psnr / n_of_files
    print('Avg_PSNR: %.3f\t' % (avg_psnr))

    return avg_psnr

def weights_init(m):
    classname = m.__class__.__name__ # 계층의 클래스 이름.
    if classname.find('Conv2d') != -1: # Conv2d 계층 초기화. # != -1은 특정 문자열에서 부분 문자열이 포함되어 있는지 확인하기 위한 조건임. 'Conv2d'라는 부분 문자열이 존재하면 그 위치의 인덱스를 반환함. 없으면 -1 반환.
        m.weight.data.normal_(0.0, 0.02) # 정규분포 평균 0.0, 표준편차 0.02로 초기화.
    elif classname.find('BatchNorm') != -1: # BatchNorm 계층 초기화.
        m.weight.data.normal_(1.0, 0.02) # 정규분포 평균 1.0, 표준편차 0.02로 초기화.
        m.bias.data.fill_(0) # 편향(bias)를 0으로 초기화.

def train(config):
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU

    ImgNet = IMG_Net()
    # ImgNet 모델의 모든 서브 모듈에 대해 weights_init 함수를 적용.
    # 모든 서브 모듈(Conv2d, BatchNorm,..)을 순회하며, 각 모듈에 weights_init 함수 호출함.
    # 모든 계층에 대해 동일한 초기화 전략을 적용하여, 훈련 결과의 일관성을 유지.
    # 문제에 적합한 초기화 방식을 직접 정의하여 적용 가능함.
    ImgNet.apply(weights_init)
    ImgNet = ImgNet.cuda() # .cuda()를 호출해야 모델의 모든 파라미터가 GPU로 이동함. # 연산이 이루어지려면 모든 요소가 동일한 장치(CPU or GPU)에 있어야 함. 그래서 모델 뿐만 아니라 입력 데이터와 레이블 데이터도 GPU로 이동해야 함. ex) inputs = inputs.cuda()


    train_dataset = dataloader.input_loader(config.train_images_path)
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size=config.train_batch_size, # 한 번의 미니배치에서 로드할 샘플 개수
                              shuffle=True, # 매 epoch마다 데이터셋의 순서를 무작위로 섞음. 모델이 데이터 순서에 의존하지 않고 일반화할 수 있도록 도와줌.
                              num_workers=config.num_workers, # 데이터를 로드할 때 사용할 병렬 프로세스(worker)의 개수를 지정함. 
                              pin_memory=True) # GPU 가속을 사용할 때, True로 설정하면 고속의 고정 메모리에 데이터를 저장하여 CPU->GPU 전송 속도를 높일 수 있음.
                                               # 일반적으로 CUDA를 사용할 경우 pin_memory=True로 설정하는 것이 성능 향상에 도움됨.
                                               # CPU에서 로드한 데이터를 바로 GPU로 전송할 때 비동기(Async) 전송이 가능해짐.
                                               # GPU를 사용하지 않는 경우에는 False로 설정하는 것이 메모리 관리 측면에서 좋음.
    sum_time = 0
    highest_psnr = 0
    highest_psnr_s = 0
    psnr_ep = 0
    
    loss_c = torch.nn.MSELoss().cuda()
    cos = torch.nn.CosineSimilarity(dim=1)
    loss_e = ModelLoss.entropy_loss().cuda()
    loss_total = ModelLoss.total_variation_loss().cuda()

    # .parameters()는 nn.Module을 상속받은 클래스에서 사용 가능함. nn.Linear, nn.Conv2d, nn.BatchNorm2d 등 학습 가능한 모든 weight와 bias를 반환함.
    # weight_decay : L2 regularization 계수로, overfitting 방지하기 위해 사용됨.
    optimizer_img = torch.optim.Adam(ImgNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 모델이 training 모드로 설정됨. 학습을 위한 설정을 활성화하며, 모델 내부의 특정 레이어들이 훈련 모드에 맞게 동작하도록 함. 
    # 예를 들어 nn.BatchNorm 레이어가 입력 데이터의 평균과 분산을 현재 배치에서 계산을 함.
    # 훈련이 끝난 후 model.eval()을 호출하면 고정된 평균과 분산을 사용하도록 전환해야 함. -> 훈련 중 저장된 평균 & 분산 사용.
    # nn.Dropout 레이어가 활성화됨. 학습 과정에서 일부 뉴련을 랜덤하게 비활성화하여 과적합을 방지하는 기법.
    # model.train() 이후로는 forward pass, loss 계산, backward pass(.backward()), optimizer step, optimizer.zero_grad() 호출하여 이전 미니배치에서 계산된 그래디언트를 초기화 하는 과정으로 training이 진행될 거임.
    # .backward()를 호출하면, 각 파리미터에 대한 그래디언트가 누적이 됨. 즉, 이전 배치에서 계산된 그래디언트가 새로운 배치의 그래디언트에 더해지므로 학습이 올바르게 진행되지 않을 수 있음. 따라서 .zero_grad()를 사용하여 매 미니배치마다 그래디언트를 초기화해야 함.
    # model.eval()로 전환하지 않으면, 테스트/추론 시에도 계속 배치 평균 업데이트가 진행되어 잘못된 예측할 수도 있음.
    ImgNet.train() 

    num_params = 0
    for param in ImgNet.parameters():
        num_params += param.numel() # .numel() : PyTorch 텐서에서 요소의 총 개수를 반환하는 함수임. 현재 텐서(param)의 총 원소의 개수 반환.
    # print('# of Imgnet params : %d' % num_params)

    ###################################################
    #               탐구가 필요한 녀석들                #
    # 논문에서 제시한 priority factor                    
    cont_c = 0.5                                       
    cont_cs = 0.3
    cont_e = 0.2

    lambda_c = 0
    lambda_cs = 0
    lambda_e = 0

    difficulty_c = 0
    difficulty_cs = 0
    difficulty_e = 0

    loss_col_0 = 0
    loss_cos_0 = 0
    loss_ent_0 = 0
    loss_0 = 0

    loss_col_2 = 0
    loss_cos_2 = 0
    loss_ent_2 = 0
    loss_2 = 0
    #                                                 #
    ###################################################

    # epoch : dataset 전체를 한 번 학습하는 단위. 한 epoch 안에서 모든 미니배치 데이터를 학습하고, 다음 epoch에서 다시 업데이트된 모델로 전체 데이터를 다시 학습함.
    for epoch in range(config.num_epochs):
        st = time.time()
        print("epoch :", epoch + 1)

        # 에포크 단위의 평균 loss 계산을 위해서 각 미니배치의 loss를 합해줌.
        sumLossCol = 0
        sumLossEnt = 0
        sumLossCos = 0
        sumLossTV = 0

        sumLoss = 0
        sumLoss_ = 0

        for iteration, (low, gt) in enumerate(train_loader):
            low = low.cuda() # 위에서 설명했듯, 연산은 한 장치에서 이루어져야 하기 때문에 다 옮겨줌.
            gt = gt.cuda()
            # hist = hist.cuda()

            img, w = ImgNet(low)
            img = img.cuda()

            loss_img = loss_c(img, gt) # 예측 img와 정답 gt와의 MSELoss 계산.
            loss_ent = loss_e(w) # weight map의 entropy loss 계산.
            loss_col = torch.mean(1 - torch.abs(cos(gt, img))) # gt와 img 간의 cos으로 loss 계산.
            loss_tv = loss_total(w) # total variation loss 계산.

            if epoch == 0:
                # loss_f = cont_c * loss_img + cont_e * loss_ent + cont_cs * loss_col
                loss_f = 0.5 * loss_img + 0.3 * loss_col + 0.2 * loss_ent + 0.0001 * loss_tv
            elif epoch == 1:
                loss_f = lambda_c * loss_img + lambda_e * loss_ent + lambda_cs * loss_col + 0.0001 * loss_tv
            else:
                loss_f = (lambda_c * difficulty_c * loss_img 
                          + lambda_cs * difficulty_cs * loss_col 
                          + lambda_e * difficulty_e * loss_ent + 0.0001 * loss_tv)
                
            #########################
            #                       #
            #   이놈의 역할은 뭘까   #
            #                       #
            #########################
            loss_ = loss_f - loss_tv



            # 이전 iteration에서 계산된 그래디언트를 초기화하는 역할을 함.
            # PyTroch에서 loss.backward()를 호출하면, 현재 미니배치에 대한 loss의 그래디언트가 누적되는데, 이를 방지하기 위해 iteration마다 zero_grad()를 호출
            # 옵티마이저가 관리하는 모든 모델의 파라미터들의 .grad 속성을 0으로 초기화하는 역할을 함.
            optimizer_img.zero_grad()  
            # 미니배치의 loss를 자동미분(Autograd) 통해 loss_f의 값을 기준으로 모델의 모든 파라미터에 대한 기울기를 자동으로 계산함.
            # backward()하면 모델의 각 파라미터의 .grad 속성에 기울기가 누적됨.
            # ex) 
            # x = torch.tensor([3.0], requires_grad=True)
            # y = x ** 2
            # z = x ** 3
            # y.backward() # tensor([6.])
            # z.backward() # tensor([27.])
            # print(x.grad)  # tensor([33.])
            loss_f.backward() 
            # 가중치 업데이트 전에 기울기 크기를 일정 범위로 제한하는 역할을 함. 
            # clip_grad_norm_() : 전체 가중치의 L2 Norm 구한 후, config.grad_clip_norm 값으로 정규화하여 기울기 크기 제한.
            # !!! 개별적으로 특정 파리미터가 config.grad_clip_norm 값을 넘었을 때 clip이 아니라, 
            # 모든 파라미터의 기울기의 L2 norm을 계산하여 값이 크면, 모든 기울기를 동일한 비율로 축소하는 방식임.
            torch.nn.utils.clip_grad_norm_(ImgNet.parameters(), config.grad_clip_norm)
            # .step() 호출하면 미리 계산된 기울기를 이용해 모델의 가중치를 업데이트 함.
            optimizer_img.step()

            sumLossCol += loss_img.item()
            sumLossEnt += loss_ent.item()
            sumLossCos += loss_col.item()
            sumLossTV += loss_tv.item()

            sumLoss += loss_f.item()
            sumLoss_ += loss_.item()

            if iteration == len(train_loader) - 1: # 마지막 미니배치 (한 epoch의 끝)
                print(f"Fus Loss: {loss_f.item()}")
                # HmtfNet.state_dict()를 .pth 파일로 저장하여 모델 가중치를 체크포인트로 저장함.
                # snapshots_folder에 "Img_tmp.pth" 이름으로 저장됨. 학습 중단 시, 이 파일을 불러와서 훈련을 이어갈 수 있음.
                # 훈련 과정 중 모델이 너무 커서 한 번에 학습을 끝낼 수 없는 경우에 중간 저장 후 다음 실행에서 이어서 학습 가능하도록 함.
                # 훈련 중 특정 에포크에서 가장 성능이 좋은 모델을 저장하여, 최적 모델을 보관할 수 있게 함.
                # ex)
                # 현재 에포크 손실이 기존 최저 손실보다 작다면, 모델 저장
                # if epoch_loss < best_loss:
                #     best_loss = epoch_loss
                #     torch.save(HmtfNet.state_dict(), config.snapshots_folder + "best_model.pth")
                #     print(f"Saved new best model at epoch {epoch + 1}, loss: {best_loss:.4f}")
                torch.save(ImgNet.state_dict(), config.snapshots_folder + "Img_tmp.pth")

                # 에포크 단위 평균 손실 계산.
                loss_col_0 = sumLossCol / len(train_loader)
                loss_ent_0 = sumLossEnt / len(train_loader)
                loss_cos_0 = sumLossCos / len(train_loader)
                loss_0 = sumLoss_ / len(train_loader)

                # SummaryWriter 객체를 사용하여 TensorBoard에 손실 값을 기록함.
                # add_scalar()를 사용하여 epoch 별 손실 곡선을 시각화할 수 있음. 
                writer.add_scalar('color_loss', sumLossCol / len(train_loader), epoch + 1)
                writer.add_scalar('transformationFunction_loss', sumLossEnt / len(train_loader), epoch + 1)
                writer.add_scalar('cosineSimilarity_loss', sumLossCos / len(train_loader), epoch + 1)
                writer.add_scalar('totalVariation_loss', sumLossTV / len(train_loader), epoch + 1)
                writer.add_scalar('total_loss', sumLoss / len(train_loader), epoch + 1)
                
        ################################################################
        #           그냥 복붙함                                         #
        if epoch == 0:
            loss_col_1 = loss_col_0
            loss_ent_1 = loss_ent_0
            loss_cos_1 = loss_cos_0
            loss_1 = loss_0
            # get loss weights
            lambda_c = cont_c * (loss_1 / loss_col_1)
            lambda_e = cont_e * (loss_1 / loss_ent_1)
            lambda_cs = cont_cs * (loss_1 / loss_cos_1)

            print()
            print('lambda_c\t' + str(lambda_c))
            print('lambda_e\t' + str(lambda_e))
            print('lambda_cs\t' + str(lambda_cs))

            print()
            # update previous losses
            loss_col_2 = loss_col_1
            loss_ent_2 = loss_ent_1
            loss_cos_2 = loss_cos_1

            loss_2 = loss_1

        else:
            loss_col_1 = loss_col_0
            loss_ent_1 = loss_ent_0
            loss_cos_1 = loss_cos_0
            loss_1 = loss_0
            # get loss weights
            lambda_c = cont_c * (loss_1 / loss_col_1)
            lambda_e = cont_e * (loss_1 / loss_ent_1)
            lambda_cs = cont_cs * (loss_1 / loss_cos_1)

            print()
            print('lambda_c\t' + str(lambda_c))
            print('lambda_e\t' + str(lambda_e))
            print('lambda_cs\t' + str(lambda_cs))
            print()
            # get difficulties
            difficulty_c = ((loss_col_1 / loss_col_2) / (loss_1 / loss_2)) ** config.beta
            difficulty_e = ((loss_ent_1 / loss_ent_2) / (loss_1 / loss_2)) ** config.beta
            difficulty_cs = ((loss_cos_1 / loss_cos_2) / (loss_1 / loss_2)) ** config.beta
            print('difficulty_c\t' + str(difficulty_c))
            print('difficulty_e\t' + str(difficulty_e))
            print('difficulty_cs\t' + str(difficulty_cs))
            print()
            # update previous losses
            loss_col_2 = loss_col_1
            loss_ent_2 = loss_ent_1
            loss_cos_2 = loss_cos_1
            loss_2 = loss_1
        #                                                              #
        ################################################################
    #############################################################################################
    #                           그냥 복붙함                                                      #
        psnr = eval(ImgNet, config.snapshots_folder + "Img_tmp.pth", save_plot=False) # 현재 epoch에서 학습된 모델의 PSNR을 계산함.
        if highest_psnr < psnr: # 기존 hightes_psnr보다 크면 업데이트
            highest_psnr = psnr
            psnr_ep = epoch + 1
            if not os.path.isdir("./data/best_score/best_psnr"):
                os.mkdir("./data/best_score/best_psnr")
            # copy_tree("./data/train_check/test", "./data/best_score/best_psnr") # 테스트 데이터 저장함.
            shutil.copytree("./data/train_check/test", "./data/best_score/best_psnr") # 테스트 데이터 저장함.
            shutil.copy("./models/Img_tmp.pth", "./models/Img_final.pth") # 최고 성능 모델 저장함.

        writer.add_scalar('PSNR', psnr, epoch + 1)
        et = time.time() - st
        print('%d epoch: %.3f' % (epoch + 1, et))
        sum_time += et
        rTime = (sum_time / (epoch + 1)) * (config.num_epochs - (epoch + 1))
        print("Estimated time remaining :%d hour %d min %d sec" % (
            rTime / 3600, (rTime % 3600) / 60, (rTime % 3600) % 60))

    print('Hightest PSNR: ' + str(highest_psnr) + '\tSSIM: ' + str(highest_psnr_s) + '\t(Epoch' + str(psnr_ep) + ')')
    _ = eval(ImgNet, config.snapshots_folder + "Img_final.pth", save_plot=True)

    f = open('./data/best_score/best_scores.txt', 'w')
    sys.stdout = f # 이 문장 이후 print()문을 실행하면 terminal이 아니라 f 파일에 출력됨.
    print('Hightest PSNR: ' + str(highest_psnr) + '\tSSIM: ' + str(highest_psnr_s) + '\t(Epoch' + str(psnr_ep) + ')')
    sys.stdout = sys.__stdout__ # 다시 원래 terminal 출력을 복구.
    f.close()
    #                                                                                           #
    #############################################################################################

if __name__ == '__main__':
    # parser를 main에서 관리하는 게 아니라 전역 변수처럼 맨 위에 선언해서 관리해주는 게 좋다고 함.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images_path', type=str, default='./data/train_data/LOL')
    parser.add_argument('--lr', type=float, default=0.0001)   
    parser.add_argument('--weight_decay', type=float, default=0.00005)   
    parser.add_argument('--train_batch_size', type=int, default=16)   
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--snapshots_folder', type=str, default="models/")

    config = parser.parse_args()

    train(config)