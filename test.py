import glob
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from loss_functions import PSNR, SSIM
from option import opt
from torchvision import transforms
from Guassian import Guassian_downsample
from torch.autograd import Variable
from build_models import VSR
from img_preprocess import imread, img_trans, modcrop, img_normal, bgr2ycbcr


def test(model, epochs, use_gpu):
    test_datasets = open(opt.test_data, 'rt').read().splitlines()
    psnr_datasets_all = 0
    ssim_datasets_all = 0
    for data_test in test_datasets:
        test_list = os.listdir(data_test)
        for test_name in test_list:
            with open('./results/test_results.txt', 'a+') as f:
                f.write('--------------------{}--------------------'.format(test_name)+ '\n')
            print('--------------------{}--------------------'.format(test_name))
            inList = sorted(glob.glob(os.path.join(data_test, test_name, '*.png')))
            psnr_all = 0
            ssim_all = 0
            HR_all = []
            trans_tensor = transforms.ToTensor()
            for i in range(0, len(inList)):
                img = imread(inList[i])
                HR_all.append(img)
            HR_all = [trans_tensor(HR) for HR in HR_all]
            HR_all = torch.stack(HR_all, dim=1)

            LR = Guassian_downsample(HR_all, opt.scale)
            LR = LR.unsqueeze(0)

            if use_gpu:
                LR = Variable(LR).cuda()
                HR_all = Variable(HR_all).cuda()

            start_time = time.time()
            with torch.no_grad():
                outputs = model(LR)
            end_time = time.time()
            cost_time = end_time - start_time
            with open('./results_time/time_all.txt', 'a+') as f:
                f.write('times = {:.6f}'.format(cost_time / 100) + '\n')
            outputs = outputs.squeeze(0)

            for i in range(0, len(inList)):
                prediction, labels = outputs[:, i, :, :],  HR_all[:, i, :, :]

                # Y channels calculation
                prediction = prediction.permute(1, 2, 0).cpu().numpy()
                save_img(prediction, data_test, test_name, False, i)
                prediction = torch.from_numpy(bgr2ycbcr(prediction)).unsqueeze(0).unsqueeze(0)
                prediction = prediction.cuda()

                labels = labels.permute(1, 2, 0).cpu().numpy()
                labels = torch.from_numpy(bgr2ycbcr(labels)).unsqueeze(0).unsqueeze(0)
                labels = labels.cuda()

                psnr = PSNR(labels, prediction)
                ssim = SSIM(labels, prediction)
                psnr_all += psnr
                ssim_all += ssim
                print('epochs:{},  psnr = {:.6f}, ssim={:.6f},'.format(epochs, psnr, ssim))
                with open('./results/test_results.txt', 'a+') as f:
                    f.write('epochs:{},  psnr = {:.6f}, ssim={:.6f},'.format(epochs, psnr, ssim) + '\n')

            psnr_avg = psnr_all / len(inList)
            ssim_avg = ssim_all / len(inList)
            print('==> Average PSNR = {:.6f}'.format(psnr_avg))
            print('==> Average SSIM = {:.6f}'.format(ssim_avg))
            with open('./results/test_results.txt', 'a+') as f:
                f.write('==> Average PSNR_Avg = {:.6f}'.format(psnr_avg) + '\n')
                f.write('==> Average SSIM_Avg = {:.6f}'.format(ssim_avg) + '\n')
            psnr_datasets_all += psnr_avg
            ssim_datasets_all += ssim_avg
        psnr_datasets_avg = psnr_datasets_all / len(test_list)
        ssim_datasets_avg = ssim_datasets_all / len(test_list)
        print('==> Average PSNR = {:.6f}'.format(psnr_datasets_avg))
        print('==> Average SSIM = {:.6f}'.format(ssim_datasets_avg))
        with open('./results/test_results.txt', 'a+') as f:
            f.write('==> {} Average PSNR_Avg = {:.6f}'.format(data_test, psnr_datasets_avg) + '\n')
            f.write('==> {} Average SSIM_Avg = {:.6f}'.format(data_test, ssim_datasets_avg) + '\n')
        with open('./results_all/test_results.txt', 'a+') as f:
            f.write('epochs:{} ==> {} Average PSNR_Avg = {:.6f}'.format(epochs, data_test, psnr_datasets_avg) + '\n')


def save_img(prediction, data_test, test_name, att, num):
    '''

    :param prediction:  img
    :param data_test:   path1
    :param test_name:   path2
    :param att: True img [0, 255], False [0, 1]
    :return:
    '''

    if att == True:
        save_dir = os.path.join(opt.image_out, data_test, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        save_dir = os.path.join(opt.image_out, data_test, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    model = VSR()

    path_checkpoint = "./X4_epoch_75.pth"
    checkpoints = torch.load(path_checkpoint)
    model.load_state_dict(checkpoints['net'])
    epochs = checkpoints['epoch']
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    test_datasets = opt.test_datasets
    test(model, epochs, use_gpu)




