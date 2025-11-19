import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
import cv2
import numpy as np

import face_segmentation as fs

import torch
import torch.utils.data as data
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict



from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models


def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))  
    return args_attack

# args_attack = parse()
# attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
# print("finish prepare")

def canny_atom(input_image):
    input_image = input_image.detach().cpu().mul(255.0).numpy().transpose((1,2,0))
    # input_image = input_image.detach().cpu().mul(255.0).numpy().squeeze(0).transpose((1,2,0))
    #detach与data类似，将变量从图中分离；不同的是detach更为安全；cpu负责转换数据为cpu格式，mul则是[0,1]到[0,255]
    #squeeze撤去N通道，transpose更改tensor数据[c, h, w]为[h, w,C]
    # cv2.imwrite("./test0.png",input_image)
    out = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR).astype(np.uint8)

    # img = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(out, (3, 3), 0)
    canny = cv2.Canny(blur, 50, 150)
    canny_num = canny.sum()/255
    return canny_num

def canny_average(input_image, n):
    canny_sum=0
    canny_num=[]
    for img in input_image:
        t=canny_atom(img)
        canny_sum += t
        canny_num.append(t)
    return 2-[float(i) for i in canny_num]/(canny_sum/n)

def canny_atom_fs(input_image):
    
    input_image = input_image.detach().cpu().add(1).mul(255.0/2).numpy().transpose((1,2,0))
    # input_image = input_image.detach().cpu().mul(255.0).numpy().squeeze(0).transpose((1,2,0))
    #detach与data类似，将变量从图中分离；不同的是detach更为安全；cpu负责转换数据为cpu格式，mul则是[0,1]到[0,255]
    #squeeze撤去N通道，transpose更改tensor数据[c, h, w]为[h, w,C]
    # cv2.imwrite("./test0.png",input_image)
    # out = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR).astype(np.uint8)
    input_image = input_image.astype(np.uint8)
    cv2.imwrite("1.png",input_image)
    out,flag = fs.get_seg_face_image(input_image)
    if flag == 1:
        return 0
    # img = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(out, (3, 3), 0)
    canny = cv2.Canny(blur, 20, 70)
    cv2.imwrite("2.png",canny)
    canny_num = canny.sum()/255
    return canny_num

def canny_average_fs(input_image, n):
    canny_sum=0
    canny_num=[]
    num = 0
    for img in input_image:
        t=canny_atom_fs(img)
        canny_sum += t
        canny_num.append(t)
        if t != 0:
            num+=1
    for i in range(len(canny_num)):
        
        if canny_num[i] == 0:
            canny_num[i] = float(1)
        else:
            # canny_num[i] = 2-float(canny_num[i])/(canny_sum/num)
            canny_num[i] = float(canny_num[i])/(canny_sum/num)
    print(canny_num)
    return canny_num

# for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
#     if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
#         break

#     canny_average, canny_num = canny_average(img_a, 16)
#     # print(numpy.shape(img_a.numpy()[0]))
#     print(canny_average, canny_num)
#     break
if __name__ == "__main__":
    args_attack = parse()
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    print("finish prepare")
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break

        canny_num = canny_average_fs(img_a, 8)
        # print(numpy.shape(img_a.numpy()[0]))
        print(canny_num)
        