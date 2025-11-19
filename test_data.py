import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from canny import canny_average
from canny import canny_average_fs

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict

import cv2

from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

from models import *
from config import Config
from torch.nn import DataParallel
import numpy as np
class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        
    return args_attack


args_attack = parse()
print(args_attack)

# init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


pgd_attack = init_Attack(args_attack)

# 载入已有扰动
# if args_attack.global_settings.universal_perturbation_path:
#     pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)



# init the attacker models
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("finished init the attacked models, only attack 2 epochs")

# attacking models
for i in range(1):
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
        # img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        # att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        # att_a = att_a.type(torch.float)

        # if idx == 0:
        #     img_a_last = copy.deepcopy(img_a)

        img = vutils.make_grid(img_a[0]).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        # img = img[32:224, 32:224, [2, 1, 0]]
        if idx == 10:
            cv2.imwrite('11111111.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            break