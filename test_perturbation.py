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



from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

perturbation_list = ["./Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3+1_HiSD_1+7_AttGAN_2.pt","./perturbation_attgan_2_expand_0.125_HiSD_1_0.325_stargan_0.31_0.3_attentiongan_0.3.pt","./perturbation_HUANG_128_8_stargan_0.1_attentiongan_0.3_0.003_attgan_2_0.44_HISD_1.pt","./Comparative_result/perturbation/single_step_128_8_0.1_StarGAN_0.3*AttentionGAN_1*HiSD_1+1_AttGAN_2.pt","./Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*HiSD_1+10_AttGAN_1.4.pt"]
for addr in perturbation_list:
    perturbation = torch.load(addr)
    p_sum = (perturbation * perturbation).sum()/(3*256*256)
    print(p_sum)
    print(perturbation)




