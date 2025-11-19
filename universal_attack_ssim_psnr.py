import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict

from skimage.measure import  compare_psnr,compare_ssim
from skimage.metrics import structural_similarity,peak_signal_noise_ratio

from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models


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
# print(args_attack)
# os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
# print("experiment dir is created")
# os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
# print("experiment config is saved")

# init attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


pgd_attack = init_Attack(args_attack)

# load the trained CMUA-Watermark
# if args_attack.global_settings.universal_perturbation_path:
#     # pgd_attack.up = torch.load('./Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3+1_HiSD_1+7_AttGAN_2.pt')
#     # pgd_attack.up = torch.load('perturbation_weight.pt')
#     # pgd_attack.up = torch.load('perturbation_CMUAV2.pt')
#     pgd_attack.up = torch.load('./Comparative_result/perturbation/CMUA_8_0.1.pt')


dir_list=['./Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3+1_HiSD_1+7_AttGAN_2.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4*HiSD_1.pt',
          './Comparative_result/perturbation/128_8_0.1_attentiongan_1.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4*HiSD_1.pt',
          './Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1+1_AttGAN_2.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4+1_HiSD_1.pt',
          './Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4+2_HiSD_1.pt'
          ]

# Init the attacked models
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("finished init the attacked models")

for perturbation_dir in dir_list:

    print("************************************************************************************************")
    print(perturbation_dir.split('/')[-1])
    pgd_attack.up = torch.load(perturbation_dir)

    l_psnr = 0.0
    l_ssim = 0.0

    k_psnr = 0.0
    k_ssim = 0.0

    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break

        # img_a = gauss
        if idx == 1000:
            break
        
        img_a_adv = torch.clamp(img_a.cpu() + pgd_attack.up.cpu(), min=-1, max=1).cpu().detach_()

        
        
        img_a_1 = img_a.cpu().transpose(1,3).numpy()[0]
        img_a_adv_1 = img_a_adv.transpose(1,3).cpu().numpy()[0]

        img_a_256 = (img_a.cpu().detach_()+1)*255/2
        img_a_256 = torch.tensor(img_a_256, dtype=torch.uint8)
        img_a_256 = img_a_256.transpose(1,3)
        img_a_256 = img_a_256.numpy()[0]
        

        img_a_adv_256 = (img_a_adv+1)*255/2
        img_a_adv_256 = torch.tensor(img_a_adv_256, dtype=torch.int)
        img_a_adv_256 = img_a_adv_256.transpose(1,3)
        img_a_adv_256 = img_a_adv_256.numpy()[0]
        

        # print(img_a_256)
        # print(img_a_adv_256)

        # for i in range(3):
            # l_psnr += compare_psnr(img_a_256[i], img_a_adv_256[i],data_range=255)/3
        l_psnr += compare_psnr(img_a_256, img_a_adv_256,data_range=255)
        l_ssim += compare_ssim(img_a_256, img_a_adv_256,data_range=255,multichannel=True)

        k_psnr += peak_signal_noise_ratio(img_a_1, img_a_adv_1)
        k_ssim += structural_similarity(img_a_1, img_a_adv_1,multichannel=True)

    print(l_psnr/1000,l_ssim/1000,k_psnr/1000,k_ssim/1000)