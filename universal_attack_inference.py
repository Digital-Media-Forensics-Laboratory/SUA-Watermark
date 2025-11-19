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

# load the trained SUA-Watermark
# if args_attack.global_settings.universal_perturbation_path:
#     # pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)
#     # pgd_attack.up = torch.load('perturbation_weight.pt')
#     pgd_attack.up = torch.load('./Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3+1_HiSD_1+7_AttGAN_2.pt')
#     # pgd_attack.up = torch.load('perturbation_HUANG_128_8_stargan_0.1_attentiongan_0.3_0.003_attgan_2_0.44_HISD_1.pt')

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
    print('The size of SUA-Watermark: ', pgd_attack.up.shape)
    evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)