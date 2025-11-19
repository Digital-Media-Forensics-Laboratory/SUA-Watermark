import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from PIL import Image


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import transforms

from AttGAN.data import check_attribute_conflict



from data import CelebA
import attacks

from model_data_prepare import prepare, init_inference_data
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


# init attacker
# def init_Attack(args_attack):
#     pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
#     return pgd_attack

if __name__ == "__main__":
    args_attack = parse()
    print(args_attack)
    # os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    # print("experiment dir is created")
    # os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
    # print("experiment config is saved")

    num1 = 200
    num2 = 1000
    num3 = 70
    weight = 1
    # pgd_attack = init_Attack(args_attack)

    with torch.no_grad():
        pt = torch.load('./Comparative_result/perturbation/single_step_128_8_attentiongan_1+1_StarGAN_0.3+1_HiSD_1+7_AttGAN_2.pt').cpu()

        # Init the attacked models
        test_dataloader = init_inference_data(args_attack)
        print("finished init the attacked models")

        
        n_samples = 10
        avg_img = None
        for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
            img_a = torch.clamp(img_a + pt, min = -1, max = 1)
            if avg_img == None:
                avg_img = img_a
            else:
                avg_img = img_a + avg_img
            
            if idx == n_samples-1:
                break

        avg_img = avg_img/n_samples
        out_file = './denoising.png'
        vutils.save_image(avg_img, out_file, nrow=1, normalize=True, range=(-1., 1.))

    