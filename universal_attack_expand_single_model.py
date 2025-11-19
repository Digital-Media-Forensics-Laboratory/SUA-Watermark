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

from model_data_prepare import prepare, prepare_single_model, prepare_dataset
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
print(args_attack)
# os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
# print("experiment dir is created")
# os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
# print("experiment config is saved")

# init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


pgd_attack = init_Attack(args_attack)
expand_factor = [0,1,1,10]

# 载入已有扰动
# pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)
# if args_attack.global_settings.universal_perturbation_path:

# pgd_attack.up = torch.load("./perturbation_attgan_2_expand_0.125_HiSD_1_0.325_stargan_0.3.pt")
# attack_3m = torch.load("./perturbation_attgan_2_expand_0.125_HiSD_1_0.325_stargan_0.3.pt")

# pgd_attack.up = torch.load("./Comparative_result/perturbation/single_step_weight_128_8_attentiongan_1+1_StarGAN_0.3.pt")
# attack_3m = torch.load("./Comparative_result/perturbation/single_step_weight_128_8_attentiongan_1+1_StarGAN_0.3.pt")

attack_dataloader, test_dataloader = prepare_dataset()

per_last = None

# attacking models
for modelid in range(4):

    # init the attacker models
    attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare_single_model(modelid)

    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
 
        if modelid == 1:
            # attack stargan
            solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)
        elif modelid == 0:
            # attack attentiongan
            attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)
        elif modelid == 2:
            # attack HiSD
            with torch.no_grad():
                c = E(img_a)
                c_trg = c
                s_trg = F(reference, 1)
                c_trg = T(c_trg, s_trg, 1)
                x_trg = G(c_trg)
                mask = abs(x_trg - img_a)
                mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
                mask[mask>0.5] = 1
                mask[mask<0.5] = 0
            pgd_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)

        else:
            # attack AttGAN
            att_b_list = [att_a]
            for i in range(attgan_args.n_attrs):
                tmp = att_a.clone()
                tmp[:, i] = 1 - tmp[:, i]
                tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
                att_b_list.append(tmp)

            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
                if i > 0:
                    att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
                with torch.no_grad():
                    gen_noattack = attgan.G(img_a, att_b_)
                x_adv, perturb = pgd_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)

        if not modelid == 0:
            # attack_expand
            pgd_attack.universal_perturb_expand_single_step_weight(img_a.cuda(), gen_models, per_last, expand_factor[modelid])

        

        torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
        print('save the SUA-Watermark')

    # torch.cuda.empty_cache()
    per_last = copy.deepcopy(pgd_attack.up)


    attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = None, None, None, None, None, None, None, None, None, None, None

with torch.no_grad():       
    _, _, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)