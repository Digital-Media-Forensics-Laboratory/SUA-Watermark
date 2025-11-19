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

import gc

def setup_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device('cuda')
            print("✓ CUDA initialized successfully")
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        except Exception as e:
            print(f"✗ CUDA initialization failed: {e}")
            device = torch.device('cpu')
            print("✓ Falling back to CPU")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = setup_cuda()

from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

perturbation_dir = './Comparative_result/perturbation'
os.makedirs(perturbation_dir, exist_ok=True)

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

perturbation_file = '128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4.pt'
args_attack.global_settings.universal_perturbation_path = os.path.join(perturbation_dir, perturbation_file)

# init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(
        model=None, 
        device=device,  
        epsilon=args_attack.attacks.epsilon, 
        k=args_attack.attacks.k, 
        a=args_attack.attacks.a, 
        star_factor=args_attack.attacks.star_factor, 
        attention_factor=args_attack.attacks.attention_factor, 
        att_factor=args_attack.attacks.att_factor, 
        HiSD_factor=args_attack.attacks.HiSD_factor, 
        args=args_attack.attacks
    )
    return pgd_attack

pgd_attack = init_Attack(args_attack)
expand_factor = 2

print("Initializing new universal perturbation...")
pgd_attack.up = torch.zeros((1, 3, 256, 256), device=device)
# pgd_attack.up = torch.zeros((1, 3, 256, 256), device=device)
attack_3m = pgd_attack.up.clone()

# init the attacker models
print("Initializing models...")
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("Finished initializing the attacked models")

# attacking models with error handling
for i in range(1):
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
        
        try:
            img_a = img_a.to(device)
            att_a = att_a.to(device)
            att_a = att_a.type(torch.float)

            if idx == 0:
                img_a_last = copy.deepcopy(img_a)

            print(f"Processing sample {idx}...")

            # attack stargan
            solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

            # attack attentiongan
            attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

            # attack HiSD 
            if all([x is not None for x in [F, T, G, E, reference]]):
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
                pgd_attack.universal_perturb_HiSD(img_a, transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)
            else:
                print(f"  Skipping HiSD - models not loaded")

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

            # attack_expand
            pgd_attack.universal_perturb_expand_single_step_weight(img_a, gen_models, attack_3m, expand_factor)

            # save
            torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
            print(f'  SUA-Watermark saved to {args_attack.global_settings.universal_perturbation_path}')

        except Exception as e:
            print(f"❌ Error at sample {idx}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

print('The size of SUA-Watermark: ', pgd_attack.up.shape)

print("Starting final evaluation...")
try:
    evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)
    print("✓ Evaluation completed successfully")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")

print("Experiment completed!")
