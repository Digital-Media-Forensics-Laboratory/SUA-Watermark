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

# init attacker
def init_Attack(args_attack):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    return pgd_attack, device

if __name__ == "__main__":
    # Use standard -i and -o parameters
    parser = argparse.ArgumentParser(description='Test the watermark effect on a single image')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the image to be tested')
    parser.add_argument('-o', '--output', type=str, default='./canny_results', help='Output directory')
    args_cmd = parser.parse_args()
    
    # Create output directory
    os.makedirs(args_cmd.output, exist_ok=True)
    
    # Check if the image exists
    if not os.path.exists(args_cmd.input):
        print(f"Error: Image file does not exist - {args_cmd.input}")
        sys.exit(1)
    
    args_attack = parse()
    print(args_attack)

    pgd_attack, device = init_Attack(args_attack)

    # load the trained CMUA-Watermark
    if args_attack.global_settings.universal_perturbation_path:
        pgd_attack.up = torch.load('Comparative_result/perturbation/128_8_0.1_StarGAN_0.3*AttentionGAN_1*AttGAN_1.4.pt')
        pgd_attack.up = pgd_attack.up.to(device)

    # Init the attacked models
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F_, T, G, E, reference, gen_models = prepare()
    print("finished init the attacked models")

    tf = transforms.Compose([
        transforms.Resize(args_attack.global_settings.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    try:
        image = Image.open(args_cmd.input)
        img = image.convert("RGB")
        img = tf(img).unsqueeze(0).to(device)  # [1, 3, 256, 256]
        print(f"Successfully loaded image: {args_cmd.input}")
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    # Extract filename without extension from image path
    image_name = os.path.splitext(os.path.basename(args_cmd.input))[0]
    
    print(f"=== Starting to test all GAN models with {image_name} ===")

    # 1. AttGAN inference and evaluating
    print("Testing AttGAN...")
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    
    # Create attribute labels for the image (assuming all attributes are 0)
    att_a = torch.zeros(1, attgan_args.n_attrs).to(device)
    att_a = att_a.type(torch.float)
    
    att_b_list = [att_a]
    for i in range(attgan_args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]  # Flip the i-th attribute
        tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
        att_b_list.append(tmp)
    
    samples = [img, img + pgd_attack.up]
    noattack_list = []
    
    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
        with torch.no_grad():
            gen = attgan.G(img + pgd_attack.up, att_b_)
            gen_noattack = attgan.G(img, att_b_)
        samples.append(gen)
        noattack_list.append(gen_noattack)
        l1_error += F.l1_loss(gen, gen_noattack)
        l2_error += F.mse_loss(gen, gen_noattack)
        l0_error += (gen - gen_noattack).norm(0)
        min_dist += (gen - gen_noattack).norm(float('-inf'))
        if F.mse_loss(gen, gen_noattack) > 0.05:
            n_dist += 1
        n_samples += 1
    
    ############# Save AttGAN results #############
    # Save original image
    out_file = f'{args_cmd.output}/{image_name}_AttGAN_original.jpg'
    vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    for j in range(len(samples)-2):
        # Save adversarial generated images
        out_file = f'{args_cmd.output}/{image_name}_AttGAN_advgen_{j}.jpg'
        vutils.save_image(samples[j+2].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
         # Save original generated images
        out_file = f'{args_cmd.output}/{image_name}_AttGAN_gen_{j}.jpg'
        vutils.save_image(noattack_list[j].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    print('AttGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    # 2. StarGAN inference and evaluating
    print("Testing StarGAN...")
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    
    # Create conditions for StarGAN
    c_org = torch.zeros(1, solver.c_dim).to(device)
    
    x_noattack_list, x_fake_list = solver.test_universal_model_level(0, img, c_org, pgd_attack.up, args_attack.stargan)
    
    for j in range(len(x_fake_list)):
        gen_noattack = x_noattack_list[j]
        gen = x_fake_list[j]
        l1_error += F.l1_loss(gen, gen_noattack)
        l2_error += F.mse_loss(gen, gen_noattack)
        l0_error += (gen - gen_noattack).norm(0)
        min_dist += (gen - gen_noattack).norm(float('-inf'))
        if F.mse_loss(gen, gen_noattack) > 0.05:
            n_dist += 1
        n_samples += 1
    
    ############# Save StarGAN results #############
    out_file = f'{args_cmd.output}/{image_name}_stargan_original.jpg'
    vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    for j in range(len(x_fake_list)):
        out_file = f'{args_cmd.output}/{image_name}_stargan_gen_{j}.jpg'
        vutils.save_image(x_noattack_list[j].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        out_file = f'{args_cmd.output}/{image_name}_stargan_advgen_{j}.jpg'
        vutils.save_image(x_fake_list[j].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    print('StarGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    # 3. AttentionGAN inference and evaluating
    print("Testing AttentionGAN...")
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    
    # Create conditions for AttentionGAN
    c_org_attn = torch.zeros(1, attentiongan_solver.c_dim).to(device)
    
    x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(0, img, c_org_attn, pgd_attack.up, args_attack.AttentionGAN)
    
    for j in range(len(x_fake_list)):
        gen_noattack = x_noattack_list[j]
        gen = x_fake_list[j]
        l1_error += F.l1_loss(gen, gen_noattack)
        l2_error += F.mse_loss(gen, gen_noattack)
        l0_error += (gen - gen_noattack).norm(0)
        min_dist += (gen - gen_noattack).norm(float('-inf'))
        if F.mse_loss(gen, gen_noattack) > 0.05:
            n_dist += 1
        n_samples += 1
    
    ############# Save AttentionGAN results #############
    out_file = f'{args_cmd.output}/{image_name}_attentiongan_original.jpg'
    vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    for j in range(len(x_fake_list)):
        out_file = f'{args_cmd.output}/{image_name}_attentiongan_gen_{j}.jpg'
        vutils.save_image(x_noattack_list[j].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        out_file = f'{args_cmd.output}/{image_name}_attentiongan_advgen_{j}.jpg'
        vutils.save_image(x_fake_list[j].cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
    
    print('AttentionGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    # 4. HiSD inference and evaluating
    print("Testing HiSD...")
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    
    with torch.no_grad():
        # clean
        c = E(img)
        c_trg = c
        s_trg = F_(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        gen_noattack = G(c_trg)
        # adv
        c = E(img + pgd_attack.up)
        c_trg = c
        s_trg = F_(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        gen = G(c_trg)
        
        mask = abs(gen_noattack - img)
        mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
        mask[mask>0.5] = 1
        mask[mask<0.5] = 0

        point_mask = abs(gen - gen_noattack)
        point_mask = point_mask[0,0,:,:] + point_mask[0,1,:,:] + point_mask[0,2,:,:]
        point_mask[point_mask>0.3] = 1
        point_mask[point_mask<0.3] = 0

        l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
        l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
        l0_error += (gen - gen_noattack).norm(0)
        min_dist += (gen - gen_noattack).norm(float('-inf'))
        if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
            n_dist += 1
        n_samples += 1

        ############# Save HiSD results #############
        out_file = f'{args_cmd.output}/{image_name}_HiSD_original.jpg'
        vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
        out_file = f'{args_cmd.output}/{image_name}_HiSD_gen.jpg'
        vutils.save_image(gen_noattack.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
        out_file = f'{args_cmd.output}/{image_name}_HiSD_advgen.jpg'
        vutils.save_image(gen.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
        out_file = f'{args_cmd.output}/{image_name}_HiSD_mask.jpg'
        vutils.save_image(mask.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
        out_file = f'{args_cmd.output}/{image_name}_HiSD_point_mask.jpg'
        vutils.save_image(point_mask.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))

    print('HiSD {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    print(f"=== Testing of all GAN models on {image_name} completed! ===")
    print(f"Results saved to: {args_cmd.output}")
