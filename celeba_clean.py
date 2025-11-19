import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from canny import canny_average

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict



from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

from face_segmentation import get_seg_face_image,celeba_clean_face
import numpy as np

img_path = "./data/img_align_celeba"
clean_num = celeba_clean_face(img_path)
print(clean_num)