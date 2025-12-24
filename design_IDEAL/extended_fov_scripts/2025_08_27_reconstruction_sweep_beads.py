# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: infobayes
#     language: python
#     name: python3
# ---

# Script for running reconstruction sweeps on ideal bead PSF designs (random and heuristic initialization).
# Evaluates reconstruction models on bead datasets with varying learning rates and random seeds.


import sys
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager/')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from encoding_information.gpu_utils import limit_gpu_memory_growth
import random
limit_gpu_memory_growth()
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

from reconstruction_architecture import main_recon, CFG

from lensless_helpers import load_single_lens_uniform, load_two_lens_uniform, load_three_lens_uniform, load_four_lens_uniform
from lensless_helpers import load_five_lens_uniform, load_six_lens_uniform, load_seven_lens_uniform, load_eight_lens_uniform, load_nine_lens_uniform

one_psf = load_single_lens_uniform()
two_psf = load_two_lens_uniform()
three_psf = load_three_lens_uniform()
four_psf = load_four_lens_uniform()
five_psf = load_five_lens_uniform()
six_psf = load_six_lens_uniform()
seven_psf = load_seven_lens_uniform()
eight_psf = load_eight_lens_uniform()
nine_psf = load_nine_lens_uniform()

ideal_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/IDEAL_psfs/'

bead_ideal_psf = np.load(ideal_dir + 'ideal_bead_0.02_psf.npy')
bead_ideal_heuristic_psf = np.load(ideal_dir + 'ideal_heuristic_bead_0.02_psf.npy')

# %%
psf_list = [bead_ideal_heuristic_psf, bead_ideal_psf, one_psf, two_psf, three_psf, four_psf, five_psf, six_psf, seven_psf, eight_psf, nine_psf]
psf_names_list = ['ideal_heuristic_beads', 'ideal_beads', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/reconstructions/'

seed_list = [4, 42, 31, 50, 77]
model_type_list = ['medium']
lr_list = [5e-3, 2e-3]
dataset_list = ['bead_0.02']
bead_sparsity = 0.02

for model_type in model_type_list:
    for lr_value in lr_list:
        for dataset in dataset_list:
            for psf_name, psf in zip(psf_names_list, psf_list):
                for seed_value in seed_list:
                    CFG.dataset = dataset
                    CFG.normalize = True
                    CFG.add_noise = True
                    CFG.mean_val = 100.0
                    CFG.bead_sparsity = bead_sparsity
                    CFG.per_item = False
                    CFG.model_type = model_type
                    CFG.epochs = 200
                    CFG.loss_type = 'mse'
                    CFG.lr = lr_value
                    CFG.batch_size = 64
                    if __name__ == "__main__":
                        print("recon for {} dataset, {} psf".format(dataset, psf_name))
                        torch.manual_seed(seed_value)
                        random.seed(seed_value)
                        logs, test_result = main_recon(psf, psf_name, visualize=False, save_dir=save_dir, seed_value=seed_value)
