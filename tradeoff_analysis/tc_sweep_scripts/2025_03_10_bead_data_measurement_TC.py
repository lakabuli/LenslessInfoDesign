# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: info_jax_flax_23
#     language: python
#     name: python3
# ---

# Script for computing Tamura Coefficient (TC) on bead measurement datasets
# with various PSF patterns and sparsity levels. Computes Tamura coefficients
# for full images, noisy images, patches, and noisy patches.


import os
from jax import config
config.update("jax_enable_x64", True)
import sys
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
import jax.numpy as np
import numpy as onp
import tensorflow as tf
import tensorflow.keras as tfk

from lensless_helpers import *

from encoding_information import extract_patches
from encoding_information.image_utils import add_noise

diffuser_psf = load_diffuser_32()
one_psf = load_single_lens_uniform(32)
two_psf = load_two_lens_uniform(32)
three_psf = load_three_lens_uniform(32)
four_psf = load_four_lens_uniform(32)
five_psf = load_five_lens_uniform(32)
six_psf = load_six_lens_uniform(32)
seven_psf = load_seven_lens_uniform(32)
eight_psf = load_eight_lens_uniform(32)
nine_psf = load_nine_lens_uniform(32)
ten_psf = load_ten_lens_uniform(32)

seed_values_full = [4, 42, 31, 50, 77]

bias = 10
mean_photon_count_list = [100]
mean_photon_count = mean_photon_count_list[0]

psf_patterns = [one_psf, two_psf, three_psf, four_psf, five_psf, six_psf, seven_psf, eight_psf, nine_psf, diffuser_psf]
psf_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'diffuser']

sparsity_levels = [0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.5]

num_x = 96
num_y = 96
num_bead_imgs = 50000

patch_size = 32
num_patches = 10000
val_set_size = 1000
test_set_size = 1500
num_samples = 8
learning_rate = 1e-3
num_iters_per_epoch = 500
patience_val = 20

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/tc_values/bead_tc_values/'

for sparsity in sparsity_levels:
    for psf_index, psf_pattern in enumerate(psf_patterns):
        for seed_value in seed_values_full[:1]:
            np.random.seed(seed_value)
            dataset = np.zeros((num_bead_imgs, num_y, num_x))
            num_points_list = []
            for i in range(dataset.shape[0]):
                vol, num_points = make_bead_volume(sparsity, bead_width_scale=1, numx=num_x, numy=num_y)
                dataset[i] = vol
                num_points_list.append(num_points)
            assert(all(x == num_points_list[0] for x in num_points_list))
            num_points_dataset = num_points_list[0]
            print("mean of dataset: ", np.mean(dataset), ", max of dataset: ", np.max(dataset))
            dataset = dataset.astype(np.float32)
            dataset_photons = dataset / np.mean(dataset)
            dataset_photons = dataset_photons * mean_photon_count
            print("mean of photon count dataset: ", np.mean(dataset_photons), ", max of photon count dataset: ", np.max(dataset_photons))

            psf_data = convolved_dataset(psf_pattern, dataset_photons)
            psf_data += bias

            psf_data_full_noise = add_noise(psf_data, seed=seed_value)
            psf_data_patch = extract_patches(psf_data, patch_size=patch_size, num_patches=num_patches, seed=seed_value, verbose=True)
            psf_data_patch_noisy = add_noise(psf_data_patch, seed=seed_value)

            tam_list_full = []
            tam_list_full_noise = []
            tam_list_patch = []
            tam_list_patch_noisy = []
            for k in range(psf_data.shape[0]):
                full_img = psf_data[k]
                tam_list_full.append(compute_tamura(full_img))
                full_img_noise = psf_data_full_noise[k]
                tam_list_full_noise.append(compute_tamura(full_img_noise))
            for k in range(psf_data_patch.shape[0]):
                patch_img = psf_data_patch[k]
                tam_list_patch.append(compute_tamura(patch_img))
                patch_img_noisy = psf_data_patch_noisy[k]
                tam_list_patch_noisy.append(compute_tamura(patch_img_noisy))
            tam_list_full = np.array(tam_list_full)
            tam_list_full_noise = np.array(tam_list_full_noise)
            tam_list_patch = np.array(tam_list_patch)
            tam_list_patch_noisy = np.array(tam_list_patch_noisy)

            tamura_values = {
                'full': [np.mean(tam_list_full), np.std(tam_list_full)],
                'full_noise': [np.mean(tam_list_full_noise), np.std(tam_list_full_noise)],
                'patch': [np.mean(tam_list_patch), np.std(tam_list_patch)],
                'patch_noisy': [np.mean(tam_list_patch_noisy), np.std(tam_list_patch_noisy)]
            }
            np.save(save_dir + 'tamura_values_{}_sparsity_{}_photons_{}_psf.npy'.format(sparsity, mean_photon_count, psf_names[psf_index]), tamura_values)
