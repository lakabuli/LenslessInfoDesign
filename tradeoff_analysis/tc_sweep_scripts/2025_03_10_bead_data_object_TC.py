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

# Script for computing Tamura Coefficient (TC) on bead object datasets
# with various sparsity levels. Computes Tamura coefficients for full images
# and cropped center regions.

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

seed_values_full = [4, 42, 31, 50, 77]

bias = 10
mean_photon_count_list = [100]
mean_photon_count = mean_photon_count_list[0]

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
        tam_list_full = []
        tam_list_crop = []
        for k in range(dataset_photons.shape[0]):
            full_image = dataset_photons[k]
            tam_list_full.append(compute_tamura(full_image))
            crop_image = full_image[32:64, 32:64]
            tam_list_crop.append(compute_tamura(crop_image))
        tam_list_full = np.array(tam_list_full)
        tam_list_crop = np.array(tam_list_crop)

        tamura_values = {
            'full': [np.mean(tam_list_full), np.std(tam_list_full)],
            'crop': [np.mean(tam_list_crop), np.std(tam_list_crop)]
        }
        np.save(save_dir + 'tamura_values_object_{}_sparsity_{}_photons.npy'.format(sparsity, mean_photon_count), tamura_values)
