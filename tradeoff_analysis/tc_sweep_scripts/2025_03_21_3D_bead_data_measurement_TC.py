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

# Script for computing Tamura Coefficient (TC) on 3D bead volume measurement datasets
# with various PSF patterns, depth planes, and sparsity levels. Computes Tamura
# coefficients for full images, noisy images, patches, and noisy patches.

import os
from jax import config
config.update("jax_enable_x64", True)
import sys
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
import jax.numpy as np
import numpy as onp
import tensorflow as tf
import tensorflow.keras as tfk
import scipy

from lensless_helpers import *

from encoding_information import extract_patches
from encoding_information.image_utils import add_noise

lenslet_locations = [[16, 16], [7, 9], [23, 21], [8, 24], [21, 5], [27, 13], [4, 16], [16, 26], [14, 7], [26, 26]]
lenslet_counts = [1, 2, 3, 4, 5]
bead_plane_counts = [1, 2, 3, 4, 5]
num_depth_planes_max = 5
depth_plane_ordering = [2, 0, 4, 1, 3]
psf_names = ['one', 'two', 'three', 'four', 'five']
normalize = False

seed_values_full = [4, 42, 31, 50, 77]

bias = 10
mean_photon_count_list = [100]
mean_photon_count = mean_photon_count_list[0]

volume_path = '/home/lkabuli_waller/10tb_extension/toy_volume_datasets/'

data_generation_seed_value = 42
num_volumes_to_generate = 20000

sparsity_levels = [0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

num_x = 96
num_y = 96

patch_size = 32
num_patches = 10000
val_set_size = 1000
test_set_size = 1500
num_samples = 8
learning_rate = 1e-3
num_iters_per_epoch = 500
patience_val = 20

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/tc_values/3D_bead_tc_values/'

volume_psfs = []
for lenslet_count in lenslet_counts:
    volume_psf = np.zeros((num_depth_planes_max, 32, 32))
    for i in range(lenslet_count):
        volume_psf[depth_plane_ordering[i]][lenslet_locations[i][0]][lenslet_locations[i][1]] = 1
        volume_psf[depth_plane_ordering[i]] = scipy.ndimage.gaussian_filter(volume_psf[depth_plane_ordering[i]], sigma=0.8)
    if normalize == True:
        volume_psf = volume_psf / np.sum(volume_psf)
    volume_psfs.append(volume_psf)
    print(np.sum(volume_psf), np.sum(volume_psf[2]))

seed_value = seed_values_full[0]

for sparsity in sparsity_levels:
    for depth_plane_count in range(1, num_depth_planes_max + 1):
        dataset = np.load(volume_path + 'dataset_{}_images_{}_planes_{}_sparsity.npy'.format(num_volumes_to_generate, num_depth_planes_max, sparsity))
        num_points_in_dataset = np.load(volume_path + 'num_points_{}_images_{}_planes_{}_sparsity.npy'.format(num_volumes_to_generate, num_depth_planes_max, sparsity))
        print(dataset.shape, num_points_in_dataset.shape, num_points_in_dataset)
        for z in range(num_depth_planes_max):
            dataset[:, z] = dataset[:, z] / np.mean(dataset[:, z]) * mean_photon_count
        print("the invalid planes are: ", depth_plane_ordering[depth_plane_count:])
        for plane_location in depth_plane_ordering[depth_plane_count:]:
            dataset[:, plane_location] = 0
            num_points_in_dataset[plane_location] = 0
        num_points_in_dataset = np.sum(num_points_in_dataset)
        print('mean of photon dataset: ', np.mean(dataset), "mean of nonzero depths: ", np.mean(dataset[:, depth_plane_ordering[:depth_plane_count]]), ", max of photon dataset: ", np.max(dataset))
        for psf_index, psf_pattern in enumerate(volume_psfs):
            psf_data = convolve_volume_dataset(psf_pattern, dataset, size=65)
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
            np.save(save_dir + 'tamura_values_{}_planes_{}_sparsity_{}_photons_{}_psf_{}_normalize.npy'.format(depth_plane_count, sparsity, mean_photon_count, psf_names[psf_index], normalize), tamura_values)
