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

# Script for estimating Mutual Information (MI) on 3D bead volume datasets
# with various PSF patterns, depth planes, and sparsity levels. Sweeps across
# multiple seed values and saves MI estimates, lower/upper bounds, and validation loss histories.

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
import scipy

from lensless_helpers import *

from encoding_information import extract_patches
from encoding_information.models import PixelCNN
from encoding_information.models import PoissonNoiseModel
from encoding_information.image_utils import add_noise
from encoding_information import estimate_information

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

sparsity_levels = [0.02, 0.08, 0.01]

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

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/mi_estimates/3D_bead_mi_estimates/'

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
            

# Pre-generate volumes of data at each seed level

# for sparsity in sparsity_levels:
#     np.random.seed(data_generation_seed_value)
#     dataset = np.zeros((num_volumes_to_generate, num_depth_planes_max, 96, 96))
#     num_points_list = [] 
#     for dataset_idx in range(dataset.shape[0]):
#         volume = np.zeros((num_depth_planes_max, 96, 96))
#         num_points = np.zeros((num_depth_planes_max))
#         # fill each volume 
#         for depth_plane in range(num_depth_planes_max):
#             volume[depth_plane], num_points[depth_plane] = make_bead_volume(sparsity, bead_width_scale=1, numx=96, numy=96)
#         dataset[dataset_idx] = volume
#         num_points_list.append(num_points)
#     num_points_list = np.array(num_points_list)
#     num_points_in_dataset = np.mean(num_points_list, axis=0)
#     print(dataset.shape, num_points_in_dataset.shape, num_points_in_dataset) 
#     # save the volume  
#     np.save(volume_path + 'dataset_{}_images_{}_planes_{}_sparsity.npy'.format(num_volumes_to_generate, num_depth_planes_max, sparsity), dataset)
#     np.save(volume_path + 'num_points_{}_images_{}_planes_{}_sparsity.npy'.format(num_volumes_to_generate, num_depth_planes_max, sparsity), num_points_in_dataset)

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
            val_loss_log = []
            mi_estimates = []
            lower_bounds = []
            upper_bounds = []

            for seed_value in seed_values_full:
                patches = extract_patches(psf_data[:-test_set_size], patch_size=patch_size, num_patches=num_patches, seed=seed_value, verbose=True)
                test_patches = extract_patches(psf_data[-test_set_size:], patch_size=patch_size, num_patches=test_set_size, seed=seed_value, verbose=True)
                full_clean_patches = onp.concatenate([patches, test_patches])
                patches_noisy = add_noise(patches, seed=seed_value)
                test_patches_noisy = add_noise(test_patches, seed=seed_value)

                pixel_cnn = PixelCNN()
                val_loss_history = pixel_cnn.fit(patches_noisy, seed=seed_value, learning_rate=learning_rate, do_lr_decay=False, steps_per_epoch=num_iters_per_epoch, patience=patience_val)
                noise_model = PoissonNoiseModel()
                pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, patches_noisy,
                                                                                                    test_patches_noisy, clean_data=full_clean_patches,
                                                                                                    confidence_interval=0.95)
                print("PixelCNN estimated information: ", pixel_cnn_info)
                print("PixelCNN lower bound: ", pixel_cnn_lower_bound)
                print("PixelCNN upper bound: ", pixel_cnn_upper_bound)
                val_loss_log.append(val_loss_history)
                mi_estimates.append(pixel_cnn_info)
                lower_bounds.append(pixel_cnn_lower_bound)
                upper_bounds.append(pixel_cnn_upper_bound)
            np.save(save_dir + 'pixelcnn_mi_estimate_{}_planes_{}_sparsity_{}_photons_{}_psf_{}_normalize_{}_lr_{}_patience_{}_steps_per_epoch'.format(depth_plane_count, sparsity, mean_photon_count, psf_names[psf_index], normalize, learning_rate, patience_val, num_iters_per_epoch), np.array([mi_estimates, lower_bounds, upper_bounds]))
            np.save(save_dir + 'pixelcnn_val_loss_{}_planes_{}_sparsity_{}_photons_{}_psf_{}_normalize_{}_lr_{}_patience_{}_steps_per_epoch'.format(depth_plane_count, sparsity, mean_photon_count, psf_names[psf_index], normalize, learning_rate, patience_val, num_iters_per_epoch), np.array(val_loss_log, dtype=object))
