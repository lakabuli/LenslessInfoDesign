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

# Script for estimating Mutual Information (MI) on bead datasets with diffuser PSF patterns
# and various aperture sizes. Sweeps across multiple seed values and saves MI estimates,
# lower/upper bounds, and validation loss histories.

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
from encoding_information.models import PixelCNN
from encoding_information.models import PoissonNoiseModel
from encoding_information.image_utils import add_noise
from encoding_information import estimate_information

diffuser_psf = load_diffuser_32()
aperture_psf = np.copy(diffuser_psf)
aperture_psf[:5] = 0
aperture_psf[-5:] = 0
aperture_psf[:,:5] = 0
aperture_psf[:,-5:] = 0
aperture_psf_2 = np.copy(diffuser_psf)
aperture_psf_2[:2] = 0
aperture_psf_2[-2:] = 0
aperture_psf_2[:,:2] = 0
aperture_psf_2[:,-2:] = 0

aperture_psf_8 = np.copy(diffuser_psf)
aperture_psf_8[:8] = 0
aperture_psf_8[-8:] = 0
aperture_psf_8[:,:8] = 0
aperture_psf_8[:,-8:] = 0

aperture_psf_10 = np.copy(diffuser_psf)
aperture_psf_10[:10] = 0
aperture_psf_10[-10:] = 0
aperture_psf_10[:,:10] = 0
aperture_psf_10[:,-10:] = 0

aperture_psf_12 = np.copy(diffuser_psf)
aperture_psf_12[:12] = 0
aperture_psf_12[-12:] = 0
aperture_psf_12[:,:12] = 0
aperture_psf_12[:,-12:] = 0

aperture_psf_14 = np.copy(diffuser_psf)
aperture_psf_14[:14] = 0
aperture_psf_14[-14:] = 0
aperture_psf_14[:,:14] = 0
aperture_psf_14[:,-14:] = 0

seed_values_full = [4, 42, 31, 50, 77]

bias = 10
mean_photon_count_list = [100]
mean_photon_count = mean_photon_count_list[0]

psf_patterns = [aperture_psf_2, aperture_psf_8, aperture_psf_10, aperture_psf_12, diffuser_psf]
psf_names = ['aperture_2', 'aperture_8', 'aperture_10', 'aperture_12', 'diffuser']

sparsity_levels = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

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

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/mi_estimates/bead_mi_estimates/'

for sparsity in sparsity_levels:
    for psf_index, psf_pattern in enumerate(psf_patterns):
        val_loss_log = []
        mi_estimates = []
        lower_bounds = []
        upper_bounds = []
        for seed_value in seed_values_full:
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
        np.save(save_dir + 'pixelcnn_mi_estimate_{}_sparsity_{}_photons_{}_psf_{}_lr_{}_patience_{}_steps_per_epoch'.format(sparsity, mean_photon_count, psf_names[psf_index], learning_rate, patience_val, num_iters_per_epoch), np.array([mi_estimates, lower_bounds, upper_bounds]))
        np.save(save_dir + 'pixelcnn_val_loss_{}_sparsity_{}_photons_{}_psf_{}_lr_{}_patience_{}_steps_per_epoch'.format(sparsity, mean_photon_count, psf_names[psf_index], learning_rate, patience_val, num_iters_per_epoch), np.array(val_loss_log, dtype=object))
