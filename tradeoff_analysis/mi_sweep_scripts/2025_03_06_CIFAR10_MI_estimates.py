# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: infotheory2025py11
#     language: python
#     name: python3
# ---

# Script for estimating Mutual Information (MI) on CIFAR10 dataset
# with various PSF patterns. Sweeps across multiple seed values and saves
# MI estimates, lower/upper bounds, and validation loss histories.

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

# set eligible psfs
psf_patterns = [one_psf, two_psf, three_psf, four_psf, five_psf, six_psf, seven_psf, eight_psf, nine_psf, diffuser_psf]
psf_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'diffuser']


patch_size = 32
num_patches = 10000
val_set_size = 1000
test_set_size = 1500
num_samples = 8
learning_rate = 1e-3
num_iters_per_epoch = 500
patience_val = 20

dataset_name = 'cifar10'

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/mi_estimates/cifar10_mi_estimates/'

for mean_photon_count in mean_photon_count_list:
    for psf_index, psf_pattern in enumerate(psf_patterns):
        val_loss_log = []
        mi_estimates = []
        lower_bounds = []
        upper_bounds = []
        for seed_value in seed_values_full:
            if dataset_name == 'cifar10':
                (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
            else:
                print("Invalid dataset name. Please use 'mnist' or 'cifar10'")
                break
            data = onp.concatenate((x_train, x_test), axis=0)
            if len(data.shape) == 4:
                data = tf.image.rgb_to_grayscale(data).numpy()
                data = data.squeeze()
            labels = onp.concatenate((y_train, y_test), axis=0)
            data = data.astype(np.float32)
            data = data / np.mean(data)
            data = data * mean_photon_count
            random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)
            psf_data = convolved_dataset(psf_pattern, random_data)
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
        np.save(save_dir + '{}_pixelcnn_mi_estimate_{}_photons_{}_psf_{}_lr_{}_patience_{}_steps_per_epoch'.format(dataset_name, mean_photon_count, psf_names[psf_index], learning_rate, patience_val, num_iters_per_epoch), np.array([mi_estimates, lower_bounds, upper_bounds]))
        np.save(save_dir + '{}_pixelcnn_val_loss_{}_photons_{}_psf_{}_lr_{}_patience_{}_steps_per_epoch'.format(dataset_name, mean_photon_count, psf_names[psf_index], learning_rate, patience_val, num_iters_per_epoch), np.array(val_loss_log, dtype=object))
