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

# Script for estimating Mutual Information (MI) on experimental imaging data (GT, RML, DiffuserCam).
# Uses PixelCNN with Poisson noise model. Sweeps across multiple seed values and imaging systems.
# Saves MI estimates, lower/upper bounds, and validation loss histories.

import os
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

import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean

save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/experimental_eval/data/'

bias = 10
train_set_index_start = 1000
test_set_index_start = 0
num_test_images = 1000
num_train_images = 10000

mean_photon_count = 500

learning_rate = 1e-3
num_iters_per_epoch = 500
patience_val = 20

imager_types = ['gt', 'rml', 'diffuser']
seed_values = [42, 4, 31, 50, 77]
patch_size = 100

for seed_value in seed_values:
    for imager_type in imager_types:
        dataset_path = save_dir + '{}_dataset_{}_patch_{}_images.npy'.format(imager_type, patch_size, num_train_images + num_test_images)
        full_dataset = np.load(dataset_path).astype(np.float32)
        print(full_dataset.shape, full_dataset.dtype, np.mean(full_dataset), np.std(full_dataset))

        full_dataset_photons = full_dataset / np.mean(full_dataset)
        full_dataset_photons = full_dataset_photons * mean_photon_count
        print(np.mean(full_dataset_photons))
        full_dataset_photons = full_dataset_photons + bias

        train_set = full_dataset_photons[train_set_index_start:train_set_index_start + num_train_images]
        test_set = full_dataset_photons[test_set_index_start:test_set_index_start + num_test_images]

        train_patches = extract_patches(train_set, patch_size=patch_size, num_patches=num_train_images, seed=seed_value, verbose=True)
        test_patches = extract_patches(test_set, patch_size=patch_size, num_patches=num_test_images, seed=seed_value, verbose=True)
        full_clean_patches = onp.concatenate([train_patches, test_patches])

        train_patches_noisy = add_noise(train_patches, seed=seed_value)
        test_patches_noisy = add_noise(test_patches, seed=seed_value)

        pixel_cnn = PixelCNN()
        val_loss_history = pixel_cnn.fit(train_patches_noisy, seed=seed_value, learning_rate=learning_rate, do_lr_decay=False, steps_per_epoch=num_iters_per_epoch, patience=patience_val)
        noise_model = PoissonNoiseModel()
        pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, train_patches_noisy,
                                                                                            test_patches_noisy, clean_data=full_clean_patches,
                                                                                            confidence_interval=0.95)
        print("PixelCNN estimated information: ", pixel_cnn_info)
        print("PixelCNN lower bound: ", pixel_cnn_lower_bound)
        print("PixelCNN upper bound: ", pixel_cnn_upper_bound)

        model_name = 'pixelcnn_{}_patchsize_{}_seed_{}'.format(imager_type, patch_size, seed_value)
        np.save(save_dir + model_name + '_val_loss.npy', val_loss_history)
        np.save(save_dir + model_name + '_mi_estimate.npy', np.array([pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound]))
