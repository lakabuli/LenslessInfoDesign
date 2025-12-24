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

# Script to run MI estimates on datasets with random PSF initialization
# Provides baseline MI estimates for comparison with optimized designs
# Supports MNIST, Fashion MNIST, and CIFAR10 datasets

import os
import sys
# set gpu to be pci bus id
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# set gpu memory usage and turnoff pre-allocated memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import optax
import equinox as eqx
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
test = jnp.zeros((10,10,10))
import jax.random as random

sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src/')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/ideal/')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager/')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/')
from imaging_system import ImagingSystem, ImagingSystemProtocol

from encoding_information.models.pixel_cnn import PixelCNN
from encoding_information.models.gaussian_process import FullGaussianProcess
from encoding_information.information_estimation import *

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist

from lensless_helpers import *
from encoding_information import extract_patches
from encoding_information.models import PoissonNoiseModel
from encoding_information.image_utils import add_noise
from encoding_information import estimate_information

# Load random PSF initialization
random_init_psf_path = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/random_psf_init_seed_42.npy'
random_init_psf = np.load(random_init_psf_path)
print(f"Random PSF sum: {np.sum(random_init_psf)}")

# Set seed values for reproducibility
seed_values_full = [4, 42, 31, 50, 77]  # 5 arbitrary seed values

# Set photon properties
bias = 10  # in photons
mean_photon_count_list = [100]  # running just the single photon count case for comparison to IDEAL as well

# Set eligible PSFs
psf_patterns = [random_init_psf]
psf_names = ['random_init_ideal']

# MI estimator parameters
patch_size = 32
num_patches = 10000
val_set_size = 1000
test_set_size = 1500
num_samples = 8
learning_rate = 1e-3
num_iters_per_epoch = 500
patience_val = 20

dataset_name = 'cifar10'

# Save directories
mi_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/tradeoff_analysis/mi_estimates/{}_mi_estimates/'.format(dataset_name)

for mean_photon_count in mean_photon_count_list:
    for psf_index, psf_pattern in enumerate(psf_patterns):
        val_loss_log = []
        mi_estimates = []
        lower_bounds = []
        upper_bounds = []
        for seed_value in seed_values_full:
            # Load dataset
            if dataset_name == 'cifar10':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            elif dataset_name == 'mnist':
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
            elif dataset_name == 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            else:
                print(f"Invalid dataset name: {dataset_name}. Please use 'mnist', 'fashion_mnist', or 'cifar10'")
                break
            
            data = np.concatenate((x_train, x_test), axis=0)
            # Convert data to grayscale if it's not already
            if len(data.shape) == 4:
                data = tf.image.rgb_to_grayscale(data).numpy()
                data = data.squeeze()
            labels = np.concatenate((y_train, y_test), axis=0)
            # Zero pad images to be 32 x 32 size - not for CIFAR10
            if dataset_name in ['mnist', 'fashion_mnist']:
                data = np.pad(data, ((0, 0), (2, 2), (2, 2)), 'constant')
            data = data.astype(np.float32)
            # Convert data to photons with mean value of photon_count
            data = data / np.mean(data)
            data = data * mean_photon_count
            # Make tiled data
            random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)
            psf_data = convolved_dataset(psf_pattern, random_data)
            # Add small bias to data
            psf_data += bias
            # Make patches for training and testing splits, random patching
            patches = extract_patches(psf_data[:-test_set_size], patch_size=patch_size, num_patches=num_patches, seed=seed_value, verbose=True)
            test_patches = extract_patches(psf_data[-test_set_size:], patch_size=patch_size, num_patches=test_set_size, seed=seed_value, verbose=True)
            # Put all the clean patches together for use in MI estimation function later
            full_clean_patches = np.concatenate([patches, test_patches])
            # Add noise to both sets
            patches_noisy = add_noise(patches, seed=seed_value)
            test_patches_noisy = add_noise(test_patches, seed=seed_value)

            # Initialize PixelCNN
            pixel_cnn = PixelCNN()
            # Fit pixelcnn to noisy patches. Defaults to 10% val samples which will be 1k as desired.
            # Use a small learning rate for stability and include seeding
            val_loss_history = pixel_cnn.fit(patches_noisy, seed=seed_value, learning_rate=learning_rate, do_lr_decay=False, 
                                            steps_per_epoch=num_iters_per_epoch, patience=patience_val)
            # Instantiate noise model
            noise_model = PoissonNoiseModel()
            # Estimate information using the fit pixelcnn and noise model, with clean data
            pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, patches_noisy,
                                                                                                test_patches_noisy, clean_data=full_clean_patches,
                                                                                                confidence_interval=0.95)
            print(f"PixelCNN estimated information: {pixel_cnn_info}")
            print(f"PixelCNN lower bound: {pixel_cnn_lower_bound}")
            print(f"PixelCNN upper bound: {pixel_cnn_upper_bound}")
            # Append results to lists
            val_loss_log.append(val_loss_history)
            mi_estimates.append(pixel_cnn_info)
            lower_bounds.append(pixel_cnn_lower_bound)
            upper_bounds.append(pixel_cnn_upper_bound)
        
        # Save results after all seeds are processed
        save_filename = f'{dataset_name}_pixelcnn_mi_estimate_{mean_photon_count}_photons_{psf_names[psf_index]}_psf_{learning_rate}_lr_{patience_val}_patience_{num_iters_per_epoch}_steps_per_epoch'
        np.save(mi_save_dir + save_filename + '.npy', np.array([mi_estimates, lower_bounds, upper_bounds]))
        np.save(mi_save_dir + save_filename.replace('mi_estimate', 'val_loss') + '.npy', np.array(val_loss_log, dtype=object))
