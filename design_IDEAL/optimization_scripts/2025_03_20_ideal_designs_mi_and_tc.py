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

# Notebook for loading and evaluating pre-optimized designs for Fig. 3 and Fig. S8

# Three datasets: MNIST, Fashion MNIST, and CIFAR10
# Load pre-optimized imaging systems and evaluate MI and TC metrics
# Uses 25 lenslets for all datasets

import os
import sys
# set gpu to be pci bus id
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/')
from imaging_system import ImagingSystem, ImagingSystemProtocol

from encoding_information.models.pixel_cnn import PixelCNN
from encoding_information.models.gaussian_process import FullGaussianProcess
from encoding_information.information_estimation import *

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
import wandb
wandb.login()

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from losses import PixelCNNLoss, GaussianLoss, GaussianEntropyLoss
from optimizers import IDEALOptimizer

# import specific imaging system's modules 
from lensless_imaging_system_diagonal_covs import RMLPSFLayer, LenslessImagingSystem # Using diagonal covariances for optimization
from lensless_data_generator import LenslessDataGenerator


optimal_numbers_of_lenslets = [25, 25, 25]
dataset_names = ['mnist', 'fashion_mnist', 'cifar10']
design_type = 'random' # 'random' or 'heuristic'

for dataset_name_idx, dataset_name in enumerate(dataset_names):
    # general parameters 
    seed_value = 42
    key = jax.random.PRNGKey(seed_value)

    # dataset parameters 
    subset_fraction = 1.0
    photon_count = 100.0
    tile_rows = 3
    tile_cols = 3
    batch_size = 50

    # RMLPSFLayer parameters 
    object_size = 96 
    num_gaussian = optimal_numbers_of_lenslets[dataset_name_idx] 
    psf_size = (32, 32) 
    measurement_bias = 10.0 

    # define parameters for IDEAL optimization 
    patch_size = 16
    num_patches = 4096
    patching_strategy = 'random' 
    num_steps = 2000
    loss_type = 'pixelcnn'
    # these are pixelcnn loss-specific parameters
    refit_every = 20
    refit_patience = 10
    refit_learning_rate = 1e-3
    refit_steps_per_epoch = 100
    reinitialize_pixelcnn = True
    use_clean_images = True # default is False, determines if clean images are used for conditional entropy estimation
    gaussian_sigma = None # if none, Poisson noise is used. Otherwise Gaussian noise with standard deviation gaussian_sigma
    learnable_parameters_w_lr = {
        'psf_layer.means': 5e-3,
        'psf_layer.covs': 1e-3,
        'psf_layer.weights': 0
    }

    # wandb parameters
    use_wandb=True
    project_name='ideal_lensless'
    run_name='{}_diagonal_cov_sigma_08_{}_init_{}_lenslets_{}_patches_5e-3_mean_lr_clean_images_seed_{}'.format( 
        dataset_name, design_type, num_gaussian, num_patches, seed_value)
    log_every = 20
    validate_every = 500

    mi_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/mi_estimates/'
    model_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/models/'
    tc_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/tc_values/'
    # define the PSF layer
    psf_layer = RMLPSFLayer(object_size, num_gaussian, psf_size, measurement_bias=measurement_bias, key=key)
    # overwrite the covariances to be narrow, sigma = 0.8, variance = 0.64
    new_covariances = jnp.zeros_like(psf_layer.covs) 
    for i in range(num_gaussian):
        new_covariances = new_covariances.at[i].set(0.64)
    
    # lenslet positions matching Fig. S7b Heuristic Point Spread Functions
    heuristic_means = jnp.zeros((10, 2)) 
    heuristic_means = heuristic_means.at[0].set((0, 0)) 
    heuristic_means = heuristic_means.at[1].set((-9, -7))
    heuristic_means = heuristic_means.at[2].set((7, 5))
    heuristic_means = heuristic_means.at[3].set((-8, 8))
    heuristic_means = heuristic_means.at[4].set((5, -11))
    heuristic_means = heuristic_means.at[5].set((11, -3))
    heuristic_means = heuristic_means.at[6].set((-12, 0))
    heuristic_means = heuristic_means.at[7].set((0, 10))
    heuristic_means = heuristic_means.at[8].set((-2, -9))
    heuristic_means = heuristic_means.at[9].set((10, 10))
    
    new_means = jnp.zeros_like(psf_layer.means) 
    for i in range(num_gaussian):
        new_means = new_means.at[i].set(heuristic_means[i])
    
    psf_layer = eqx.tree_at(lambda layer: (layer.covs, layer.weights, layer.means), 
                            psf_layer, 
                            (new_covariances, psf_layer.weights, new_means))
    
    # Define the imaging system
    imaging_system = LenslessImagingSystem(psf_layer)
    # imaging_system.display_optics()

    loaded_imaging_system = eqx.tree_deserialise_leaves(model_save_dir + run_name + "_optimized_imaging_system.eqx", imaging_system)

    # MI estimation on the final design

    import sys
    sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager/') 
    from lensless_helpers import * 
    from encoding_information import extract_patches
    from encoding_information.models import PixelCNN
    from encoding_information.plot_utils import plot_samples
    from encoding_information.models import PoissonNoiseModel, AnalyticGaussianNoiseModel
    from encoding_information.image_utils import add_noise
    from encoding_information import estimate_information

    from tensorflow.keras.datasets import mnist

    # MI estimator parameters 
    bias = 10 
    patch_size = 32
    num_patches = 10000
    test_set_size = 1500 
    learning_rate = 1e-3 
    num_iters_per_epoch = 500
    patience_val = 20

    psf_image = loaded_imaging_system.psf_layer.compute_psf()
    plt.figure()
    plt.imshow(psf_image)
    plt.colorbar()

    # load dataset 
    if dataset_name == 'mnist':
        print("Loading MNIST dataset")
        (x_train, y_train), (x_test, y_test) = mnist.load_data() 
        data = np.concatenate([x_train, x_test], axis=0)
        labels = np.concatenate([y_train, y_test], axis=0)
        data = np.pad(data, ((0, 0), (2, 2), (2, 2)), 'constant')
    elif dataset_name == 'fashion_mnist':
        print("Loading Fashion MNIST dataset")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        data = np.concatenate([x_train, x_test], axis=0)
        labels = np.concatenate([y_train, y_test], axis=0)
        data = np.pad(data, ((0, 0), (2, 2), (2, 2)), 'constant')
    elif dataset_name == 'cifar10':
        print("Loading CIFAR10 dataset")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        data = np.concatenate([x_train, x_test], axis=0)
        labels = np.concatenate([y_train, y_test], axis=0)
      
    else:
        raise ValueError('Invalid dataset name')
    if len(data.shape) == 4:
        data = tf.image.rgb_to_grayscale(data).numpy()
        data = data.squeeze()
    data = data.astype(np.float32) 

    # convert to photons 
    data = data / np.mean(data) 
    data = data * photon_count 

    # make tiled data 
    random_data, random_labels = generate_random_tiled_data(data, labels, seed_value) 

    # convolve the data 
    psf_data = convolved_dataset(psf_image, random_data) 
    # add small bias to data 
    psf_data += bias 

    # make patches for training and testing splits, random patching 
    patches = extract_patches(psf_data[:-test_set_size], patch_size=patch_size, num_patches=num_patches, seed=seed_value, verbose=True)
    test_patches = extract_patches(psf_data[-test_set_size:], patch_size=patch_size, num_patches=test_set_size, seed=seed_value, verbose=True)  
    # put all the clean patches together for use in MI estimation function later
    full_clean_patches = np.concatenate([patches, test_patches])
    # add noise to both sets 
    patches_noisy = add_noise(patches, seed=seed_value)
    test_patches_noisy = add_noise(test_patches, seed=seed_value)
    # adding in clip 
    patches_noisy = np.maximum(patches_noisy, 1e-8)
    test_patches_noisy = np.maximum(test_patches_noisy, 1e-8)

    # initialize pixelcnn 
    pixel_cnn = PixelCNN() 
    # fit pixelcnn to noisy patches. 
    val_loss_history = pixel_cnn.fit(patches_noisy, seed=seed_value, learning_rate=learning_rate, do_lr_decay=False, 
                                    steps_per_epoch=num_iters_per_epoch, patience=patience_val)

    # instantiate noise model
    noise_model = PoissonNoiseModel()
    # estimate information using the fit pixelcnn and noise model, with clean data
    pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, patches_noisy, 
                                                                                        test_patches_noisy, clean_data=full_clean_patches, 
                                                                                        confidence_interval=0.95)
    print("PixelCNN estimated information: ", pixel_cnn_info)
    print("PixelCNN lower bound: ", pixel_cnn_lower_bound)
    print("PixelCNN upper bound: ", pixel_cnn_upper_bound)
    np.save(mi_save_dir + run_name + '.npy', [pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound])

    # Compute TC on measurements
    # make full dataset with noise 
    psf_data_full_noise = add_noise(psf_data, seed=seed_value) 
    # patch and add noise 
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
    np.save(tc_save_dir + run_name + '.npy', tamura_values)
