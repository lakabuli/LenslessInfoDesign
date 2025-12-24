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

# Notebook for optimization with full covariance matrices for Supplement S5.3

# Optimize lensless imaging systems on MNIST with full covariance matrices (not diagonal)
# Uses random initialization for lenslet positions
# The stock initialization is seed 42, 100 photon count, 25 gaussians, patch size 16, 4096 patches, 2000 steps. 
# Learning rates are 3e-2, 1e-3, and 1e-4 respectively for means, covariances, and weights.
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
from lensless_imaging_system import RMLPSFLayer, LenslessImagingSystem # Using full degrees of freedom in covariances for optimization
from lensless_data_generator import LenslessDataGenerator


for seed_value in [42]: 
    # general parameters 
    key = jax.random.PRNGKey(seed_value)

    # dataset parameters 
    subset_fraction = 1.0
    photon_count = 100.0
    tile_rows = 3
    tile_cols = 3
    batch_size = 50
    dataset_name = 'mnist'

    # RMLPSFLayer parameters 
    object_size = 96 
    num_gaussian = 25 
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
        'psf_layer.means': 3e-2,
        'psf_layer.covs': 1e-3,
        'psf_layer.weights': 1e-4
    }

    # wandb parameters
    use_wandb=True
    project_name='ideal_lensless'
    run_name='{}_full_cov_sigma_08_random_init_{}_lenslets_{}_patches_3e-2_mean_1e-4_weight_lr_clean_images_seed_{}'.format( 
        dataset_name, num_gaussian, num_patches, seed_value)
    log_every = 20
    validate_every = 500

    mi_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/mi_estimates/'
    model_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/models/'
    tc_save_dir = '/home/lakabuli/workspace/LenslessInfoDesign/design_IDEAL/data/tc_values/'

    # define the PSF layer
    psf_layer = RMLPSFLayer(object_size, num_gaussian, psf_size, measurement_bias=measurement_bias, key=key)

    # Define the imaging system
    imaging_system = LenslessImagingSystem(psf_layer)
    # imaging_system.display_optics()

    # Create wandb config dictionary with grouped parameters
    wandb_config = {
        'general': {
            'seed_value': seed_value,
        },
        
        'dataset': {
            'subset_fraction': subset_fraction,
            'photon_count': photon_count,
            'tile_rows': tile_rows,
            'tile_cols': tile_cols,
            'batch_size': batch_size,
            'dataset_name': dataset_name,
        },
        
        'psf_layer': {
            'object_size': object_size,
            'num_gaussian': num_gaussian,
            'psf_size': psf_size,
            'measurement_bias': measurement_bias,
        },
        
        'optimization': {
            'patch_size': patch_size,
            'num_patches': num_patches,
            'patching_strategy': patching_strategy,
            'num_steps': num_steps,
            'loss_type': loss_type,
            'refit_every': refit_every,
            'refit_patience': refit_patience,
            'refit_learning_rate': refit_learning_rate,
            'refit_steps_per_epoch': refit_steps_per_epoch,
            'reinitialize_pixelcnn': reinitialize_pixelcnn,
            'gaussian_sigma': gaussian_sigma,
            'psf_layer.means': learnable_parameters_w_lr['psf_layer.means'],
            'psf_layer.covs': learnable_parameters_w_lr['psf_layer.covs'],
            'psf_layer.weights': learnable_parameters_w_lr['psf_layer.weights']

        },
        
        'logging': {
            'use_wandb': use_wandb,
            'project_name': project_name,
            'run_name': run_name,
            'log_every': log_every,
            'validate_every': validate_every
        }
    }

    # Create a Data Generator 
    data_generator = LenslessDataGenerator(photon_count, subset_fraction=subset_fraction, seed=seed_value) # data loaded is auto-scaled to photon count.

    # Load CIFAR10 data 
    if dataset_name == 'cifar10':
        x_train, x_test = data_generator.load_cifar10_data()
    elif dataset_name == 'mnist':
        x_train, x_test = data_generator.load_mnist_data()
    elif dataset_name == 'fashion_mnist':
        x_train, x_test = data_generator.load_fashion_mnist_data()
    elif dataset_name == 'mnist_lb': 
        x_train, x_test = data_generator.load_mnist_lb_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # create training dataset with 3x3 tiled images 
    train_dataset = data_generator.create_dataset(
        x_train, 
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        batch_size=batch_size   
    )
    test_dataset = data_generator.create_dataset(
        x_test, 
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        batch_size=batch_size
    )

    # Define the Loss Function

    if loss_type == 'pixelcnn':
        loss_fn = PixelCNNLoss(refit_every=refit_every, refit_patience=refit_patience, 
                                refit_learning_rate=refit_learning_rate, refit_steps_per_epoch=refit_steps_per_epoch, 
                                reinitialize_pixelcnn=reinitialize_pixelcnn, use_clean_images=use_clean_images)
    elif loss_type == 'gaussian_entropy':
        loss_fn = GaussianEntropyLoss()
    elif loss_type == 'gaussian':
        loss_fn = GaussianLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    # Create the Optimizer

    ideal_optimizer = IDEALOptimizer(
        imaging_system, 
        learnable_parameters_w_lr,
        loss_fn,
        patch_size = patch_size,
        num_patches= num_patches,
        patching_strategy=patching_strategy,
        gaussian_sigma=gaussian_sigma,
        use_wandb=use_wandb,
        project_name=project_name,
        run_name=run_name,
        wandb_config=wandb_config
    )

    # Optimize!!!!!
    optimized_imaging_system = ideal_optimizer.optimize(
        train_dataset,
        num_steps,
        log_every=log_every,
        validate_every=validate_every
    )

    # Save the optimized imaging system
    eqx.tree_serialise_leaves(model_save_dir + run_name + "_optimized_imaging_system.eqx", ideal_optimizer.imaging_system)

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

    imaging_system = ideal_optimizer.imaging_system
    psf_image = imaging_system.psf_layer.compute_psf()
    plt.figure()
    plt.imshow(psf_image)
    plt.colorbar()

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