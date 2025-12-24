import os
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import sys
sys.path.append('/home/lakabuli/workspace/EncodingInformation/lensless_imager')
from lensless_helpers import make_bead_volume

class BeadDataGenerator:
    """A data generator class for creating imaging datasets from beads with a desired sparsity and bead width.
    
    This class handles loading the image dataset, converting it to photon counts, and tiling images for training."""

    def __init__(self, mean_photon_count, nonzero_pixel_fraction, bead_width_scale=1.0, subset_fraction=1.0,
                 num_x=96, num_y=96, num_bead_imgs=60000, num_test_imgs=10000, seed=42):
        """Initialize the data generator."""
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.mean_photon_count = mean_photon_count
        self.nonzero_pixel_fraction = nonzero_pixel_fraction  # this is basically the sparsity of the sample, lower numbers are more sparse
        self.bead_width_scale = bead_width_scale
        self.num_x = num_x
        self.num_y = num_y
        self.num_bead_imgs = num_bead_imgs
        self.num_test_imgs = num_test_imgs

    def load_bead_data(self):
        """Load and preprocess beads that are 96x96 pixels in size.
        
        Returns:
            tuple: (x_train, x_test) converted to photon counts and subset of bead data with specific bead sparsity and bead width
        """
        dataset = np.zeros((self.num_bead_imgs, self.num_y, self.num_x))
        num_points_list = []
        for i in range(dataset.shape[0]):
            vol, num_points = make_bead_volume(self.nonzero_pixel_fraction, bead_width_scale=self.bead_width_scale, numx=self.num_x, numy=self.num_y)
            dataset[i] = vol
            num_points_list.append(num_points)
        assert(all(x == num_points_list[0] for x in num_points_list))
        num_points_dataset = num_points_list[0]

        dataset = dataset.astype('float32')
        dataset_photons = dataset / jnp.mean(dataset)
        dataset_photons = dataset_photons * self.mean_photon_count
        x_train = dataset_photons[:-self.num_test_imgs]
        x_test = dataset_photons[-self.num_test_imgs:]

        # take subset if specified
        if self.subset_fraction < 1.0:
            train_size = int(len(x_train) * self.subset_fraction)
            test_size = int(len(x_test) * self.subset_fraction)
            x_train = x_train[:train_size]
            x_test = x_test[:test_size]

        return x_train, x_test, num_points_dataset

    def create_dataset(self, x_data, batch_size=32):
        """Creates a TensorFlow dataset that makes bead images.
        
        Args:
            x_data (np.ndarray): Array of grayscale images with shape (N, H, W)
            batch_size: Number of images per batch
        
        Returns:
            tf.data.Dataset: Dataset of bead images with shape (N, H, W)
        """

        dataset = tf.data.Dataset.from_tensor_slices(x_data)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
