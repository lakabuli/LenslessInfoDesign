import sys
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src')

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

from encoding_information.image_utils import add_noise as add_noise_fn

# ==============================================
# Config
# ==============================================
class CFG:
    dataset = 'mnist'
    normalize = False
    add_noise = False
    mean_val = 1.0
    cell_size = 32
    grid_hw = 3
    psf_size = 32
    num_train = 20000
    num_val = 2000
    num_test = 2000

    batch_size = 64
    num_workers = 0
    epochs = 50
    es_patience = 10
    es_min_delta = 1e-3
    lr = 2e-3
    weight_decay = 1e-4
    center_only = False
    head_name = 'basic'
    model = 'basic'
    resnet_depth = 20
    resnet_shortcut = "A"


# ==============================================
# PSF & convolution helpers (CPU)
# ==============================================

def make_conv_kernel_from_psf(psf: torch.Tensor) -> torch.Tensor:
    """Flip H,W so conv2d is TRUE convolution (not correlation)."""
    return torch.flip(psf, dims=(-1, -2)).contiguous()


def valid_conv_cpu(
    x: torch.Tensor,
    psf_conv: torch.Tensor,
    normalize: bool = False,
    add_noise: bool = False,
    per_item: bool = True,
    mean_val: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    CPU-only valid conv with optional:
      1) Pre-conv mean scaling (divide by per-sample mean, multiply by mean_val)
      2) Convolution
      3) Add noise (via add_noise_fn(images, seed))
      4) Optional post-conv scale by max value so max=1 (no min subtraction)

    Args:
        x:         (B,1,96,96) input mosaics (float32 expected)
        psf_conv:  (1,1,32,32) already flipped PSF kernel for true convolution
        mean_val:  Target mean after pre-scale (default 1.0)
        seed:      Seed passed to add_noise_fn
        max_scale: If True, divide each example by its own max after noise

    Returns:
        (B,1,65,65) tensor
    """
    if not normalize:
        return F.conv2d(x, psf_conv, padding=0)
    if normalize and not add_noise:
        convolved = F.conv2d(x, psf_conv, padding=0)
        return convolved / torch.max(convolved)
    elif normalize and add_noise and per_item: # this is default, used in classification. Since doing per-batch normalization it makes sense to keep everything per-item, since max's across batches could vary.
        # 1) Pre-conv mean scaling
        means = x.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        x_scaled = x / means * float(mean_val)

        # 2) Convolution
        y = F.conv2d(x_scaled, psf_conv, padding=0)

        # 3) Add noise
        y_np = y.detach().cpu().numpy()
        y_np = add_noise_fn(y_np, seed=seed)
        y_np = np.array(y_np, copy=True)
        y = torch.tensor(y_np, dtype=torch.float32, device=y.device)

        # 4) Post-conv max scaling (no min subtraction), with per-item max norm
        max_vals = y.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        y = y / max_vals

        return y.contiguous()

    elif normalize and add_noise and not per_item:
        # 1) Pre-conv mean scaling, by the whole batch
        means = x.mean().clamp_min(1e-8)
        x_scaled = x / means * float(mean_val)
        max_object_val = x_scaled.amax().clamp_min(1e-8)

        # 2) Convolution
        y = F.conv2d(x_scaled, psf_conv, padding=0)

        # 3) Add noise
        y_np = y.detach().cpu().numpy()
        y_np = add_noise_fn(y_np, seed=seed)
        y_np = np.array(y_np, copy=True)
        y = torch.tensor(y_np, dtype=torch.float32, device=y.device)

        # 4) Post-conv max scaling (no min subtraction) based on the maximum object value pre-convolution in photon counts -> scales everything down to 0-1 based on object max.
        y = y / max_object_val

        return y.contiguous()

    else:
        raise ValueError("Invalid combination of arguments.")


# ==============================================
# Mosaic generator (CPU) and precompute dataset
# ==============================================

def load_base_datasets(dataset_name, cell_size, data_root="./data"):
    if dataset_name == "mnist":
        base_train_full = datasets.MNIST(root=data_root, train=True, download=False, transform=None)
        base_test = datasets.MNIST(root=data_root, train=False, download=False, transform=None)
        tf_digit = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(padding=(cell_size - 28) // 2, fill=0),
        ])

    elif dataset_name == "fashion_mnist":
        base_train_full = datasets.FashionMNIST(root=data_root, train=True, download=False, transform=None)
        base_test = datasets.FashionMNIST(root=data_root, train=False, download=False, transform=None)
        tf_digit = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(padding=(cell_size - 28) // 2, fill=0),
        ])

    elif dataset_name == "cifar10":
        base_train_full = datasets.CIFAR10(root=data_root, train=True, download=False, transform=None)
        base_test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=None)
        tf_digit = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((cell_size, cell_size)),
            transforms.ToTensor(),
        ])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return base_train_full, base_test, tf_digit


def split_train_val(base_train_full, val_fraction=0.1, seed=0):
    rng = random.Random(seed)
    idx_all = list(range(len(base_train_full)))
    rng.shuffle(idx_all)
    n_val = int(len(idx_all) * val_fraction)
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]
    return Subset(base_train_full, train_idx), Subset(base_train_full, val_idx)


def build_mosaic_batch(base_dataset, n, cell=32, grid=3, tf_digit=None, seed=0):
    """
    Return:
      x96: (n,1,96,96) float32
      y_grid: (n,3,3) long
    """
    rng = random.Random(seed)
    H = W = cell * grid
    x96 = torch.zeros(n, 1, H, W, dtype=torch.float32)
    y_grid = torch.empty(n, grid, grid, dtype=torch.long)

    # We'll draw random digits from base_dataset for each cell
    for b in range(n):
        canvas = torch.zeros(1, H, W, dtype=torch.float32)
        labels = torch.empty(grid, grid, dtype=torch.long)
        for i in range(grid):
            for j in range(grid):
                idx = rng.randint(0, len(base_dataset) - 1)
                img, lab = base_dataset[idx]  # PIL, int
                if tf_digit is not None:
                    img = tf_digit(img)        # (1,cell,cell) float in [0,1]
                else:
                    # Fallback: MNIST default ToTensor() then resize
                    img = transforms.ToTensor()(img)
                    img = transforms.Pad(( (cell - img.shape[1]) // 2, ) * 4, fill=0)(img)

                y0, x0 = i * cell, j * cell
                canvas[:, y0:y0+cell, x0:x0+cell] = img
                labels[i, j] = lab
        x96[b] = canvas
        y_grid[b] = labels
    return x96, y_grid


def build_loaders_from_precomputed(train_ds, val_ds, test_ds, cfg=CFG, device="cpu"):
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
