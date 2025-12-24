import sys
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/src')
sys.path.append('/home/lakabuli/workspace/LenslessInfoDesign/EncodingInformation/lensless_imager/')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import transforms
import random
from lensless_helpers import make_bead_volume


# Import utilities from recon helpers
from recon_helpers import (
    CFG, load_base_datasets, split_train_val,
    build_mosaic_batch, make_conv_kernel_from_psf, valid_conv_cpu,
    build_loaders_from_precomputed
)


### From Lensless Learning Paper (Monakhova et al. 2019) tweaked to work for extended FOV recon 
### https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28075

BN_EPS = 1e-4

# ============================
# Pads measurement to 96x96
# ============================
def pad_to_size(x, target_h, target_w):
    _, _, h, w = x.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    def forward(self, x, down_tensor):
        _, _, h, w = down_tensor.size()
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([x, down_tensor], dim=1)
        return self.decode(x)


class UNet270480(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.down1 = StackEncoder(in_ch,  24, kernel_size=3)
        self.down2 = StackEncoder(24,     64, kernel_size=3)
        self.down3 = StackEncoder(64,    128, kernel_size=3)
        self.down4 = StackEncoder(128,   256, kernel_size=3)
        self.down5 = StackEncoder(256,   512, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128,  64, kernel_size=3)
        self.up2 = StackDecoder( 64,  64,  24, kernel_size=3)
        self.up1 = StackDecoder( 24,  24,  24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_ch, kernel_size=1, bias=True)
    def forward(self, x):
        out = x
        d1, out = self.down1(out)
        d2, out = self.down2(out)
        d3, out = self.down3(out)
        d4, out = self.down4(out)
        d5, out = self.down5(out)
        out = self.center(out)
        out = self.up5(out, d5)
        out = self.up4(out, d4)
        out = self.up3(out, d3)
        out = self.up2(out, d2)
        out = self.up1(out, d1)
        out = self.classify(out)  # (N,1,96,96)
        return out

# --- Small U-Net variant (also runs at 96x96 after padding) ---
class UNet_small(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.down1 = StackEncoder(in_ch, 24, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(24, 24, kernel_size=3, padding=1))
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        d1, out = self.down1(out)
        out = self.center(out)
        out = self.up1(out, d1)
        out = self.classify(out)  # (N, out_ch, 96, 96)
        return out
    

##################
##### I added these ones adapted from above
##################

class UNet_medium(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.down1 = StackEncoder(in_ch, 24, kernel_size=3)
        self.down2 = StackEncoder(24,     64, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(64, 64, kernel_size=3, padding=1))
        self.up2 = StackDecoder( 64,  64,  24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        d1, out = self.down1(out)
        d2, out = self.down2(out)
        out = self.center(out)
        out = self.up2(out, d2)
        out = self.up1(out, d1)
        out = self.classify(out)  # (N, out_ch, 96, 96)
        return out

class UNet_medlarge(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.down1 = StackEncoder(in_ch, 24, kernel_size=3)
        self.down2 = StackEncoder(24,     64, kernel_size=3)
        self.down3 = StackEncoder(64,    128, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(128, 128, kernel_size=3, padding=1))
        self.up3 = StackDecoder(128, 128,  64, kernel_size=3)
        self.up2 = StackDecoder( 64,  64,  24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        d1, out = self.down1(out)
        d2, out = self.down2(out)
        d3, out = self.down3(out)
        out = self.center(out)
        out = self.up3(out, d3)
        out = self.up2(out, d2)
        out = self.up1(out, d1)
        out = self.classify(out)  # (N, out_ch, 96, 96)
        return out


# ============================
# Precompute datasets (meas, x96)
# ============================
def precompute_recon_datasets(
    psf_cpu,
    base_train,
    base_val,
    base_test,
    cfg=CFG,
    seed=0,
    tf_digit=None,
    normalize=False,
    add_noise=False,
    mean_val=1.0,
    only_gt_measurements=False,
):
    assert tf_digit is not None, "tf_digit must be provided"
    # 1) Build mosaics
    x96_tr, _ = build_mosaic_batch(base_train, cfg.num_train, cfg.cell_size, cfg.grid_hw, tf_digit, seed=seed)
    x96_va, _ = build_mosaic_batch(base_val,   cfg.num_val,   cfg.cell_size, cfg.grid_hw, tf_digit, seed=seed+1)
    x96_te, _ = build_mosaic_batch(base_test,  cfg.num_test,  cfg.cell_size, cfg.grid_hw, tf_digit, seed=seed+2)

    if only_gt_measurements:
        train_ds = TensorDataset(x96_tr.contiguous())
        val_ds   = TensorDataset(x96_va.contiguous())
        test_ds  = TensorDataset(x96_te.contiguous())
        return train_ds, val_ds, test_ds

    # 2) Fixed conv kernel
    psf_conv_cpu = make_conv_kernel_from_psf(psf_cpu)

    # 3) Convolve to measurements and zero pad to the 96x96 object size
    with torch.no_grad():
        meas_tr = valid_conv_cpu(x96_tr, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed)
        meas_va = valid_conv_cpu(x96_va, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed+1)
        meas_te = valid_conv_cpu(x96_te, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed+2)

        H, W = x96_tr.shape[-2], x96_tr.shape[-1] 
        meas_tr = pad_to_size(meas_tr, H, W)
        meas_va = pad_to_size(meas_va, H, W)
        meas_te = pad_to_size(meas_te, H, W)


    # 4) TensorDatasets: (measurement, target_image)
    train_ds = TensorDataset(meas_tr.contiguous(), x96_tr.contiguous())
    val_ds   = TensorDataset(meas_va.contiguous(), x96_va.contiguous())
    test_ds  = TensorDataset(meas_te.contiguous(), x96_te.contiguous())
    return train_ds, val_ds, test_ds


# version for beads 
def build_mosaic_batch_beads(base_dataset, n, dim=96, tf_digit=None, seed=0):
    """
    Return:
      x96: (n,1,96,96) float32
    """
    rng = random.Random(seed)
    H = W = dim
    x96 = torch.zeros(n, 1, H, W, dtype=torch.float32)

    # draw n indices from range (0, len(base_dataset) - 1) without replacement
    indices = rng.sample(range(len(base_dataset)), n)
    for b in range(n):
        idx = indices[b]
        img = tf_digit(base_dataset[idx]) # draw that base dataset image and transform it to tensor 
        x96[b] = img
    return x96

def precompute_recon_datasets_beads(
    psf_cpu,
    base_train,
    base_val,
    base_test,
    cfg=CFG,
    seed=0,
    tf_digit=None,
    normalize=False,
    add_noise=False,
    mean_val=1.0,
    only_gt_measurements=False,
):
    assert tf_digit is not None, "tf_digit must be provided"
    # 1) Build mosaics
    x96_tr = build_mosaic_batch_beads(base_train, cfg.num_train, cfg.grid_hw * cfg.psf_size, tf_digit, seed=seed)
    x96_va = build_mosaic_batch_beads(base_val,   cfg.num_val,   cfg.grid_hw * cfg.psf_size, tf_digit, seed=seed+1)
    x96_te = build_mosaic_batch_beads(base_test,  cfg.num_test,  cfg.grid_hw * cfg.psf_size, tf_digit, seed=seed+2)

    if only_gt_measurements:
        train_ds = TensorDataset(x96_tr.contiguous())
        val_ds   = TensorDataset(x96_va.contiguous())
        test_ds  = TensorDataset(x96_te.contiguous())
        return train_ds, val_ds, test_ds

    # 2) Fixed conv kernel
    psf_conv_cpu = make_conv_kernel_from_psf(psf_cpu)

    # 3) Convolve to measurements and zero pad to the 96x96 object size
    with torch.no_grad():
        meas_tr = valid_conv_cpu(x96_tr, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed)
        meas_va = valid_conv_cpu(x96_va, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed+1)
        meas_te = valid_conv_cpu(x96_te, psf_conv_cpu, normalize=normalize, add_noise=add_noise, mean_val=mean_val, seed=seed+2)

        H, W = x96_tr.shape[-2], x96_tr.shape[-1] 
        meas_tr = pad_to_size(meas_tr, H, W)
        meas_va = pad_to_size(meas_va, H, W)
        meas_te = pad_to_size(meas_te, H, W)


    # 4) TensorDatasets: (measurement, target_image)
    train_ds = TensorDataset(meas_tr.contiguous(), x96_tr.contiguous())
    val_ds   = TensorDataset(meas_va.contiguous(), x96_va.contiguous())
    test_ds  = TensorDataset(meas_te.contiguous(), x96_te.contiguous())
    return train_ds, val_ds, test_ds



# ============================
# Loss & metrics
# ============================
def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def mse_loss(pred, target):
    return F.mse_loss(pred, target) # use default settings


@torch.no_grad()
def psnr(pred, target, eps=1e-8):
    # assumes inputs in [0,1]; clamp to avoid negative/overflow
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = F.mse_loss(pred, target, reduction="mean")
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + eps))

@torch.no_grad()
def mse(pred, target, reduction='mean'):
    return F.mse_loss(pred, target, reduction=reduction) # default is reduction='mean'


@torch.no_grad()
def visualize_dataset_triplet(model, dataset_loader, device, pin, save_path=None, show=True):
    """
    Shows/saves a 3-panel figure: [Ground Truth 96x96 | Measurement 65x65 | Reconstruction 96x96]
    Also saves 5 example images from the dataset as a .npy file
    save_path assumed to include no suffix for the datatype.
    """
    model.eval()
    # Grab a single batch
    for meas, x_gt in dataset_loader:
        meas = meas.to(device, non_blocking=pin)
        x_gt = x_gt.to(device, non_blocking=pin)
        pred = model(meas)

        # Take the first sample
        m = meas[0, 0].detach().cpu().numpy()
        g = x_gt[0, 0].detach().cpu().numpy()
        p = pred[0, 0].detach().cpu().numpy()

        fig = plt.figure(figsize=(9, 3))
        ax1 = plt.subplot(1, 3, 1); ax1.imshow(g, cmap="gray", vmin=0, vmax=1); ax1.set_title("Ground Truth (96×96)"); ax1.axis("off")
        ax2 = plt.subplot(1, 3, 2); ax2.imshow(m, cmap="gray", vmin=0, vmax=1); ax2.set_title("Measurement (65×65)"); ax2.axis("off")
        ax3 = plt.subplot(1, 3, 3); ax3.imshow(p, cmap="gray", vmin=0, vmax=1); ax3.set_title("Reconstruction (96×96)"); ax3.axis("off")
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path + '.png', dpi=300, bbox_inches="tight")
            plt.close(fig)
            meas_examples = meas[:5, 0].detach().cpu().numpy()
            gt_examples = x_gt[:5, 0].detach().cpu().numpy()
            pred_examples = pred[:5, 0].detach().cpu().numpy()
            all_examples = np.stack([meas_examples, gt_examples, pred_examples], axis=0)
            np.save(save_path + '.npy', all_examples)
        elif show:
            plt.show()
        break


# ============================
# Training / Validation / Test
# ============================
def train_one_epoch(model, loader, opt, device, pin, loss_function):
    model.train()
    total = 0.0
    n = 0
    for meas, x_gt in loader:
        meas = meas.to(device, non_blocking=pin)
        x_gt = x_gt.to(device, non_blocking=pin)

        pred = model(meas)
        loss = loss_function(pred, x_gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = meas.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n,1)

@torch.no_grad()
def evaluate(model, loader, device, pin, loss_function):
    model.eval()
    total = 0.0
    n = 0
    psnr_sum = 0.0
    mse_sum = 0.0
    for meas, x_gt in loader:
        meas = meas.to(device, non_blocking=pin)
        x_gt = x_gt.to(device, non_blocking=pin)

        pred = model(meas)
        loss = loss_function(pred, x_gt)
        total += loss.item() * meas.size(0)
        n += meas.size(0)
        psnr_sum += psnr(pred, x_gt).item()
        mse_sum += mse(pred, x_gt).item()
        
    avg_loss = total / max(n,1)
    avg_psnr = psnr_sum / max(len(loader),1)
    avg_mse = mse_sum / max(len(loader),1)
    return avg_loss, {"psnr": avg_psnr, "mse": avg_mse}

# ============================
# Main fn for recon
# ============================
def main_recon(psf_image, psf_name, visualize=False, save_dir=None, seed_value=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device == "cuda")
    torch.backends.cudnn.benchmark = True

    if CFG.bead_sparsity is not None:
        dataset = np.zeros((50000, 96, 96))
        np.random.seed(0)
        for i in range(dataset.shape[0]):
            vol, _ = make_bead_volume(CFG.bead_sparsity, bead_width_scale=1, numx=96, numy=96)
            dataset[i] = vol
        base_train_full = dataset[:40000]
        base_test = dataset[40000:]
        tf_digit = transforms.Compose([transforms.ToTensor()])
        base_train, base_val = split_train_val(base_train_full, val_fraction=0.1, seed=0)
        psf_cpu = torch.tensor(psf_image, dtype=torch.float32, device="cpu").view(1, 1, CFG.psf_size, CFG.psf_size)
        train_ds, val_ds, test_ds = precompute_recon_datasets_beads(
            psf_cpu, base_train, base_val, base_test, cfg=CFG, seed=0,
            tf_digit=tf_digit, normalize=CFG.normalize, add_noise=CFG.add_noise, mean_val=CFG.mean_val
        )

    else:
        base_train_full, base_test, tf_digit = load_base_datasets(CFG.dataset, CFG.cell_size)
        base_train, base_val = split_train_val(base_train_full, val_fraction=0.1, seed=0)

        psf_cpu = torch.tensor(psf_image, dtype=torch.float32, device="cpu").view(1, 1, CFG.psf_size, CFG.psf_size)

        train_ds, val_ds, test_ds = precompute_recon_datasets(
            psf_cpu, base_train, base_val, base_test, cfg=CFG, seed=0,
            tf_digit=tf_digit, normalize=CFG.normalize, add_noise=CFG.add_noise, mean_val=CFG.mean_val
        )

    train_loader, val_loader, test_loader = build_loaders_from_precomputed(train_ds, val_ds, test_ds, cfg=CFG, device=device)

    if CFG.model_type == 'small':
        model = UNet_small(in_ch=1, out_ch=1).to(device)
    elif CFG.model_type == 'medium':
        model = UNet_medium(in_ch=1, out_ch=1).to(device)
    elif CFG.model_type == 'medlarge':
        model = UNet_medlarge(in_ch=1, out_ch=1).to(device)
    elif CFG.model_type == 'large':
        model = UNet270480(in_ch=1, out_ch=1).to(device)
    else:
        raise ValueError(f"Unknown model type: {CFG.model_type}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs)

    if CFG.loss_type == 'l1':
        loss_function = l1_loss
    elif CFG.loss_type == 'mse':
        loss_function = mse_loss
    else:
        raise ValueError(f"Unknown loss type: {CFG.loss_type}")

    best_val = float("inf")
    best_state = None
    best_epoch = None
    epochs_no_improve = 0

    train_losses, val_losses, val_psnrs, val_mses = [], [], [], []

    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, pin, loss_function)
        val_loss, val_metrics = evaluate(model, val_loader, device, pin, loss_function)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_metrics["psnr"])
        val_mses.append(val_metrics["mse"])

        if (epoch % 10) == 0 or epoch == 1 or epoch == CFG.epochs:
            print(f"[{epoch:03d}/{CFG.epochs}] train L1={tr_loss:.4f} | val L1={val_loss:.4f} | val PSNR={val_metrics['psnr']:.2f} dB | val MSE={val_metrics['mse']:.4f}")
            if visualize:
                visualize_dataset_triplet(model, val_loader, device, pin, save_path=None, show=True)


        # Early stopping on val loss
        if val_loss < best_val - CFG.es_min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG.es_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {CFG.es_patience} epochs).")
                break

        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, device, pin, loss_function)

    logs_array = np.asarray([train_losses, val_losses, val_psnrs, val_mses], dtype=np.float32)
    bepoch = -1 if best_epoch is None else best_epoch
    final_vector = np.asarray([test_loss, test_metrics["psnr"], test_metrics["mse"], float(bepoch)], dtype=np.float32) 

    if save_dir is not None:
        save_path = f"{save_dir}{CFG.dataset}_{psf_name}_psf_{CFG.model_type}_model_{CFG.lr}_lr_{CFG.batch_size}_batch_{CFG.epochs}_epoch_max_{seed_value}_recon_images"
        visualize_dataset_triplet(model, test_loader, device, pin, save_path=save_path, show=False)
        np.save(f"{save_dir}{CFG.dataset}_{psf_name}_psf_{CFG.model_type}_model_{CFG.lr}_lr_{CFG.batch_size}_batch_{CFG.epochs}_epoch_max_{seed_value}_recon_logs.npy", logs_array)
        np.save(f"{save_dir}{CFG.dataset}_{psf_name}_psf_{CFG.model_type}_model_{CFG.lr}_lr_{CFG.batch_size}_batch_{CFG.epochs}_epoch_max_{seed_value}_final_metrics.npy", final_vector)


    print("\n=== FINAL TEST METRICS (best-val checkpoint) ===")
    print(f"best epoch={bepoch} | best val loss={best_val:.4f}")
    print(f"test L1={test_loss:.4f} | test PSNR={test_metrics['psnr']:.2f} dB")

    return logs_array, final_vector

# Default configurations
class CFG:
    dataset     = 'mnist'     # 'mnist', 'fashion_mnist', or 'CIFAR10'
    normalize   = False        
    add_noise   = False        # if normalize and add noise and per_item, then it'll do per-item normalization
    per_item    = False        # if normalize, add_noise, and not per_item, then it'll do batch normalization
    mean_val    = 1.0          # mean photon count 
    cell_size   = 32           # each MNIST tile -> 32x32
    grid_hw     = 3            # 3x3 grid -> 96x96 canvas
    psf_size    = 32
    num_train   = 20000        # how many precomputed training mosaics
    num_val     = 2000         # how many precomputed validation mosaics
    num_test    = 2000         # how many precomputed test mosaics
    bead_sparsity = None       # what the bead sparsity is, if not none, should be 0.02.

    model_type = 'medium'
    loss_type = 'mse'

    batch_size = 64
    num_workers = 0
    epochs = 100
    es_patience = 10
    es_min_delta = 1e-5
    lr = 2e-3
    weight_decay = 1e-4


