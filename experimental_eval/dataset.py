# Adapted from Vasilisa Ponomarenko

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from natsort import natsorted
import matplotlib.pyplot as plt
from skimage.transform import resize

class ClaraMirflickr(Dataset):
    def __init__(self, gt_dir, meas_dir, is_rml=True, train=True):
        super().__init__()
        self.gt_dir = gt_dir
        self.meas_dir = meas_dir
        self.is_rml = is_rml
        if self.is_rml:
            self.data_dir = os.path.join(meas_dir, "rml")
        else:
            self.data_dir = os.path.join(meas_dir, "diffusercam")
        self.target_dir = gt_dir

        full_data_list = natsorted(os.listdir(self.data_dir))
        full_target_list = natsorted(os.listdir(self.target_dir))

        if train:
            self.data_list = full_data_list[1000:]
            self.target_list = full_target_list[1000:]
        else:
            self.data_list = full_data_list[:1000]
            self.target_list = full_target_list[:1000]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image = plt.imread(os.path.join(self.data_dir, self.data_list[index]))
        target = plt.imread(os.path.join(self.target_dir, self.target_list[index]))
        img_name = self.data_list[index][:-5]  # get image name without the extension

        # remove alpha channel if present
        if image.shape[-1] == 4:
            image = image[..., :-1]
        if target.shape[-1] == 4:
            target = target[..., :-1]

        # convert images to 0 to 1 range and convert to float
        image = (image / 255.0).astype(np.float32)
        target = (target / 255.0).astype(np.float32)

        # resize measurement (8x downsampling)
        image = resize(image, (150, 240), anti_aliasing=True).astype(np.float32)

        # verify sizing
        assert image.shape == target.shape

        # clip to valid range
        image = np.clip(image, 0, 1)
        target = np.clip(target, 0, 1)

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        # move channels to the front
        image = torch.moveaxis(image, -1, 0)
        target = torch.moveaxis(target, -1, 0)

        return image, target, img_name
    

def get_loader(dataset, batch_size, num_workers, is_rml=True, gt_dir='/home/lakabuli/cosmos_drive/', meas_dir='/home/lakabuli/cosmos_drive/0-25k/'):
    if dataset == "ClaraMirflickr":
        if is_rml:
            gt_dir = '/home/lakabuli/cosmos_drive/undistorted_GT2RML/'
        else:
            gt_dir = '/home/lakabuli/cosmos_drive/undistorted_GT2DC/'

        trainset = ClaraMirflickr(gt_dir, meas_dir, is_rml, train=True)
        testset = ClaraMirflickr(gt_dir, meas_dir, is_rml, train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
    


def load_image_pair_rml(img_number):
    ground_truth = plt.imread(f"/home/lakabuli/cosmos_drive/undistorted_GT2RML/warped_undistorted_img_{img_number}_cam_2.tiff")[:, :, :-1]
    rml = plt.imread(f"/home/lakabuli/cosmos_drive/0-25k/rml/img_{img_number}_cam_1.tiff")
    if rml.shape[-1] == 4:
        rml = rml[:, :, :-1]

    ground_truth = (ground_truth / 255.0).astype(np.float32)
    rml = (rml / 255.0).astype(np.float32)
    rml = resize(rml, (150, 240), anti_aliasing=True).astype(np.float32)

    ground_truth = np.clip(ground_truth, 0, 1)
    rml = np.clip(rml, 0, 1)

    rml = torch.from_numpy(rml)
    rml = torch.moveaxis(rml, -1, 0)
    rml = rml.unsqueeze(0)

    return rml, ground_truth

def load_image_pair_diffuser(img_number):
    ground_truth = plt.imread(f"/home/lakabuli/cosmos_drive/undistorted_GT2DC/warped_undistorted_img_{img_number}_cam_2.tiff")[:, :, :-1]
    dc = plt.imread(f"/home/lakabuli/cosmos_drive/0-25k/diffusercam/img_{img_number}_cam_0.tiff")
    if dc.shape[-1] == 4:
        dc = dc[:, :, :-1]

    ground_truth = (ground_truth / 255.0).astype(np.float32)
    dc = (dc / 255.0).astype(np.float32)
    dc = resize(dc, (150, 240), anti_aliasing=True).astype(np.float32)

    ground_truth = np.clip(ground_truth, 0, 1)
    dc = np.clip(dc, 0, 1)

    dc = torch.from_numpy(dc)
    dc = torch.moveaxis(dc, -1, 0)
    dc = dc.unsqueeze(0)

    return dc, ground_truth