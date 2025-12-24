# from Vasilisa Ponomarenko
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Combining perceptual evaluation metric with MSE loss to create a custom loss function
class MSE_Perceptual_Loss(nn.Module):
    def __init__(self, device, alpha=0.5, is_lpips=False):
        super().__init__()
        self.alpha = alpha
        self.is_lpips = is_lpips

        self.mse_metric = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)   # normalize=True expects inputs in range [0, 1] instead of [-1, 1]

    def forward(self, input, target):      
        mse_val = self.mse_metric(input, target)
        perceptual_loss = 0
        
        if self.is_lpips:
            # Need to normalize image to [0, 1] range so that LPIPs works
            input_min = torch.min(input)
            normalized_input = (input - input_min)/ (torch.max(input) - input_min)
            perceptual_loss = self.lpips_metric(normalized_input, target)
        else:
            ssim_val = self.ssim_metric(input, target)
            perceptual_loss = 1 - ssim_val   # Want to minimize the loss, but a 1 for ssim is good so need to take the complement.

        total_loss = mse_val * (1 - self.alpha) + perceptual_loss * self.alpha
        return total_loss 