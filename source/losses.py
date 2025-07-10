"""
Loss functions for the model.
"""

import torch.nn.functional as F
import piq
import config

def mse_loss(reconstructed_features, original_features):
    """MSE loss"""
    return F.mse_loss(reconstructed_features, original_features)
def denormalize(img):
    """Denormalize the image"""
    return (img*0.5)+0.5
def ANOViT_loss(reconstructed_features, original_features, lambda_recon=config.LAMBDA_RECON):
    """SSIM loss"""
    loss_ssim= 1-piq.ssim(denormalize(reconstructed_features),denormalize(original_features) ,data_range=1.0,reduction='mean')
    loss_mse = mse_loss(reconstructed_features, original_features)
    loss_total = lambda_recon*loss_mse+(1-lambda_recon)*loss_ssim
    return loss_total