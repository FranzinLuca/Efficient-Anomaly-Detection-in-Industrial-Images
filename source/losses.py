"""
Loss functions for the model.
"""

import torch.nn.functional as F

def mse_loss(reconstructed_features, original_features):
    """MSE loss"""
    return F.mse_loss(reconstructed_features, original_features)