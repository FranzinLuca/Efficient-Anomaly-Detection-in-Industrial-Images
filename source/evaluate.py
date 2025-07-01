"""
Evaluate the model on the validation/test dataset.
This function computes the image-level and pixel-level AUROC, precision, and F1 score.
"""

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the given dataloader.
    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader containing the validation/test dataset.
        device: Device to run the evaluation on (CPU or GPU).
    Returns: 
        image-level AUROC.
        pixel-level AUROC.
        precision.
        F1 score.
    """
    model.eval()
    
    all_img_scores = []
    all_img_labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    with torch.no_grad():
        for images, labels, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            original_features, reconstructed_features = model(images)
            anomaly_map = model.get_anomaly_map(original_features, reconstructed_features)
            
            # Apply Gaussian blur to the anomaly map
            anomaly_map = gaussian_blur(anomaly_map, kernel_size=7, sigma=1.5)
            
            # Upsample anomaly map to image size
            anomaly_map_resized = F.interpolate(anomaly_map, size=images.shape[2:], mode='bilinear', align_corners=False)
            
            # Image-level score
            img_scores = torch.max(anomaly_map_resized.view(images.size(0), -1), dim=1)[0]
            all_img_scores.extend(img_scores.cpu().numpy())
            all_img_labels.extend(labels.numpy())

            for i in range(images.size(0)):
                all_pixel_scores.extend(anomaly_map_resized[i].cpu().numpy().flatten())
                all_pixel_labels.extend(masks[i].cpu().numpy().flatten())

            
    # Calculate AUROC scores
    image_auroc = roc_auc_score(all_img_labels, all_img_scores)
    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
  
    # Find the optimal threshold from the ROC curve
    fpr, tpr, thresholds = roc_curve(all_img_labels, all_img_scores)
    # Youden's J statistic to find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Generate binary predictions based on the threshold
    predicted_labels = [1 if score >= optimal_threshold else 0 for score in all_img_scores]
    
    # Calculate precision and F1 score
    accuracy = accuracy_score(all_img_labels, predicted_labels)
    f1 = f1_score(all_img_labels, predicted_labels)
    
    return image_auroc, pixel_auroc, accuracy, f1