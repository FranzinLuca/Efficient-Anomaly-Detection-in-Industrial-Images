"""
Evaluate the model on the validation/test dataset.
This function computes the image-level and pixel-level AUROC, precision, and F1 score.
"""

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, average_precision_score 
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
import config

def evaluate_model(model, dataloader):
    """
    Evaluate the model on the given dataloader.
    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader containing the validation/test dataset.
        device: Device to run the evaluation on (CPU or GPU).
    Returns: 
        image-level AUROC.
        pixel-level AUROC.
        accuracy.
        F1 score.
    """
    device = config.DEVICE
    model.eval()
    
    all_img_scores = []
    all_img_labels = []
    all_pixel_scores = []
    all_pixel_labels = []
    per_image_pixel_aupr = []

    with torch.no_grad():
        for images, labels, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            original_features, reconstructed_features = model(images)
            anomaly_map = model.get_anomaly_map(original_features, reconstructed_features)
            
            # Upsample anomaly map to image size
            anomaly_map_resized = F.interpolate(anomaly_map, size=images.shape[2:], mode='bilinear', align_corners=False)
            
            # Apply Gaussian blur to the anomaly map
            anomaly_map_blurred = gaussian_blur(anomaly_map_resized, kernel_size=config.KERNEL_SIZE, sigma=config.SIGMA)
            
            B, _, H, W = anomaly_map_blurred.shape
            k = int(H * W * 0.01) # Top 1% of pixels
            if k == 0: k = 1 # Ensure k is at least 1
            
            anomaly_scores_flat = anomaly_map_blurred.view(B, -1)
            top_k_scores, _ = torch.topk(anomaly_scores_flat, k, dim=1)
            img_scores = torch.mean(top_k_scores, dim=1)
            
            all_img_scores.extend(img_scores.cpu().numpy())
            all_img_labels.extend(labels.numpy())
            
            all_pixel_scores.extend(anomaly_map_blurred.cpu().numpy().flatten())
            all_pixel_labels.extend(masks.cpu().numpy().flatten())
            
            for i in range(images.size(0)):
                # Check if the image is anomalous (label == 1)
                if labels[i] == 1:
                    # Get the ground truth mask and prediction for this single image
                    gt_mask_i = masks[i].cpu().numpy().flatten()
                    pred_scores_i = anomaly_map_blurred[i].cpu().numpy().flatten()
                    
                    # Calculate AUPR for this image and append to our list
                    aupr = average_precision_score(gt_mask_i, pred_scores_i)
                    per_image_pixel_aupr.append(aupr)

            for i in range(images.size(0)):
                all_pixel_scores.extend(anomaly_map_blurred[i].cpu().numpy().flatten())
                all_pixel_labels.extend(masks[i].cpu().numpy().flatten())

            
    # Calculate AUROC scores
    image_auroc = roc_auc_score(all_img_labels, all_img_scores)
    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    
    pixel_aupr = np.mean(per_image_pixel_aupr)
  
    # Find the optimal threshold from the ROC curve
    fpr, tpr, thresholds = roc_curve(all_img_labels, all_img_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Generate binary predictions based on the threshold
    predicted_labels = [1 if score >= optimal_threshold else 0 for score in all_img_scores]
    
    # Calculate precision and F1 score
    accuracy = accuracy_score(all_img_labels, predicted_labels)
    f1 = f1_score(all_img_labels, predicted_labels)
    
    return image_auroc, pixel_auroc, pixel_aupr, accuracy, f1