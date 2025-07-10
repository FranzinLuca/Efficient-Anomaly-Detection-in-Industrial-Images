from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, average_precision_score
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
import config
from sklearn.metrics import accuracy_score, f1_score,roc_curve,roc_auc_score
import piq
from losses import denormalize 

def evaluate_model(model, dataloader):
    """
    Evaluate the model on the validation/test dataset.
    This function computes the image-level and pixel-level AUROC, precision, and F1 score.
    """
    device = config.DEVICE
    model.eval()
    num_test_images = len(dataloader.dataset)
    first_images, _, _ = next(iter(dataloader))
    _, _, H, W = first_images.shape
    
    total_pixels = num_test_images * H * W
    
    all_img_scores_np = np.zeros(num_test_images)
    all_img_labels_np = np.zeros(num_test_images)
    all_pixel_scores_np = np.zeros(total_pixels)
    all_pixel_labels_np = np.zeros(total_pixels)
    
    per_image_pixel_aupr = []
    
    current_img_idx = 0
    current_pixel_idx = 0

    with torch.no_grad():
        for images, labels, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            original_features, reconstructed_features = model(images)
            anomaly_map = model.get_anomaly_map(original_features, reconstructed_features)

            anomaly_map_resized = F.interpolate(anomaly_map, size=images.shape[2:], mode='bilinear', align_corners=False)
            anomaly_map_blurred = gaussian_blur(anomaly_map_resized, kernel_size=config.KERNEL_SIZE, sigma=config.SIGMA)

            B, _, H, W = anomaly_map_blurred.shape
            k = int(H * W * 0.01)
            if k == 0: k = 1

            anomaly_scores_flat = anomaly_map_blurred.view(B, -1)
            top_k_scores, _ = torch.topk(anomaly_scores_flat, k, dim=1)
            img_scores = torch.mean(top_k_scores, dim=1)

            batch_size = images.size(0)
            num_pixels_in_batch = batch_size * H * W
            
            all_img_scores_np[current_img_idx : current_img_idx + batch_size] = img_scores.cpu().numpy()
            all_img_labels_np[current_img_idx : current_img_idx + batch_size] = labels.numpy()
            all_pixel_scores_np[current_pixel_idx : current_pixel_idx + num_pixels_in_batch] = anomaly_map_blurred.cpu().numpy().flatten()
            all_pixel_labels_np[current_pixel_idx : current_pixel_idx + num_pixels_in_batch] = masks.cpu().numpy().flatten()
            
            current_img_idx += batch_size
            current_pixel_idx += num_pixels_in_batch

            for i in range(batch_size):
                if labels[i] == 1:
                    gt_mask_i = masks[i].cpu().numpy().flatten()
                    if np.sum(gt_mask_i) > 0:  # Ensure there is an anomaly to score
                        pred_scores_i = anomaly_map_blurred[i].cpu().numpy().flatten()
                        aupr = average_precision_score(gt_mask_i, pred_scores_i)
                        per_image_pixel_aupr.append(aupr)

    print("Finished collecting scores. Now calculating metrics...")

    image_auroc = roc_auc_score(all_img_labels_np, all_img_scores_np)
    pixel_auroc = roc_auc_score(all_pixel_labels_np, all_pixel_scores_np)
    
    pixel_aupr = np.mean(per_image_pixel_aupr) if per_image_pixel_aupr else 0.0

    fpr, tpr, thresholds = roc_curve(all_img_labels_np, all_img_scores_np)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predicted_labels = (all_img_scores_np >= optimal_threshold).astype(int)

    accuracy = accuracy_score(all_img_labels_np, predicted_labels)
    f1 = f1_score(all_img_labels_np, predicted_labels)
    
    print("Metrics calculation complete.")

    return image_auroc, pixel_auroc, pixel_aupr, accuracy, f1

def evaluate_ANOViT(model, dataloader, device):
    model.eval()

    lambda_mean = 0.5
    lambda_ssim = 0.5
    total_weight = lambda_mean + lambda_ssim
    
    all_labels = []
    mean_error_scores = []
    ssim_dissimilarity_scores = []
    all_pixel_maps = []
    all_gt_masks = []

    with torch.no_grad():
        for image, anomaly_label, gt_mask in tqdm(dataloader, desc="Evaluating ANOVit"):

            image, gt_mask = image.to(device), gt_mask.to(device)
            recon_img = model(image)
            
            anomaly_map = model.get_anomaly_map(image, recon_img)

            flat_map = anomaly_map.view(anomaly_map.shape[0], -1)
            mean_error_per_image = flat_map.mean(dim=1)
            
            ssim_per_image = piq.ssim(denormalize(recon_img),denormalize(image), data_range=1.0, reduction="none")
            ssim_dissimilarity_per_image = 1.0 - ssim_per_image
            
            all_labels.extend(anomaly_label.numpy())
            mean_error_scores.append(mean_error_per_image)
            ssim_dissimilarity_scores.append(ssim_dissimilarity_per_image)
            all_pixel_maps.append(anomaly_map)
            all_gt_masks.append(gt_mask)

    mean_scores_tensor = torch.cat(mean_error_scores).cpu().numpy()
    ssim_scores_tensor = torch.cat(ssim_dissimilarity_scores).cpu().numpy()
    
    all_pixel_maps_np = torch.cat(all_pixel_maps).cpu().numpy()
    all_gt_masks_np = torch.cat(all_gt_masks).cpu().numpy()
    all_labels_np = np.array(all_labels)

    mse_norm = (mean_scores_tensor - mean_scores_tensor.min()) / (mean_scores_tensor.max() - mean_scores_tensor.min() + 1e-8)
    ssim_norm = (ssim_scores_tensor - ssim_scores_tensor.min()) / (ssim_scores_tensor.max() - ssim_scores_tensor.min() + 1e-8)
    combined_scores = ((lambda_mean / total_weight) * mse_norm) + ((lambda_ssim / total_weight) * ssim_norm)

    image_auroc = roc_auc_score(all_labels_np, combined_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels_np, combined_scores)
    optimal_idx = np.argmax(tpr - fpr)  # Youden's J-statistic
    best_thresh = thresholds[optimal_idx]
    
    predicted_labels = (combined_scores >= best_thresh).astype(int)
    accuracy = accuracy_score(all_labels_np, predicted_labels)
    f1 = f1_score(all_labels_np, predicted_labels)

    gt_pixel_labels = all_gt_masks_np.flatten()
    pred_pixel_scores = all_pixel_maps_np.flatten()
    
    pixel_auroc = roc_auc_score(gt_pixel_labels, pred_pixel_scores)

    aupr_scores = []
    is_anomalous_mask = (all_labels_np == 1)
    anomalous_gt = all_gt_masks_np[is_anomalous_mask]
    anomalous_pred = all_pixel_maps_np[is_anomalous_mask]

    for i in range(len(anomalous_gt)):
        if anomalous_gt[i].sum() > 0:
            aupr = average_precision_score(anomalous_gt[i].flatten(), anomalous_pred[i].flatten())
            aupr_scores.append(aupr)
    
    pixel_aupr = np.mean(aupr_scores) if aupr_scores else 0.0

    print("ANOVit Evaluation Complete.")
    return image_auroc, pixel_auroc, pixel_aupr, accuracy, f1