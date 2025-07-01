"""
Utility functions
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import csv

def save_visualizations(model, data_loader, category, config):
    """
    Generates and saves 3-column visualizations for anomaly detection results.
    
    Args:
        model: The trained anomaly detection model.
        data_loader: DataLoader for the test set.
        device: The device to run the model on.
        save_path: The directory path where the output images will be saved.
        n_images: The total number of images to generate and save.
    Returns:
        path_images: List of file paths where the images are saved.
    """
    # 1. Create the destination folder if it doesn't exist
    save_path = config.IMAGE_FOLDER 
    n_images = config.NUM_IMAGES_TO_SAVE
    path_images = []   
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving {n_images} visualization images to: {save_path}")

    model.eval()
    saved_count = 0

    # Use a progress bar for better user experience
    pbar = tqdm(total=n_images, desc="Generating and Saving Images")

    # 2. Iterate through the data loader to get enough images
    with torch.no_grad():
        for images, labels, masks in data_loader:
            # Stop if we have already saved the required number of images
            if saved_count >= n_images:
                break

            images = images.to(config.DEVICE)

            # Get model predictions
            original_features, reconstructed_features = model(images)
            anomaly_map = model.get_anomaly_map(original_features, reconstructed_features)
            anomaly_map_resized = F.interpolate(anomaly_map, size=images.shape[2:], mode='bilinear', align_corners=False)

            # Move data to CPU for plotting
            images_cpu = images.cpu()
            masks_cpu = masks.cpu().numpy()
            anomaly_map_cpu = anomaly_map_resized.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            # 3. Process and save each image in the current batch
            for i in range(images.size(0)):
                if saved_count >= n_images:
                    break

                img_to_show = images_cpu[i].permute(1, 2, 0).numpy().clip(0, 1)
                label_text = "Anomaly" if labels_cpu[i] == 1 else "Normal"

                # Create a 3-column plot for the current image
                fig, axs = plt.subplots(3, 1, figsize=(6, 18))
                fig.suptitle(f'Sample {saved_count + 1}: ({label_text})', fontsize=16)

                # Column 1: Original Image
                axs[0].imshow(img_to_show)
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                # Column 2: Ground Truth Mask
                axs[1].imshow(masks_cpu[i, 0], cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[1].axis('off')

                # Column 3: Predicted Anomaly Overlay
                axs[2].imshow(img_to_show)
                axs[2].imshow(anomaly_map_cpu[i, 0], cmap='jet', alpha=0.6) # alpha adjusted for better visibility
                axs[2].set_title("Predicted Anomaly")
                axs[2].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # Save the generated figure to the specified path
                file_path = os.path.join(save_path, f'{config.MODEL}_{category}_{saved_count + 1}_.png')
                path_images.append(file_path)
                plt.savefig(file_path)
                plt.close(fig)  # Close the figure to free up memory

                saved_count += 1
                pbar.update(1)

    pbar.close()
    print("Done.")
    
    return path_images

def save_results_to_csv(model_name, category_name, image_auroc, pixel_auroc, accuracy, f1, path_images, save_dir):
    """
    Saves experiment results to a model-specific CSV file.

    Args:
        model_name: The name of the model.
        category_name: The data category being evaluated.
        image_auroc: The image-level AUROC score.
        pixel_auroc: The pixel-level AUROC score.
        precision: The precision score.
        f1: The F1-score.
        path_images: A list of paths to saved visualization images.
        save_dir: The directory where the CSV files will be stored.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_results.csv")
    new_row_data = {
        'Category': category_name,
        'Image-AUROC': f"{image_auroc:.4f}",
        'Pixel-AUROC': f"{pixel_auroc:.4f}",
        'Accuracy': f"{accuracy:.4f}",
        'F1-Score': f"{f1:.4f}",
        'Image Paths': "; ".join(path_images)
    }
    
    # 2. Check if the CSV file already exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Check if the category is already in the DataFrame
        if category_name in df['Category'].values:
            # Update the existing row
            idx = df.index[df['Category'] == category_name].tolist()[0]
            df.loc[idx] = new_row_data
            print(f"Updating results for category '{category_name}' in '{file_path}'")
        else:
            # Append the new row if the category doesn't exist
            new_df = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"Adding new results for category '{category_name}' to '{file_path}'")
    else:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame([new_row_data])
        print(f"Creating new results file for category '{category_name}' at '{file_path}'")

    # Save the modified or new DataFrame back to the CSV
    df.to_csv(file_path, index=False)
    
    
def plot_all_categories_with_images(csv_path, img_to_plot=[], save_path=None):
    """
    Reads a results CSV and creates a grid of plots, with each plot
    showing the metrics and a sample image for a single category.

    Args:
        csv_path: The path to the results CSV file.
        save_path: If provided, saves the entire grid plot.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV at '{csv_path}'")
        return

    num_categories = len(df)
    fig, axes = plt.subplots(num_categories, 2, figsize=(10, 4 * num_categories))

    if num_categories == 1:
        axes = np.array([axes])

    fig.suptitle("Model Performance Summary by Category", fontsize=20, y=1.0)

    metrics_to_plot = ['Image-AUROC', 'Pixel-AUROC', 'Precision', 'F1-Score']

    for index, row in df.iterrows():
        category_name = row['Category']
        ax_bar = axes[index, 0]
        ax_img = axes[index, 1]

        # --- Panel 1: Bar Chart for the Category ---
        values = [row[metric] for metric in metrics_to_plot]
        ax_bar.bar(metrics_to_plot, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax_bar.set_title(f"Metrics for '{category_name}'", fontsize=14)
        ax_bar.set_ylabel("Score")
        ax_bar.set_ylim(0, 1.1)
        ax_bar.tick_params(axis='x', rotation=45)

        # --- Panel 2: Sample Image for the Category ---
        try:
            index_to_plot = img_to_plot[index] if index < len(img_to_plot) else 0
            image_path = row['Image Paths'].split('; ')[index_to_plot]
            img = mpimg.imread(image_path)
            ax_img.imshow(img)
        except (FileNotFoundError, IndexError):
            ax_img.text(0.5, 0.5, "Image not found", ha='center', va='center')

        ax_img.axis('off') # Hide axes for the image

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Grid plot saved to '{save_path}'")

    plt.show()