import os
import glob
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_train_paths(main_folder, class_selected=None):                                    
    img_train_paths = []
    try:
        # Determine which subfolders to process
        if class_selected:
            # If a specific class is selected, only consider that one
            class_path = os.path.join(main_folder, class_selected)
            subfolders = [class_selected] if os.path.isdir(class_path) else []
            if not subfolders:
                logging.warning(f"Selected class folder '{class_selected}' not found in '{main_folder}'.")
        else:
            # Otherwise, get all actual directories in the main folder
            subfolders = [d.name for d in os.scandir(main_folder) if d.is_dir()]

        # Iterate through the selected class folders
        for class_folder in subfolders:
            train_folder = os.path.join(main_folder, class_folder, 'train')

            # Check if the 'train' subdirectory exists before trying to access it
            if not os.path.isdir(train_folder):
                logging.info(f"Skipping '{class_folder}': 'train' directory not found.")
                continue

            # Iterate through the subdirectories within the 'train' folder (e.g., 'good', 'bad')
            for item in os.scandir(train_folder):
                if item.is_dir():
                    image_folder = item.path
                    try:
                        # Find all files within the image folder
                        for img_file in os.listdir(image_folder):
                            full_path = os.path.join(image_folder, img_file)
                            # Add a check to ensure we are only adding files, not nested directories
                            if os.path.isfile(full_path):
                                img_train_paths.append(full_path)
                    except FileNotFoundError:
                        logging.error(f"Error accessing directory: {image_folder}")

    except FileNotFoundError:
        logging.error(f"Main folder not found: {main_folder}")
        return [] # Return an empty list if the main directory doesn't exist

    return img_train_paths

def load_test_paths(main_folder, class_selected=None):
    test_image_paths = []
    gt_image_paths = []
    try:
        # Determine which top-level class folders to process
        if class_selected:
            subfolders = [class_selected] if os.path.isdir(os.path.join(main_folder, class_selected)) else []
            if not subfolders:
                logging.warning(f"Selected class folder '{class_selected}' not found in '{main_folder}'.")
        else:
            subfolders = [d.name for d in os.scandir(main_folder) if d.is_dir()]

    except FileNotFoundError:
        logging.error(f"Main folder not found: {main_folder}")
        return [], []

    # Process each class folder (e.g., 'wood', 'carpet')
    for class_folder in subfolders:
        test_dir = os.path.join(main_folder, class_folder, 'test')
        gt_dir = os.path.join(main_folder, class_folder, 'ground_truth')

        # Get subfolder names for test and ground_truth, if they exist
        try:
            test_subfolders = {d.name for d in os.scandir(test_dir) if d.is_dir()}
        except FileNotFoundError:
            logging.warning(f"Skipping class '{class_folder}': 'test' directory not found.")
            continue # Move to the next class_folder

        try:
            gt_subfolders = {d.name for d in os.scandir(gt_dir) if d.is_dir()}
        except FileNotFoundError:
            gt_subfolders = set() # Assume no ground truth folders if the directory is missing
            logging.info(f"Class '{class_folder}': 'ground_truth' directory not found.")

        # Find subfolders that are in 'test' but NOT in 'ground_truth'
        extra_test_folders = test_subfolders - gt_subfolders

        # Process each subfolder found in the test directory (e.g., 'good', 'bad')
        for subfolder_name in sorted(list(test_subfolders)): # sorted for deterministic order
            current_test_path = os.path.join(test_dir, subfolder_name)
            
            try:
                # Get all image file paths from the current test subfolder
                image_files = [f for f in os.listdir(current_test_path) if os.path.isfile(os.path.join(current_test_path, f))]
                test_image_paths.extend([os.path.join(current_test_path, img) for img in image_files])
                
                # --- THIS IS THE CORRECTED CONDITIONAL ---
                # Check if this subfolder is one of the "extra" ones
                if subfolder_name in extra_test_folders:
                    # If so, add None placeholders for the ground truth paths
                    gt_image_paths.extend([None] * len(image_files))
                else:
                    # Otherwise, get the corresponding ground truth image paths
                    current_gt_path = os.path.join(gt_dir, subfolder_name)
                    gt_files = [f for f in os.listdir(current_gt_path) if os.path.isfile(os.path.join(current_gt_path, f))]
                    gt_image_paths.extend([os.path.join(current_gt_path, img) for img in gt_files])

            except FileNotFoundError:
                logging.warning(f"Could not access directory or contents for '{current_test_path}'")
                continue # Skip this subfolder and move to the next

    return test_image_paths, gt_image_paths, extra_test_folders

class Img_Dataset(Dataset):
    def __init__(self, image_paths,ground_truth_paths=None, transform = None, good_fld=None):
        self.paths = image_paths
        self.ground_truth_paths = ground_truth_paths if ground_truth_paths is not None else []
        self.transform = transform
        self.good_folder = good_fld
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path_string = self.paths[i]
        img = Image.open(path_string).convert('RGB')
        if self.transform:
            img=self.transform(img)

        if self.good_folder == None:
            label = 1
        elif self.good_folder in path_string:
            label = 1
        else:
            label = 0

        gt_path = self.ground_truth_paths[i] if self.ground_truth_paths and i < len(self.ground_truth_paths) else None

        if gt_path is not None:
            img_gt = Image.open(gt_path).convert('L')
            if self.transform:
                img_gt=self.transform(img_gt)
                img_gt = (mask > 0.5).long()
        else:
            _, H, W = img.shape
            img_gt = torch.zeros((1, H, W))
        
        return img, label, img_gt

def load_dataset(main_path, transform_train=None,transform_test=None, batch_size=32, pin_memory=True, class_selected=None):
    train_paths = load_train_paths(main_path, class_selected=class_selected)
    test_paths, gt_paths, gd_folder = load_test_paths(main_path,  class_selected=class_selected)

    good_folder = list(gd_folder)[0]
    
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    dataset_train = Img_Dataset(train_paths, transform=transform_train)
    dataset_test = Img_Dataset(test_paths, transform=transform_test, ground_truth_paths=gt_paths, good_fld=good_folder)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_dataloader, test_dataloader
