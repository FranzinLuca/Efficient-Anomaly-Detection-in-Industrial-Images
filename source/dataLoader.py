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
    # This function is well-written, no changes needed.
    img_train_paths = []
    try:
        if class_selected:
            class_path = os.path.join(main_folder, class_selected)
            subfolders = [class_selected] if os.path.isdir(class_path) else []
            if not subfolders:
                logging.warning(f"Selected class folder '{class_selected}' not found in '{main_folder}'.")
        else:
            subfolders = [d.name for d in os.scandir(main_folder) if d.is_dir()]

        for class_folder in subfolders:
            train_folder = os.path.join(main_folder, class_folder, 'train')
            if not os.path.isdir(train_folder):
                logging.info(f"Skipping '{class_folder}': 'train' directory not found.")
                continue

            for item in os.scandir(train_folder):
                if item.is_dir():
                    image_folder = item.path
                    try:
                        for img_file in os.listdir(image_folder):
                            full_path = os.path.join(image_folder, img_file)
                            if os.path.isfile(full_path):
                                img_train_paths.append(full_path)
                    except FileNotFoundError:
                        logging.error(f"Error accessing directory: {image_folder}")

    except FileNotFoundError:
        logging.error(f"Main folder not found: {main_folder}")
        return []

    return img_train_paths

def load_test_paths(main_folder, class_selected=None):
    test_image_paths = []
    gt_image_paths = []
    try:
        if class_selected:
            subfolders = [class_selected] if os.path.isdir(os.path.join(main_folder, class_selected)) else []
            if not subfolders:
                logging.warning(f"Selected class folder '{class_selected}' not found in '{main_folder}'.")
        else:
            subfolders = [d.name for d in os.scandir(main_folder) if d.is_dir()]

    except FileNotFoundError:
        logging.error(f"Main folder not found: {main_folder}")
        return [], [], set() # <<< REFINEMENT: Return empty set on error

    extra_test_folders = set()

    for class_folder in subfolders:
        test_dir = os.path.join(main_folder, class_folder, 'test')
        gt_dir = os.path.join(main_folder, class_folder, 'ground_truth')

        try:
            test_subfolders = {d.name for d in os.scandir(test_dir) if d.is_dir()}
        except FileNotFoundError:
            logging.warning(f"Skipping class '{class_folder}': 'test' directory not found.")
            continue

        try:
            gt_subfolders = {d.name for d in os.scandir(gt_dir) if d.is_dir()}
        except FileNotFoundError:
            gt_subfolders = set()
            logging.info(f"Class '{class_folder}': 'ground_truth' directory not found.")

        class_extra_folders = test_subfolders - gt_subfolders
        extra_test_folders.update(class_extra_folders)

        for subfolder_name in sorted(list(test_subfolders)):
            current_test_path = os.path.join(test_dir, subfolder_name)
            try:
                # <<< REFINEMENT: Sort file lists to ensure correct image-mask pairing >>>
                image_files = sorted([f for f in os.listdir(current_test_path) if os.path.isfile(os.path.join(current_test_path, f))])
                test_image_paths.extend([os.path.join(current_test_path, img) for img in image_files])
                
                if subfolder_name in class_extra_folders:
                    gt_image_paths.extend([None] * len(image_files))
                else:
                    current_gt_path = os.path.join(gt_dir, subfolder_name)
                    # <<< REFINEMENT: Sort ground truth files as well >>>
                    gt_files = sorted([f for f in os.listdir(current_gt_path) if os.path.isfile(os.path.join(current_gt_path, f))])
                    
                    # <<< REFINEMENT: Add a check for mismatched file counts >>>
                    if len(image_files) != len(gt_files):
                        logging.warning(f"Mismatch in file count for '{subfolder_name}': {len(image_files)} test images vs {len(gt_files)} ground truth masks. Skipping ground truth for this folder.")
                        gt_image_paths.extend([None] * len(image_files)) # Add None to avoid crashing
                    else:
                        gt_image_paths.extend([os.path.join(current_gt_path, img) for img in gt_files])

            except FileNotFoundError:
                logging.warning(f"Could not access directory or contents for '{current_test_path}'")
                continue

    return test_image_paths, gt_image_paths, extra_test_folders


class Img_Dataset(Dataset):
    # This class is well-written, no changes needed.
    def __init__(self, image_paths, ground_truth_paths=None, transform=None, good_fld=None):
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
            img = self.transform(img)

        if self.good_folder is None: # For training set where only good samples are used
            label = 0
        elif self.good_folder in path_string:
            label = 0
        else:
            label = 1

        gt_path = self.ground_truth_paths[i] if self.ground_truth_paths and i < len(self.ground_truth_paths) else None

        if gt_path is not None:
            img_gt = Image.open(gt_path).convert('L') # Greyscale mask
            if self.transform:
                img_gt = self.transform(img_gt)
                img_gt = (img_gt > 0.5).long() # Binarize the mask
        else:
            # Create a placeholder zero tensor if no ground truth
            _, H, W = img.shape
            img_gt = torch.zeros((1, H, W))
        
        return img, label, img_gt

def load_dataset(main_path, transform_train=None, transform_test=None, batch_size=32, pin_memory=True, class_selected=None):
    train_paths = load_train_paths(main_path, class_selected=class_selected)
    test_paths, gt_paths, gd_folders = load_test_paths(main_path, class_selected=class_selected)

    # <<< REFINEMENT: Make the identification of the 'good' folder more robust >>>
    good_folder = None
    if len(gd_folders) == 1:
        good_folder = list(gd_folders)[0]
        logging.info(f"Identified '{good_folder}' as the normal sample folder.")
    elif len(gd_folders) > 1:
        logging.warning(f"Found multiple folders without ground truth: {gd_folders}. Cannot determine the 'good' folder. Labels for test set may be incorrect.")
        # Optionally, you could pick one, e.g., 'good' if it exists in the set
        if 'good' in gd_folders:
             good_folder = 'good'
             logging.info("Defaulting to 'good' as the normal sample folder.")
    else:
        logging.warning("Could not find a 'good' folder (a folder in 'test' that is not in 'ground_truth'). Labels for test set may be incorrect.")

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
    
    # For training, we don't need to distinguish good/bad, as we assume all are good
    dataset_train = Img_Dataset(train_paths, transform=transform_train, good_fld=None)
    
    # For testing, we provide the identified 'good_folder' to generate correct labels
    dataset_test = Img_Dataset(test_paths, transform=transform_test, ground_truth_paths=gt_paths, good_fld=good_folder)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    return train_dataloader, test_dataloader
