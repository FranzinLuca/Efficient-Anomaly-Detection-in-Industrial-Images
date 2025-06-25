import os
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import glob
import numpy as np
from tqdm.notebook import tqdm
from collections import Counter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
from torch.utils.data import random_split

class Dataset_MVTEC_AD(Dataset):
    def __init__(self,root_dir,mode='train',transform=None,augment=False,aug_prob=0.5):
        # Initialize the dataset
        self.root_dir=root_dir  # Path to the dataset directory
        self.mode=mode # 'train' or 'test'
        self.transform=transform # Transformations to apply to the images
        self.augment=augment and mode=='train' # Whether to apply augmentations
        self.aug_prob=aug_prob # Probability of applying augmentations
        self.classes=[] # List of class names 
        self.samples = [] # List to hold image paths and labels
        self.class_to_idx = {} # Dictionary mapping class names to indices

        if self.mode=='train':
            # Load training data
             for class_name in tqdm(os.listdir(root_dir), desc="Classes"):
                 
                 class_path = os.path.join(root_dir,class_name,"train","good") # Path to the class directory
                 if not os.path.isdir(class_path):
                    continue
                 self.classes.append(class_name)
                # Check if the class name is already in the dictionary, if not, add it
                 if class_name not in self.class_to_idx: 
                     self.class_to_idx[class_name]=len(self.class_to_idx)
                     
                 img_paths=glob.glob(os.path.join(class_path, "*.png")) # Get all image paths in the class directory
                 
                 for img_path in tqdm(img_paths,desc=f"Loading {class_name}", leave=False):
                     self.samples.append((img_path,self.class_to_idx[class_name])) # Append the image path and label to the samples list
        elif self.mode=='test':
            for class_name in tqdm(os.listdir(root_dir),desc="Classes"):
                
                class_path=os.path.join(root_dir,class_name,"test")
                path_ground_truth=os.path.join(root_dir,class_name,"ground_truth") # Path to the ground truth directory
                
                if not os.path.isdir(class_path):
                    continue
                list_path=[img_modified_path for img_modified_path in os.listdir(class_path)] # List of subdirectories in the test directory
                
                if class_name not in self.class_to_idx:
                     self.class_to_idx[class_name]=len(self.class_to_idx)
                    
                for path_test_img in list_path:
                    
                    img_paths=glob.glob(os.path.join(path_test_img, "*.png")) # Get all image paths in the test subdirectory
                    # Check if the subdirectory is not "good" to include ground truth images
                    if path_test_img!="good":

                        img_ground_truth=glob.glob(os.path.join(path_ground_truth,path_test_img,"*.png"))
                        for img_path in range(0,len(img_paths)):
                            self.samples.append((img_paths[img_path],img_ground_truth[img_path],self.class_to_idx[class_name]))
                    
                    elif path_test_img=="good":
                        # If the subdirectory is "good", only include the image paths without ground truth
                        for img_path in range(0,len(img_paths)):
                            self.samples.append((img_paths[img_path],None,self.class_to_idx[class_name]))
                        
        else:
            assert("error of parameter mode!=train and mode!=test")
            
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if self.mode=="test":
            img_path,_,label=self.samples[idx]
        else:
            img_path, label = self.samples[idx]
        try:
            img=Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images at idx {idx}:")
            print(f"RGB path: {img_path}")
            raise e
        if self.transform:
            img=self.transform(img)
        #if self.augment:
         #   img=apply_random_augmentation(img,self.aug_prob)
        return img,label

transform = transforms.Compose([
    transforms.Resize((512,512)), # Resize images to 512x512
    transforms.ToTensor() # Convert images to PyTorch tensors
])
# Download the MVTec AD dataset from Kaggle
path = kagglehub.dataset_download("ipythonx/mvtec-ad")
print("Path to dataset files:", path)

# train and test dataset paths
train_dataset=Dataset_MVTEC_AD(path,mode="train",transform=transform,augment=False)
test_dataset=Dataset_MVTEC_AD(path,mode="test",transform=transform,augment=False)

train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
# Split the training dataset into training and validation subsets
train_subset, val_subset = random_split(train_dataset, [train_len, val_len])


# Create DataLoader for training,validation and test datasets
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
