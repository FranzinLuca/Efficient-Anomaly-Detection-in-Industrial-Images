import os
import glob
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ------------------------------------------------ #
# MVTEC_AD #
# ------------------------------------------------ #

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

def load_MVTEC_AD(main_path, transform, batch_size=32):
    # Download the MVTec AD dataset from Kaggle
    #path = kagglehub.dataset_download("ipythonx/mvtec-ad")
    #print("Path to dataset files:", path)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    # train and test dataset paths
    train_dataset=Dataset_MVTEC_AD(main_path,mode="train",transform=transform)
    test_dataset=Dataset_MVTEC_AD(main_path,mode="test",transform=transform)

    # Create DataLoader for training,validation and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader, test_loader



# ------------------------------------------------ #
# BTAD #
# ------------------------------------------------ #
class load_dataset_BTAD(Dataset):
    def __init__(self, image_paths, transform = None, ground_truth_paths=[]):
        self.paths = image_paths
        self.ground_truth_paths = ground_truth_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path_string = self.paths[i]
        img = Image.open(path_string).convert('RGB')
        if self.transform:
            img=self.transform(img)
        
        if "ko" in path_string:
            label = 0
        elif "ok" in path_string:
            label = 1

        if len(self.ground_truth_paths)>0:
            img_gt = Image.open(self.ground_truth_paths[i]).convert('RGB')
            if self.transform:
                img_gt=self.transform(img_gt)
        else:
            img_gt = torch.zeros((1, 512, 512))
        
        return img, label, img_gt
    
def load_train_paths_BTAD(main_folder):
    subfolders = os.listdir(main_folder)
    train_folder_paths = [f"{main_folder}/{base_path}/train/ok" for base_path in subfolders]
    img_train_paths = [f"{path}/{img}" for path in train_folder_paths for img in os.listdir(path)]
    return img_train_paths, None

def load_test_paths_BTAD(main_folder):
    subfolders = os.listdir(main_folder)
    test_folder_paths_ko = [f"{main_folder}/{base_path}/test/ko" for base_path in subfolders]
    test_folder_paths_ok = [f"{main_folder}/{base_path}/test/ok" for base_path in subfolders]
    gt_folder_paths_ko = [f"{main_folder}/{base_path}/ground_truth/ko" for base_path in subfolders]
    test_folder_paths = test_folder_paths_ko + test_folder_paths_ok
    img_test_paths = [f"{path}/{img}" for path in test_folder_paths for img in os.listdir(path)]
    img_gt_paths = [f"{path}/{img}" for path in gt_folder_paths_ko for img in os.listdir(path)]
    return img_test_paths, img_gt_paths

def load_BTAD(main_path, transform=None, batch_size=32):
    train_paths, _ = load_train_paths_BTAD(main_path)
    test_paths, gt_paths = load_test_paths_BTAD(main_path)
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    dataset_train = load_dataset_BTAD(train_paths, transform=transform)
    dataset_test = load_dataset_BTAD(test_paths, transform=transform, ground_truth_paths=gt_paths)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
