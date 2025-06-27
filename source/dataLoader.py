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
    def __init__(self,root_dir,mode,transform=None,class_selected="all",augment=False,aug_prob=0.5):
        self.root_dir=root_dir
        self.mode=mode
        self.transform=transform
        self.augment=augment and mode=='train'
        self.aug_prob=aug_prob
        self.classes=[]
        self.samples = []
        self.class_to_idx =0
        self.type_class=class_selected
        
        if self.mode=='train':
            for class_name in tqdm(os.listdir(root_dir), desc="Classes"):
                if self.type_class==class_name:
                    class_path = os.path.join(root_dir,class_name,"train","good")
                    if not os.path.isdir(class_path):
                        continue
                    
                    img_paths=glob.glob(os.path.join(class_path, "*.png"))
                     
                    for img_path in tqdm(img_paths,desc=f"Loading {class_name}", leave=False):
                        self.samples.append((img_path,self.class_to_idx,1))
                    print(f"takes all sample class {class_name}")
                elif self.type_class=="all":
                    class_path = os.path.join(root_dir,class_name,"train","good")
                    if not os.path.isdir(class_path):
                        continue
                    img_paths=glob.glob(os.path.join(class_path, "*.png"))
                    
                    self.classes.append(class_name)
                    
                    for img_path in tqdm(img_paths,desc=f"Loading {class_name}", leave=False):
                        self.samples.append((img_path,self.class_to_idx,1))
                    
                self.class_to_idx+=1
        elif self.mode=='test':
            for class_name in tqdm(os.listdir(root_dir),desc="Classes"):
                if self.type_class==class_name:
                    class_path=os.path.join(root_dir,class_name,"test")
                    path_ground_truth=os.path.join(root_dir,class_name,"ground_truth")
                
                    if not os.path.isdir(class_path):
                        continue
                    list_path=[img_modified_path for img_modified_path in os.listdir(class_path)]
    
                    
                    for path_test_img in list_path:
                        img_paths=glob.glob(os.path.join(class_path,path_test_img,"*.png"))
                        if path_test_img!="good":
                            img_ground_truth=glob.glob(os.path.join(path_ground_truth,path_test_img,"*.png"))
                            for img_path in range(0,len(img_paths)):
                                self.samples.append((img_paths[img_path],img_ground_truth[img_path],self.class_to_idx,0))
                        elif path_test_img=="good":
                            for img_path in range(0,len(img_paths)):
                                self.samples.append((img_paths[img_path],None,self.class_to_idx,1))
                   
                elif self.type_class=="all":
                    class_path=os.path.join(root_dir,class_name,"test")
                    path_ground_truth=os.path.join(root_dir,class_name,"ground_truth")
                    
                    if not os.path.isdir(class_path):
                        continue
                    list_path=[img_modified_path for img_modified_path in os.listdir(class_path)]
                    
                    self.classes.append(class_name)
                    
                    for path_test_img in list_path:
                        img_paths=glob.glob(os.path.join(class_path,path_test_img,"*.png"))
                        if path_test_img!="good":
                            img_ground_truth=glob.glob(os.path.join(path_ground_truth,path_test_img,"*.png"))
                            for img_path in range(0,len(img_paths)):
                                self.samples.append((img_paths[img_path],img_ground_truth[img_path],self.class_to_idx,0))
                        elif path_test_img=="good":
                            for img_path in range(0,len(img_paths)):
                                self.samples.append((img_paths[img_path],None,self.class_to_idx,1))
                self.class_to_idx+=1
        else:
            assert("error of parameter mode!=train and mode!=test")
            
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if self.mode=="test":
            img_path,_,label,normal_img=self.samples[idx]
        else:
            img_path, label,normal_img = self.samples[idx]
        try:
            img=Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images at idx {idx}:")
            print(f"RGB path: {img_path}")
            raise e
        if self.transform:
            img=self.transform(img)
        if self.augment:
            img=apply_random_augmentation(img,self.aug_prob)
        return img,label,normal_img


def load_MVTEC_AD(main_path, transform, batch_size=32, class_selected=None):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, class_selected=class_selected)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, class_selected=class_selected)
    return train_loader, test_loader



# ------------------------------------------------ #
# BTAD #
# ------------------------------------------------ #
class load_dataset_BTAD(Dataset):
    def __init__(self, image_paths, transform = None, ground_truth_paths=None):
        self.paths = image_paths
        self.ground_truth_paths = ground_truth_paths if ground_truth_paths is not None else []
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
        else:
            label = 1

        gt_path = self.ground_truth_paths[i] if self.ground_truth_paths and i < len(self.ground_truth_paths) else None

        if gt_path is not None:
            img_gt = Image.open(gt_path).convert('L')
            if self.transform:
                img_gt=self.transform(img_gt)
        else:
            _, H, W = img.shape
            img_gt = torch.zeros((1, H, W))
        
        return img, label, img_gt
    
def load_train_paths_BTAD(main_folder, class_selected=None):
    subfolders = os.listdir(main_folder)
    if class_selected in subfolders:
        subfolders=[class_selected]
    train_folder_paths = [f"{main_folder}/{base_path}/train/ok" for base_path in subfolders]
    img_train_paths = [f"{path}/{img}" for path in train_folder_paths for img in os.listdir(path)]
    return img_train_paths, None

def load_test_paths_BTAD(main_folder, class_selected=None):
    subfolders = os.listdir(main_folder)
    if class_selected in subfolders:
        subfolders=[class_selected]
    test_folder_paths_ko = [f"{main_folder}/{base_path}/test/ko" for base_path in subfolders]
    test_folder_paths_ko = [p for p in test_folder_paths_ko if os.path.isdir(p)]

    test_folder_paths_ok = [f"{main_folder}/{base_path}/test/ok" for base_path in subfolders]
    test_folder_paths_ok = [p for p in test_folder_paths_ok if os.path.isdir(p)]

    gt_folder_paths_ko = [f"{main_folder}/{base_path}/ground_truth/ko" for base_path in subfolders]
    gt_folder_paths_ko = [p for p in gt_folder_paths_ko if os.path.isdir(p)]

    test_folder_paths = test_folder_paths_ko + test_folder_paths_ok

    img_test_paths = [f"{path}/{img}" for path in test_folder_paths for img in os.listdir(path)]
    img_gt_paths = [f"{path}/{img}" for path in gt_folder_paths_ko for img in os.listdir(path)]

    num_test = len(img_test_paths)
    num_gt = len(img_gt_paths)

    if num_test > num_gt:
        padding = [None] * (num_test - num_gt)
        img_gt_paths.extend(padding)

    return img_test_paths, img_gt_paths

def load_BTAD(main_path, transform_train=None,transform_test=None, batch_size=32, pin_memory=True, class_selected=None):
    train_paths, _ = load_train_paths_BTAD(main_path, class_selected=class_selected)
    test_paths, gt_paths = load_test_paths_BTAD(main_path,  class_selected=class_selected)
    
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
    
    dataset_train = load_dataset_BTAD(train_paths, transform=transform_train)
    dataset_test = load_dataset_BTAD(test_paths, transform=transform_test, ground_truth_paths=gt_paths)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_dataloader, test_dataloader
