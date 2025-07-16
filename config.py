import torch

# -- Model Configuration ---
USE_DYT = False  # Use Dynamic Tensor for ADTR
MODEL = "ADTR_FPN" # ADTR_FPN, ADTR, ANOVit
SUBPATH = "DyT" if USE_DYT else "Norm"
CHECKPOINT_DIR = f"checkpoints/{MODEL}/{SUBPATH}"
IMAGE_FOLDER = f"images/{MODEL}/{SUBPATH}"
RESULT_FOLDER = f"results/{MODEL}/{SUBPATH}"
TRAIN_MODEL = True  # Set to False if you want to use a pre-trained model
LOAD_WEIGHTS = False  # Set to False if you want to train from scratch

# --- Data configurations ---
DOWNLOAD_DATASET = True  # Set to False if you have already downloaded the dataset
DELETE_CACHE_DATASET = False  # Set to True if you want to delete the cached dataset when the script ends
VAL_SPLIT = 0.2  # Set to None if you don't want to split the training set into training and validation sets
DATASET_TO_USE = "btad" # mvtec, btad

MVTEC_ROOT = f"dataset/mvtec-ad" # if you have downloaded the dataset, set the path here
MVTEC_KAGGLE_DOWNLOAD_URL = "ipythonx/mvtec-ad"

BTAD_ROOT = f"dataset/btad-beantech-anomaly-detection/BTech_Dataset_transformed" # if you have downloaded the dataset, set the path here
BTAD_KAGGLE_DOWNLOAD_URL = "thtuan/btad-beantech-anomaly-detection"

# All categories in MVTEC dataset
# "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
#  "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
#  "transistor", "wood", "zipper"
MVTEC_CATEGORIES = [
     "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
     "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
     "transistor", "wood", "zipper"
     ]

# All categories in BTAD dataset
# "01", "02", "03"
BTAD_CATEGORIES = ["01", "02", "03"]

# --- Training configurations ---
IMG_SIZE = (512, 512)
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
# -- ANOViT specific configurations --
PATCH_SIZE = (16, 16)  # Patch size for ANOViT
D_MODEL = 1024  # Dimension of the model for ANOViT
N_CHANNELS = 3  # Number of input channels (RGB)
N_HEADS = 16  # Number of attention heads for ANOViT
N_LAYERS = 12  # Number of transformer layers for ANOViT
LAMBDA_RECON = 0.85  # Weight for reconstruction loss in ANOViT


# -- Runtime Settings --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 6
RANDOM_SEED = 42
NUM_IMAGES_TO_SAVE = 8

# -- Plotting configurations ---
IMAGE_TO_PLOT_MVTEC = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Indices of images to plot for each category
IMAGE_TO_PLOT_BTAD = [0,0,0]  # Indices of images to plot for each category
KERNEL_SIZE = 7  # Kernel size for Gaussian blur
SIGMA = 1.0  # Sigma for Gaussian blur