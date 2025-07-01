import torch

# -- Model Configuration ---
MODEL = "ADTR_FPN" # ADTR_FPN, ADTR
CHECKPOINT_DIR = f"checkpoints/{MODEL}"
IMAGE_FOLDER = f"images/{MODEL}"
RESULT_FOLDER = f"results/{MODEL}"
TRAIN_MODEL = False  # Set to False if you want to use a pre-trained model
LOAD_WEIGHTS = True  # Set to False if you want to train from scratch

# --- Data configurations ---
DOWNLOAD_DATASET = True  # Set to False if you have already downloaded the dataset
DELETE_CACHE_DATASET = False  # Set to True if you want to delete the cached dataset when the script ends

DATASET_TO_USE = "mvtec" # mvtec, batd

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
IMG_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_DYT = True  # Use Dynamic Tensor for ADTR

# -- Runtime Settings --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
RANDOM_SEED = 42
NUM_IMAGES_TO_SAVE = 8

# -- Plotting configurations ---
IMAGE_TO_PLOT_MVTEC = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Indices of images to plot for each category
IMAGE_TO_PLOT_BTAD = [0,0,0]  # Indices of images to plot for each category