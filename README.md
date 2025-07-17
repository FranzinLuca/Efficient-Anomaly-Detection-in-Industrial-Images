# Efficient-Anomaly-Detection

This project provides a powerful and flexible framework for **Efficient Anomaly Detection** in images using state-of-the-art deep learning models. It features implementations of **ADTR**, **ADTR\_FPN**, and **ANOVit**, and is designed for use with the **MVTec AD** and **BTAD** datasets.

## Prerequisites

Make sure you have Python 3 and pip installed.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/FranzinLuca/Efficient-Anomaly-Detection-in-Industrial-Images.git
    cd Efficient-Anomaly-Detection-in-Industrial-Images
    ```

2.  Install the required packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Configuration

The primary configuration for the project is handled in the `config.py` file. Key parameters include:

  * `USE_DYT`: Enable or disable Dynamic Tanh for ADTR.
  * `MODEL`: Select the model: "ADTR\_FPN", "ADTR", or "ANOVit".
  * `DATASET_TO_USE`: Choose the dataset: "mvtec" or "btad".
  * `TRAIN_MODEL`: Set to `True` to train the model, or `False` to load a pre-trained model.
  * `EPOCHS`: The number of training epochs.
  * `BATCH_SIZE`: The batch size for training.
  * `LAMBDA_RECON`: The lambda value for the reconstruction loss in the ANOViT model.

-----

## Usage

### Data Preparation

The project can automatically download the **MVTec AD** and **BTAD** datasets from Kaggle if they are not found locally. The data loading and preprocessing are handled by `source/dataLoader.py`, which creates `DataLoader` instances for training, validation, and testing.

### Training

To train a model, configure the desired parameters in `config.py` and run the `main.ipynb` notebook. The training process, managed by `source/train.py`, will save model checkpoints in the `checkpoints/` directory. The training script supports a learning rate scheduler with a linear warmup and cosine decay, which can be configured in `source/utils.py`.

### Evaluation

After training, the evaluation is performed automatically by the `source/evaluate.py` script. This script calculates a comprehensive set of metrics, including image-level and pixel-level AUROC, AUPR, accuracy, and F1-score.

### Results and Visualizations

The evaluation results are saved to a CSV file in the `results/` directory. Additionally, the `source/utils.py` script generates and saves detailed visualizations of the anomaly detection results in the `images/` directory. These visualizations include the original image, the ground truth mask, and the predicted anomaly map, providing a clear visual representation of the model's performance.

-----

## Models

This project includes three powerful models for anomaly detection:

### ADTR (Anomaly Detection Transformer)

The **ADTR** model is an efficient anomaly detection transformer that leverages a pre-trained EfficientNet-B5 as a backbone to extract features from the input images. These features are then processed by a transformer-based architecture to reconstruct the image and identify anomalies.

**Key Features:**

  * **EfficientNet-B5 Backbone**: Utilizes a powerful pre-trained CNN for feature extraction.
  * **Transformer Architecture**: Employs a transformer encoder and decoder to learn the relationship between different parts of the image and reconstruct it.
  * **Learnable Query**: Uses a learnable tensor as the initial query for the transformer decoder, which helps in generating a more accurate reconstruction of the normal features of the image.

### ADTR\_FPN (ADTR with Feature Pyramid Network)

The **ADTR\_FPN** model is an enhanced version of the ADTR model that incorporates a Feature Pyramid Network (FPN) to improve the detection of anomalies at different scales. The FPN allows the model to combine features from different levels of the backbone network, resulting in a more robust and accurate anomaly detection system.

**Key Features:**

  * **BiFPN**: Implements a Bi-directional Feature Pyramid Network (BiFPN) that allows for more efficient and effective feature fusion from different scales.
  * **Weighted Feature Fusion**: Uses a weighted feature fusion mechanism to combine features from different levels of the FPN, allowing the model to learn the importance of each feature map dynamically.
  * **Improved Multi-Scale Detection**: The FPN enables the model to better detect anomalies of various sizes and shapes by providing a richer set of features at different resolutions.

### ANOVit (Anomaly Vision Transformer)

The **ANOVit** model is a Vision Transformer (ViT) based approach to anomaly detection. It uses a pure transformer architecture to learn the normal patterns in the training data and then identifies anomalies by measuring the reconstruction error between the input image and the output of the model.

**Key Features:**

  * **Vision Transformer (ViT) Encoder**: The model uses a standard ViT encoder to learn a rich representation of the input image.
  * **Decoder for Reconstruction**: A decoder network is used to reconstruct the original image from the learned features.
  * **Pixel-Level Anomaly Detection**: The model can generate a pixel-level anomaly map by comparing the reconstructed image with the original input, allowing for precise localization of anomalies.

-----

## Project Structure

```
├── config.py               # Main configuration file
├── EDA.ipynb               # Exploratory Data Analysis notebook
├── main.ipynb              # Main notebook for training and evaluation
├── README.md               # This README file
├── requirements.txt        # Python package dependencies
├── results/                # Final results
├── images/                 # Images generated
└── source/
    ├── dataLoader.py       # Data loading and preprocessing
    ├── evaluate.py         # Evaluation functions
    ├── losses.py           # Loss functions
    ├── train.py            # Training loop
    ├── utils.py            # Utility functions for saving results and visualizations
    └── models/
        ├── ADTR.py         # ADTR model
        ├── ADTR_FPN.py     # ADTR_FPN model
        └── ANOVit.py       # ANOVit model
```

-----

## Code Modules

  * **`source/dataLoader.py`**: This module is responsible for loading and preparing the datasets. It includes functions to load training and test paths, and a custom `Img_Dataset` class to handle the images and masks.
  * **`source/losses.py`**: This file defines the loss functions used for training the models. It includes a standard MSE loss and a combined SSIM and MSE loss for the ANOViT model.
  * **`source/train.py`**: This module contains the core training logic, including the functions `Train_one_epoch`, `run_validation_epoch`, and `train_model`, which orchestrate the training and validation loops.
  * **`source/evaluate.py`**: This module is dedicated to evaluating the performance of the trained models. It contains functions to calculate various metrics such as AUROC, AUPR, accuracy, and F1-score at both the image and pixel level.
  * **`source/utils.py`**: This is a collection of utility functions that support the main workflow. Its key responsibilities include saving visualizations of the model's predictions, logging evaluation metrics to a CSV file, and creating a learning rate scheduler.
  * **`source/models/ANOVit.py`**: This file contains the implementation of the ANOVit model, a Vision Transformer (ViT) based approach to anomaly detection.
  * **`source/models/ADTR.py`**: This file contains the implementation of the ADTR model, an efficient anomaly detection transformer that leverages a pre-trained EfficientNet-B5 as a backbone.
  * **`source/models/ADTR_FPN.py`**: This file contains the implementation of the ADTR\_FPN model, an enhanced version of the ADTR model that incorporates a Feature Pyramid Network (FPN) to improve the detection of anomalies at different scales.
