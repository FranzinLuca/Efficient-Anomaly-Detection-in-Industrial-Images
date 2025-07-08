"""
Training script for the model.
"""

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch
import config
import os

def Train_one_epoch(optimizer, model, dataloader_train, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    avg_loss = 0.0
    progress_bar = tqdm(dataloader_train, desc="Training")

    for images, _, _ in progress_bar:
        images = images.to(device)

        optimizer.zero_grad()

        original_features, reconstructed_features = model(images)
        loss = criterion(reconstructed_features, original_features)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = running_loss / len(dataloader_train)
    return avg_loss

def run_validation_epoch(model, val_loader, loss_fn, device):
    
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(device)
            original_features, reconstructed_features = model(imgs)
            loss = loss_fn(reconstructed_features, original_features)
            running_loss += loss.item()
    average_loss = running_loss / len(val_loader)
    return average_loss

def train_model(model, dataloader_train, dataloader_val, optimizer, criterion, scheduler=None):
    epochs = config.EPOCHS
    device = config.DEVICE
    
    for epoch in range(epochs):
        train_loss = Train_one_epoch(optimizer, model, dataloader_train, criterion, device, scheduler)
        if dataloader_val is not None:
            val_loss = run_validation_epoch(model, dataloader_val, criterion, device)
            print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")

        else:
            print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_loss:.6f}")