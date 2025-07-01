"""
Training script for the model.
"""

from tqdm import tqdm

def Train_one_epoch(optimizer, model, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    avg_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, _, _ in progress_bar:
        images = images.to(device)

        optimizer.zero_grad()

        original_features, reconstructed_features = model(images)
        loss = criterion(original_features, reconstructed_features)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def train_model(model, dataloader, optimizer, criterion, device, epochs, scheduler=None):
    for epoch in range(epochs):
        avg_loss = Train_one_epoch(optimizer, model, dataloader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} | Average Training Loss: {avg_loss:.6f}")
        if scheduler is not None:
            scheduler.step()