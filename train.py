import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
from model import SingleObjectKeypointDetector, WingLossWithVisibility
from dataset import KeypointDataset, load_yolo_dataset, get_train_transforms, get_val_transforms

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, leave=True)
    for images, targets, masks in loop:
        images, targets, masks = images.to(device), targets.to(device), masks.to(device)

        preds = model(images)
        loss, loss_coord, loss_vis = criterion(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Train Loss: {loss.item():.5f} (Coord: {loss_coord:.4f}, Vis: {loss_vis:.4f})")
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    if len(loader) == 0:
        return 0.0
    with torch.no_grad():
        for images, targets, masks in loader:
            images, targets, masks = images.to(device), targets.to(device), masks.to(device)
            preds = model(images)
            loss, _, _ = criterion(preds, targets, masks)
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    # Setup
    os.makedirs(config.CHECKPOINT_SAVE_DIR, exist_ok=True)
    model = SingleObjectKeypointDetector(num_keypoints=config.NUM_KEYPOINTS).to(config.DEVICE)
    criterion = WingLossWithVisibility()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # We will handle the LR scheduling manually, so we remove the scheduler object.

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if specified
    resume_path = config.RESUME_CHECKPOINT or os.path.join(config.CHECKPOINT_SAVE_DIR, config.LAST_MODEL_NAME)
    if os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}")
        try:
            checkpoint = torch.load(resume_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch} with best validation loss {best_val_loss:.5f}")
        except (KeyError, TypeError) as e:
            print(f"Could not load checkpoint from {resume_path}. It might be from an old version. Starting fresh.")
            print(f"Error details: {e}")


    # Data Loading
    print("Loading data...")
    train_samples = load_yolo_dataset(config.DATA_DIR, split='train')
    val_samples = load_yolo_dataset(config.DATA_DIR, split='val')
    
    if not train_samples:
        print("Error: No training data found. Please check the `DATA_DIR` in `config.py` and your dataset structure.")
        return

    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")

    train_dataset = KeypointDataset(train_samples, transform=get_train_transforms())
    val_dataset = KeypointDataset(val_samples, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Training Loop
    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")
    for epoch in range(start_epoch, config.EPOCHS):
        # Manual LR scheduling
        if epoch < config.WARMUP_EPOCHS:
            # Linear warmup
            warmup_factor = (epoch + 1) / config.WARMUP_EPOCHS
            new_lr = config.LEARNING_RATE * warmup_factor
        else:
            # Cosine annealing phase
            progress = (epoch - config.WARMUP_EPOCHS) / (config.EPOCHS - config.WARMUP_EPOCHS)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            new_lr = config.MIN_LR + (config.LEARNING_RATE - config.MIN_LR) * cosine_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        val_loss = 0.0
        if val_loader:
            val_loss = validate(model, val_loader, criterion, config.DEVICE)
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {train_loss:.5f} | LR: {current_lr:.6f}")

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(config.CHECKPOINT_SAVE_DIR, config.LAST_MODEL_NAME))

        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(config.CHECKPOINT_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, periodic_path)
            print(f"Periodic checkpoint saved: {periodic_path}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save just the model state dict for the best model
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_SAVE_DIR, config.BEST_MODEL_NAME))
            print(f"New best model saved with validation loss: {best_val_loss:.5f}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Find the best model at: {os.path.join(config.CHECKPOINT_SAVE_DIR, config.BEST_MODEL_NAME)}")
    print(f"To export the model for deployment, run: python export.py")


if __name__ == '__main__':
    main()
