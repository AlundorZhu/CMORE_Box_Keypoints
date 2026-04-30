import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
from model import SingleObjectKeypointDetector, WingLossWithVisibility
from dataset import KeypointDataset, DataPrefetcher, load_yolo_dataset, get_train_transforms, get_val_transforms

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

    # Phase 1: freeze backbone, train head only
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = (list(model.compress_last.parameters()) +
                   list(model.compress_prev.parameters()) +
                   list(model.head.parameters()))
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0.0,                  'weight_decay': config.WEIGHT_DECAY},
        {'params': head_params,                 'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY},
    ])

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

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

    train_dataset = KeypointDataset(train_samples, transform=get_train_transforms(), flip_pairs=config.FLIP_PAIRS)
    val_dataset = KeypointDataset(val_samples, transform=get_val_transforms())

    train_loader = DataPrefetcher(DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True, persistent_workers=True), config.DEVICE)
    val_loader = DataPrefetcher(DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True, persistent_workers=True), config.DEVICE)

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

        # Phase 2: unfreeze backbone at FREEZE_BACKBONE_EPOCHS
        if epoch == config.FREEZE_BACKBONE_EPOCHS:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        optimizer.param_groups[0]['lr'] = new_lr * config.BACKBONE_LR_SCALE if epoch >= config.FREEZE_BACKBONE_EPOCHS else 0.0
        optimizer.param_groups[1]['lr'] = new_lr
        current_lr = optimizer.param_groups[1]['lr']

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
            epochs_without_improvement = 0
            # Save just the model state dict for the best model
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_SAVE_DIR, config.BEST_MODEL_NAME))
            print(f"New best model saved with validation loss: {best_val_loss:.5f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping: no improvement for {config.EARLY_STOPPING_PATIENCE} consecutive epochs.")
                break

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Find the best model at: {os.path.join(config.CHECKPOINT_SAVE_DIR, config.BEST_MODEL_NAME)}")
    print(f"To export the model for deployment, run: python export.py")


if __name__ == '__main__':
    main()
