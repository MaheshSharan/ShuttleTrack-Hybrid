import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import ShuttleTrackDataset
from models.shuttletrack import build_model_from_config
from torch.utils.tensorboard import SummaryWriter

# --- Utility functions ---
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)

# --- Loss functions ---
def compute_losses(pred, target):
    # pred, target: (B, T, 3) [visibility, x, y]
    vis_pred = pred[..., 0]
    vis_true = target[..., 0]
    xy_pred = pred[..., 1:]
    xy_true = target[..., 1:]
    # Visibility loss (BCE)
    bce = nn.BCEWithLogitsLoss()(vis_pred, vis_true)
    # Localization loss (MSE)
    mse = nn.MSELoss()(xy_pred, xy_true)
    # Trajectory continuity loss (L2 on differences)
    diff_pred = xy_pred[:, 1:, :] - xy_pred[:, :-1, :]
    diff_true = xy_true[:, 1:, :] - xy_true[:, :-1, :]
    smooth = nn.MSELoss()(diff_pred, diff_true)
    return bce, mse, smooth

# --- Training loop ---
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_bce, total_mse, total_smooth = 0, 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        pred = model(frames, diffs)
        bce, mse, smooth = compute_losses(pred, labels)
        loss = bce + mse + 0.1 * smooth
        loss.backward()
        optimizer.step()
        total_bce += bce.item() * frames.size(0)
        total_mse += mse.item() * frames.size(0)
        total_smooth += smooth.item() * frames.size(0)
    n = len(loader.dataset)
    return total_bce / n, total_mse / n, total_smooth / n

def validate(model, loader, device):
    model.eval()
    total_bce, total_mse, total_smooth = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid', leave=False):
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            labels = batch['labels'].to(device)
            pred = model(frames, diffs)
            bce, mse, smooth = compute_losses(pred, labels)
            total_bce += bce.item() * frames.size(0)
            total_mse += mse.item() * frames.size(0)
            total_smooth += smooth.item() * frames.size(0)
    n = len(loader.dataset)
    return total_bce / n, total_mse / n, total_smooth / n

# --- Main training script ---
def main():
    config = load_config('config/shuttletrack.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    # Data
    train_set = ShuttleTrackDataset('processed_data', split='Train', sequence_length=config['model']['sequence_length'], augment=True)
    valid_set = ShuttleTrackDataset('processed_data', split='valid', sequence_length=config['model']['sequence_length'], augment=False)
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    valid_loader = DataLoader(valid_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])
    # Model
    model = build_model_from_config(config).to(device)
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/shuttletrack')
    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f'\nEpoch {epoch}/{config["training"]["epochs"]}')
        train_bce, train_mse, train_smooth = train_one_epoch(model, train_loader, optimizer, device)
        val_bce, val_mse, val_smooth = validate(model, valid_loader, device)
        scheduler.step()
        val_loss = val_bce + val_mse + 0.1 * val_smooth
        print(f'Train BCE: {train_bce:.4f}, MSE: {train_mse:.4f}, Smooth: {train_smooth:.4f}')
        print(f'Valid BCE: {val_bce:.4f}, MSE: {val_mse:.4f}, Smooth: {val_smooth:.4f}')
        # TensorBoard logging
        writer.add_scalar('Loss/train_BCE', train_bce, epoch)
        writer.add_scalar('Loss/train_MSE', train_mse, epoch)
        writer.add_scalar('Loss/train_Smooth', train_smooth, epoch)
        writer.add_scalar('Loss/val_BCE', val_bce, epoch)
        writer.add_scalar('Loss/val_MSE', val_mse, epoch)
        writer.add_scalar('Loss/val_Smooth', val_smooth, epoch)
        writer.add_scalar('Loss/val_Total', val_loss, epoch)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_best.pth')
            print('  [*] Saved new best model!')
        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_last.pth')
    writer.close()

if __name__ == '__main__':
    main() 