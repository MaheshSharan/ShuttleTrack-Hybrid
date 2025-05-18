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
import argparse
from utils.eval import evaluate
import time
from tabulate import tabulate
from datetime import timedelta

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

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# --- Main training script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    args = parser.parse_args()

    config = load_config('config/shuttletrack.yaml')

    if args.kaggle:
        config['data']['processed_dataset_path'] = '/kaggle/input/shuttletrack-processed-data/processed_data'
        config['data']['raw_dataset_path'] = '/kaggle/input/shuttlecockframedataset'

    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f'Using device: {device}')

    # Data
    input_size_cfg = config['model']['input_size']
    if isinstance(input_size_cfg, int):
        input_size_tuple = (input_size_cfg, input_size_cfg)
    else:
        input_size_tuple = tuple(input_size_cfg)

    train_set = ShuttleTrackDataset(config['data']['processed_dataset_path'], split='Train', sequence_length=config['model']['sequence_length'], augment=True, input_size=input_size_tuple)
    valid_set = ShuttleTrackDataset(config['data']['processed_dataset_path'], split='valid', sequence_length=config['model']['sequence_length'], augment=False, input_size=input_size_tuple)
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=(device.type == 'cuda'))
    valid_loader = DataLoader(valid_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=(device.type == 'cuda'))

    # Model
    model = build_model_from_config(config).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/shuttletrack')

    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)

    # --- Resume logic ---
    start_epoch = 1
    checkpoint_path = 'checkpoints/checkpoint_last.pth'
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f"[INFO] Resumed training from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
    else:
        best_val_loss = float('inf')
        print("[INFO] No checkpoint found. Starting training from scratch.")

    # Initialize best metrics tracking
    best_metrics = {
        'val_loss': float('inf'),
        'val_bce': float('inf'),
        'val_mse': float('inf'), 
        'val_distance_error': float('inf'),
        'val_visibility_f1': 0,
        'val_precision': 0,
        'val_recall': 0,
        'val_within_5px': 0,
        'val_within_10px': 0,
        'epoch': 0
    }

    # Record start time
    training_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}/{config["training"]["epochs"]}')
        
        # Training
        train_bce, train_mse, train_smooth = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validation
        val_bce, val_mse, val_smooth = validate(model, valid_loader, device)
        
        # Calculate and log additional metrics
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, valid_loader, device)
        
        val_loss = val_bce + val_mse + 0.1 * val_smooth
        scheduler.step()
        
        # Update best metrics
        is_best = False
        if val_loss < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_loss
            best_metrics['val_bce'] = val_bce
            best_metrics['val_mse'] = val_mse
            best_metrics['epoch'] = epoch
            is_best = True
        
        for key, value in val_metrics.items():
            val_key = f'val_{key}'
            if val_key not in best_metrics:
                best_metrics[val_key] = -float('inf') if 'f1' in key or 'within' in key or 'precision' in key or 'recall' in key else float('inf')
            
            # Update if better (higher is better for F1, precision, recall, within_X)
            if ('f1' in key or 'within' in key or 'precision' in key or 'recall' in key):
                if value > best_metrics[val_key]:
                    best_metrics[val_key] = value
                    is_best = True
            # Lower is better for distance_error
            elif value < best_metrics[val_key]:
                best_metrics[val_key] = value
                is_best = True
        
        # Epoch time calculation
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        # Log to tensorboard
        writer.add_scalar('Loss/train_BCE', train_bce, epoch)
        writer.add_scalar('Loss/train_MSE', train_mse, epoch)
        writer.add_scalar('Loss/train_Smooth', train_smooth, epoch)
        writer.add_scalar('Loss/val_BCE', val_bce, epoch)
        writer.add_scalar('Loss/val_MSE', val_mse, epoch)
        writer.add_scalar('Loss/val_Smooth', val_smooth, epoch)
        writer.add_scalar('Loss/val_Total', val_loss, epoch)
        
        for key, value in train_metrics.items():
            writer.add_scalar(f'Metrics/train_{key}', value, epoch)
        
        for key, value in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{key}', value, epoch)
        
        # Print initial loss info
        print(f'Train BCE: {train_bce:.4f}, MSE: {train_mse:.4f}, Smooth: {train_smooth:.4f}')
        print(f'Valid BCE: {val_bce:.4f}, MSE: {val_mse:.4f}, Smooth: {val_smooth:.4f}')
        
        # Generate metrics table
        headers = ["Metric", "Train", "Valid", "Best"]
        table_data = [
            ["BCE Loss", f"{train_bce:.4f}", f"{val_bce:.4f}", f"{best_metrics['val_bce']:.4f}"],
            ["MSE Loss", f"{train_mse:.4f}", f"{val_mse:.4f}", f"{best_metrics['val_mse']:.4f}"],
            ["Distance Error", f"{train_metrics['distance_error']:.4f}", f"{val_metrics['distance_error']:.4f}", f"{best_metrics['val_distance_error']:.4f}"],
            ["Visibility F1", f"{train_metrics['visibility_f1']:.4f}", f"{val_metrics['visibility_f1']:.4f}", f"{best_metrics['val_visibility_f1']:.4f}"],
            ["Precision", f"{train_metrics['precision']:.4f}", f"{val_metrics['precision']:.4f}", f"{best_metrics['val_precision']:.4f}"],
            ["Recall", f"{train_metrics['recall']:.4f}", f"{val_metrics['recall']:.4f}", f"{best_metrics['val_recall']:.4f}"],
            ["Within 5px", f"{train_metrics['within_5px']*100:.1f}%", f"{val_metrics['within_5px']*100:.1f}%", f"{best_metrics['val_within_5px']*100:.1f}%"],
            ["Within 10px", f"{train_metrics['within_10px']*100:.1f}%", f"{val_metrics['within_10px']*100:.1f}%", f"{best_metrics['val_within_10px']*100:.1f}%"]
        ]
        
        # Print metrics table
        print(f"\nEpoch {epoch} Summary | Time: {format_time(epoch_time)} | Total: {format_time(total_time)}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Save best model
        if is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_best.pth')
            print('  [*] Saved new best model!')

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_last.pth')
        print(f'  [*] Saved checkpoint for epoch {epoch}.')

    writer.close()

if __name__ == '__main__':
    main()
