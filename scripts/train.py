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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

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

def generate_human_insights(current_metrics, previous_metrics=None, best_metrics=None, epoch=0, num_epochs=50, num_epochs_without_improvement=0):
    """Generate human-readable insights about the training progress."""
    insights = []
    
    # If this is the first epoch, just provide basic info
    if previous_metrics is None:
        insights.append("🚀 Training started! Initial metrics captured.")
        return insights
    
    # Check for overall improvement in key metrics
    improved_val_metrics = 0
    total_key_metrics = 0
    
    # Key metrics to track (add weights to prioritize certain metrics)
    key_metric_weights = {
        'val_loss': 1.0,
        'val_visibility_f1': 2.0,  # Important for detection performance
        'val_within_5px': 2.0,     # Important for precise localization
        'val_distance_error': 1.5
    }
    
    for metric, weight in key_metric_weights.items():
        total_key_metrics += weight
        
        # Skip if metric not in both current and previous
        if metric not in current_metrics or metric not in previous_metrics:
            continue
            
        # For metrics where higher is better
        if 'f1' in metric or 'precision' in metric or 'recall' in metric or 'within' in metric:
            if current_metrics[metric] > previous_metrics[metric]:
                improved_val_metrics += weight
        # For metrics where lower is better
        else:
            if current_metrics[metric] < previous_metrics[metric]:
                improved_val_metrics += weight
    
    improvement_ratio = improved_val_metrics / total_key_metrics if total_key_metrics > 0 else 0
    
    # Add improvement insights
    if improvement_ratio >= 0.7:
        insights.append(f"✅ Good progress! Model is improving on {int(improvement_ratio * 100)}% of key metrics.")
    elif improvement_ratio >= 0.5:
        insights.append(f"🔄 Moderate improvement on {int(improvement_ratio * 100)}% of key metrics.")
    elif improvement_ratio > 0:
        insights.append(f"⚠️ Limited improvement (only {int(improvement_ratio * 100)}% of key metrics).")
    else:
        insights.append("❌ No improvement on validation metrics in this epoch.")
    
    # Check for overfitting
    train_val_gap = current_metrics['train_loss'] - current_metrics['val_loss']
    prev_train_val_gap = previous_metrics['train_loss'] - previous_metrics['val_loss']
    
    if train_val_gap < 0 and abs(train_val_gap) > abs(prev_train_val_gap):
        # Train loss higher than val loss and getting worse - model is struggling
        insights.append("⚠️ Model is struggling to learn, validation loss is better than training!")
    elif current_metrics['train_loss'] < previous_metrics['train_loss'] and current_metrics['val_loss'] > previous_metrics['val_loss']:
        # Train loss decreasing but val loss increasing - classic overfitting
        insights.append("⚠️ Warning: Possible overfitting! Training loss ↓ but validation loss ↑")
    
    # Check for improvement on specific metrics
    if current_metrics['val_visibility_f1'] > previous_metrics['val_visibility_f1']:
        insights.append(f"👁️ Detection accuracy improved: {current_metrics['val_visibility_f1']:.1%} (↑{(current_metrics['val_visibility_f1']-previous_metrics['val_visibility_f1'])*100:.1f}%)")
    
    if current_metrics['val_within_5px'] > previous_metrics['val_within_5px']:
        insights.append(f"🎯 Position accuracy improved: {current_metrics['val_within_5px']*100:.1f}% of predictions within 5px (↑{(current_metrics['val_within_5px']-previous_metrics['val_within_5px'])*100:.1f}%)")
    
    # Epochs without improvement
    if num_epochs_without_improvement > 0:
        if num_epochs_without_improvement >= 10:
            insights.append(f"⛔ No improvement for {num_epochs_without_improvement} epochs! Consider early stopping.")
        elif num_epochs_without_improvement >= 5:
            insights.append(f"⚠️ No improvement for {num_epochs_without_improvement} epochs. Consider adjusting learning rate.")
        else:
            insights.append(f"ℹ️ No improvement for {num_epochs_without_improvement} epochs.")
            
    # Compare to best metrics
    if best_metrics:
        best_f1_gap = best_metrics['val_visibility_f1'] - current_metrics['val_visibility_f1']
        if best_f1_gap > 0:
            insights.append(f"🔍 Still {best_f1_gap*100:.1f}% behind best detection accuracy.")
        
        # How close to 100% accuracy are we?
        best_within_5px = best_metrics['val_within_5px'] 
        insights.append(f"📈 Current best position accuracy: {best_within_5px*100:.1f}% (Target: >95%)")
        
    # Progress toward completion
    progress = epoch / num_epochs if num_epochs > 0 else 0
    if progress > 0.8:
        insights.append(f"🏁 Training {progress*100:.0f}% complete ({epoch}/{num_epochs} epochs)")
        
    return insights

# --- Loss functions ---
def compute_losses(pred, target, config):
    # pred: dict with 'heatmap' (B, T, 1, H, W), 'visibility' (B, T)
    # target: dict with 'heatmap' (B, T, H, W), 'visibility' (B, T)
    pred_heatmap = pred['heatmap'].squeeze(2)  # (B, T, H, W)
    true_heatmap = target['heatmap']  # (B, T, H, W)
    pred_vis = pred['visibility']  # (B, T)
    true_vis = target['visibility']  # (B, T)
    
    # Get epsilon for numerical stability
    epsilon = config.get('training', {}).get('numerical_stability', {}).get('epsilon', 1e-6)
    
    # Clamp predictions to prevent extreme values
    pred_heatmap = torch.clamp(pred_heatmap, min=-100.0, max=100.0)
    pred_vis = torch.clamp(pred_vis, min=-100.0, max=100.0)
    
    # Visibility loss (BCE)
    bce = nn.BCEWithLogitsLoss()(pred_vis, true_vis)
    
    # Heatmap loss (MSE)
    # Add epsilon to denominators for stability
    mse = nn.MSELoss()(pred_heatmap, true_heatmap)
    
    # Check for NaNs in loss components
    if torch.isnan(bce):
        print(f"[WARNING] NaN detected in BCE loss! Using default value.")
        bce = torch.tensor(1.0, device=bce.device)
    
    if torch.isnan(mse):
        print(f"[WARNING] NaN detected in MSE loss! Using default value.")
        mse = torch.tensor(1.0, device=mse.device)
    
    # Optionally, add smoothness loss on heatmap peaks (not implemented here)
    loss_weights = config.get('training', {}).get('loss_weights', {'bce': 1.0, 'mse': 1.0})
    bce_weight = loss_weights.get('bce', 1.0)
    mse_weight = loss_weights.get('mse', 1.0)
    total_loss = bce_weight * bce + mse_weight * mse
    return total_loss, bce, mse, torch.tensor(0.0, device=bce.device)

def has_nan(tensor):
    return torch.isnan(tensor).any().item()

def print_tensor_stats(name, tensor):
    print(f"  {name}: shape={tuple(tensor.shape)}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, has_nan={torch.isnan(tensor).any().item()}")

def save_batch_data(batch, pred, loss_dict, batch_idx, split='train'):
    """Save batch data to disk when NaNs are detected."""
    debug_dir = os.path.join('debug_dumps', f"{split}_batch_{batch_idx}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Prepare a report of which component had a NaN
    # loss_dict contains boolean flags: True if NaN, False otherwise.
    nan_source_report = {key: "Detected NaN" if value else "OK" for key, value in loss_dict.items()}
    
    # Save metadata
    metadata = {
        'batch_idx': batch_idx,
        'split': split,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'nan_source_component_status': nan_source_report,  # Changed from 'loss_values'
        'shapes': {
            'frames': tuple(batch['frames'].shape),
            'diffs': tuple(batch['diffs'].shape),
            'heatmap': tuple(batch['heatmap'].shape),
            'visibility': tuple(batch['visibility'].shape)
        }
    }
    
    if pred:
        metadata['shapes'].update({
            'pred_heatmap': tuple(pred['heatmap'].shape),
            'pred_visibility': tuple(pred['visibility'].shape)
        })
    
    with open(os.path.join(debug_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save tensors as numpy arrays
    torch.save(batch, os.path.join(debug_dir, 'batch.pt'))
    if pred:
        torch.save(pred, os.path.join(debug_dir, 'predictions.pt'))
    
    # Save sample visualizations (first sequence in batch)
    os.makedirs(os.path.join(debug_dir, 'visualizations'), exist_ok=True)
    
    # Visualize first sequence in batch
    for t in range(min(5, batch['frames'].shape[1])):  # Up to 5 frames in sequence
        # Save frame
        frame = batch['frames'][0, t].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(frame)
        plt.title(f"Frame {t}")
        plt.axis('off')
        
        # Save diff
        diff = batch['diffs'][0, t].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        plt.subplot(2, 2, 2)
        plt.imshow(diff)
        plt.title(f"Diff {t}")
        plt.axis('off')
        
        # Save ground truth heatmap
        gt_heatmap = batch['heatmap'][0, t].cpu().numpy()
        plt.subplot(2, 2, 3)
        plt.imshow(gt_heatmap, cmap='hot')
        plt.title(f"GT Heatmap {t} (vis={batch['visibility'][0, t].item():.2f})")
        plt.axis('off')
        
        # Save predicted heatmap if available
        if pred:
            pred_heatmap = pred['heatmap'][0, t, 0].cpu().numpy()
            plt.subplot(2, 2, 4)
            plt.imshow(pred_heatmap, cmap='hot')
            plt.title(f"Pred Heatmap {t} (vis={torch.sigmoid(pred['visibility'][0, t]).item():.2f})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, 'visualizations', f'frame_{t}.png'))
        plt.close()
    
    print(f"[DEBUG] Saved problematic batch data to {debug_dir}")

# --- Training loop ---
def train_one_epoch(model, loader, optimizer, device, config, debug=False):
    model.train()
    total_loss = 0
    total_bce, total_mse, total_smooth = 0, 0, 0
    skipped_batches = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        labels = {
            'heatmap': batch['heatmap'].to(device),
            'visibility': batch['visibility'].to(device)
        }
        if debug:
            print(f"[DEBUG][Train][Batch {batch_idx}]")
            print_tensor_stats('frames', frames)
            print_tensor_stats('diffs', diffs)
            print_tensor_stats('heatmap', labels['heatmap'])
            print_tensor_stats('visibility', labels['visibility'])
        # Check for NaNs in input
        if has_nan(frames) or has_nan(diffs) or has_nan(labels['heatmap']) or has_nan(labels['visibility']):
            print(f"[NaN WARNING][Train][Batch {batch_idx}] NaN detected in input. Skipping batch.")
            # Save the problematic batch data
            save_batch_data(batch, None, {'frames_nan': has_nan(frames), 'diffs_nan': has_nan(diffs), 
                            'heatmap_nan': has_nan(labels['heatmap']), 'visibility_nan': has_nan(labels['visibility'])},
                           batch_idx, split='train')
            skipped_batches += 1
            if debug:
                print("[DEBUG] Skipping batch due to NaN in input.")
            if debug:
                break
            continue
        optimizer.zero_grad()
        pred = model(frames, diffs)
        if debug:
            print_tensor_stats('pred_heatmap', pred['heatmap'])
            print_tensor_stats('pred_visibility', pred['visibility'])
        # Check for NaNs in model output
        if has_nan(pred['heatmap']) or has_nan(pred['visibility']):
            print(f"[NaN WARNING][Train][Batch {batch_idx}] NaN detected in model output. Skipping batch.")
            # Save the problematic batch and prediction data
            save_batch_data(batch, pred, {'pred_heatmap_nan': has_nan(pred['heatmap']), 
                            'pred_visibility_nan': has_nan(pred['visibility'])},
                           batch_idx, split='train')
            skipped_batches += 1
            if debug:
                print("[DEBUG] Skipping batch due to NaN in model output.")
            if debug:
                break
            continue
        loss, bce, mse, smooth = compute_losses(pred, labels, config)
        if debug:
            print(f"  loss: {loss.item():.4f}, bce: {bce.item():.4f}, mse: {mse.item():.4f}, smooth: {smooth.item():.4f}")
        # Check for NaNs in loss
        if (has_nan(loss) or has_nan(bce) or has_nan(mse) or has_nan(smooth)):
            print(f"[NaN WARNING][Train][Batch {batch_idx}] NaN detected in loss. Skipping batch.")
            # Save the problematic batch, prediction, and loss data
            save_batch_data(batch, pred, {'loss_nan': has_nan(loss), 'bce_nan': has_nan(bce), 
                            'mse_nan': has_nan(mse), 'smooth_nan': has_nan(smooth)},
                           batch_idx, split='train')
            skipped_batches += 1
            if debug:
                print("[DEBUG] Skipping batch due to NaN in loss.")
            if debug:
                break
            continue
        loss.backward()
        if 'gradient_clip_val' in config.get('training', {}):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
        optimizer.step()
        total_loss += loss.item() * frames.size(0)
        total_bce += bce.item() * frames.size(0)
        total_mse += mse.item() * frames.size(0)
        total_smooth += smooth.item() * frames.size(0)
        if debug:
            print("[DEBUG] Completed 1 batch in debug mode. Exiting train_one_epoch.")
            break
    n = len(loader.dataset)
    if skipped_batches > 0:
        print(f"[NaN WARNING][Train] Skipped {skipped_batches} batches due to NaNs.")
    return total_loss / n, total_bce / n, total_mse / n, total_smooth / n

def validate(model, loader, device, config, debug=False):
    model.eval()
    total_loss = 0
    total_bce, total_mse, total_smooth = 0, 0, 0
    skipped_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Valid', leave=False)):
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            labels = {
                'heatmap': batch['heatmap'].to(device),
                'visibility': batch['visibility'].to(device)
            }
            if debug:
                print(f"[DEBUG][Valid][Batch {batch_idx}]")
                print_tensor_stats('frames', frames)
                print_tensor_stats('diffs', diffs)
                print_tensor_stats('heatmap', labels['heatmap'])
                print_tensor_stats('visibility', labels['visibility'])
            # Check for NaNs in input
            if has_nan(frames) or has_nan(diffs) or has_nan(labels['heatmap']) or has_nan(labels['visibility']):
                print(f"[NaN WARNING][Valid][Batch {batch_idx}] NaN detected in input. Skipping batch.")
                # Save the problematic batch data
                save_batch_data(batch, None, {'frames_nan': has_nan(frames), 'diffs_nan': has_nan(diffs), 
                               'heatmap_nan': has_nan(labels['heatmap']), 'visibility_nan': has_nan(labels['visibility'])},
                              batch_idx, split='valid')
                skipped_batches += 1
                if debug:
                    print("[DEBUG] Skipping batch due to NaN in input.")
                if debug:
                    break
                continue
            pred = model(frames, diffs)
            if debug:
                print_tensor_stats('pred_heatmap', pred['heatmap'])
                print_tensor_stats('pred_visibility', pred['visibility'])
            # Check for NaNs in model output
            if has_nan(pred['heatmap']) or has_nan(pred['visibility']):
                print(f"[NaN WARNING][Valid][Batch {batch_idx}] NaN detected in model output. Skipping batch.")
                # Save the problematic batch and prediction data
                save_batch_data(batch, pred, {'pred_heatmap_nan': has_nan(pred['heatmap']), 
                               'pred_visibility_nan': has_nan(pred['visibility'])},
                              batch_idx, split='valid')
                skipped_batches += 1
                if debug:
                    print("[DEBUG] Skipping batch due to NaN in model output.")
                if debug:
                    break
                continue
            loss, bce, mse, smooth = compute_losses(pred, labels, config)
            if debug:
                print(f"  loss: {loss.item():.4f}, bce: {bce.item():.4f}, mse: {mse.item():.4f}, smooth: {smooth.item():.4f}")
            # Check for NaNs in loss
            if (has_nan(loss) or has_nan(bce) or has_nan(mse) or has_nan(smooth)):
                print(f"[NaN WARNING][Valid][Batch {batch_idx}] NaN detected in loss. Skipping batch.")
                # Save the problematic batch, prediction, and loss data
                save_batch_data(batch, pred, {'loss_nan': has_nan(loss), 'bce_nan': has_nan(bce), 
                               'mse_nan': has_nan(mse), 'smooth_nan': has_nan(smooth)},
                              batch_idx, split='valid')
                skipped_batches += 1
                if debug:
                    print("[DEBUG] Skipping batch due to NaN in loss.")
                if debug:
                    break
                continue
            total_loss += loss.item() * frames.size(0)
            total_bce += bce.item() * frames.size(0)
            total_mse += mse.item() * frames.size(0)
            total_smooth += smooth.item() * frames.size(0)
            if debug:
                print("[DEBUG] Completed 1 batch in debug mode. Exiting validate.")
                break
    n = len(loader.dataset)
    if skipped_batches > 0:
        print(f"[NaN WARNING][Valid] Skipped {skipped_batches} batches due to NaNs.")
    return total_loss / n, total_bce / n, total_mse / n, total_smooth / n

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# --- Main training script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    parser.add_argument('--debug', action='store_true', help='Run only 1 batch for train and valid, print details, and exit after 1 epoch')
    args = parser.parse_args()

    # Create debug directories if they don't exist
    os.makedirs('debug_dumps', exist_ok=True)

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

    train_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'],
        split='Train',
        sequence_length=config['model']['sequence_length'],
        augment=True,
        input_size=input_size_tuple,
        augmentation_config=config['training'].get('augmentations', {})
    )
    valid_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'],
        split='valid',
        sequence_length=config['model']['sequence_length'],
        augment=False,
        input_size=input_size_tuple,
        augmentation_config=config['training'].get('augmentations', {})
    )
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

    # For tracking metrics over time
    previous_metrics = None
    epochs_without_improvement = 0
    best_val_loss = float('inf')
    num_epochs = config['training']['epochs']

    # Record start time
    training_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, (start_epoch+1) if args.debug else (config['training']['epochs'] + 1)):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}/{config["training"]["epochs"]}')
        
        # Training
        train_loss, train_bce, train_mse, train_smooth = train_one_epoch(model, train_loader, optimizer, device, config, debug=args.debug)
        print("Finished training epoch, starting validation...")
        # Validation
        val_loss, val_bce, val_mse, val_smooth = validate(model, valid_loader, device, config, debug=args.debug)
        print("Finished validation, starting evaluation on valid set...")
        val_metrics = evaluate(model, valid_loader, device)
        print("Finished all evaluations, proceeding to logging and checkpointing...")
        
        scheduler.step()
        
        # Update best metrics
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
            
        if val_loss < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_loss
            best_metrics['val_bce'] = val_bce
            best_metrics['val_mse'] = val_mse
            best_metrics['epoch'] = epoch
        
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
        
        # Compile current metrics into one dict for the insights generator
        current_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_visibility_f1': val_metrics['visibility_f1'],
            'val_within_5px': val_metrics['within_5px'],
            'val_within_10px': val_metrics['within_10px'],
            'val_distance_error': val_metrics['distance_error']
        }
        
        # Generate human-readable insights
        insights = generate_human_insights(
            current_metrics, 
            previous_metrics, 
            best_metrics, 
            epoch, 
            num_epochs,
            epochs_without_improvement
        )
        
        # Store current metrics for next epoch comparison
        previous_metrics = current_metrics
        
        # Epoch time calculation
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        # Log to tensorboard
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Loss/train_BCE', train_bce, epoch)
        writer.add_scalar('Loss/train_MSE', train_mse, epoch)
        writer.add_scalar('Loss/train_Smooth', train_smooth, epoch)
        writer.add_scalar('Loss/val_total', val_loss, epoch)
        writer.add_scalar('Loss/val_BCE', val_bce, epoch)
        writer.add_scalar('Loss/val_MSE', val_mse, epoch)
        writer.add_scalar('Loss/val_Smooth', val_smooth, epoch)
        
        for key, value in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{key}', value, epoch)
        
        # Print initial loss info
        print(f'Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, MSE: {train_mse:.4f}, Smooth: {train_smooth:.4f})')
        print(f'Valid Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, MSE: {val_mse:.4f}, Smooth: {val_smooth:.4f})')
        
        # Generate metrics table
        headers = ["Metric", "Train", "Valid", "Best"]
        table_data = [
            ["Total Loss", f"{train_loss:.4f}", f"{val_loss:.4f}", f"{best_metrics['val_loss']:.4f}"],
            ["BCE Loss", f"{train_bce:.4f}", f"{val_bce:.4f}", f"{best_metrics['val_bce']:.4f}"],
            ["MSE Loss", f"{train_mse:.4f}", f"{val_mse:.4f}", f"{best_metrics['val_mse']:.4f}"],
            ["Distance Error", "-", f"{val_metrics['distance_error']:.4f}", f"{best_metrics['val_distance_error']:.4f}"],
            ["Visibility F1", "-", f"{val_metrics['visibility_f1']:.4f}", f"{best_metrics['val_visibility_f1']:.4f}"],
            ["Precision", "-", f"{val_metrics['precision']:.4f}", f"{best_metrics['val_precision']:.4f}"],
            ["Recall", "-", f"{val_metrics['recall']:.4f}", f"{best_metrics['val_recall']:.4f}"],
            ["Within 5px", "-", f"{val_metrics['within_5px']*100:.1f}%", f"{best_metrics['val_within_5px']*100:.1f}%"],
            ["Within 10px", "-", f"{val_metrics['within_10px']*100:.1f}%", f"{best_metrics['val_within_10px']*100:.1f}%"]
        ]
        
        # Print metrics table
        print(f"\nEpoch {epoch} Summary | Time: {format_time(epoch_time)} | Total: {format_time(total_time)}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print human-readable insights
        print("\n📊 Training Insights:")
        for insight in insights:
            print(f"  {insight}")
        print("")
        
        # Save best model
        if is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_best.pth')
            print('  [*] Saved new best model!')

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/checkpoint_last.pth')
        print(f'  [*] Saved checkpoint for epoch {epoch}.')

        if args.debug:
            print("[DEBUG] Exiting after 1 epoch in debug mode.")
            break

    writer.close()

if __name__ == '__main__':
    main()
