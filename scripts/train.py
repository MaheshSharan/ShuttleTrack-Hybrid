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
from models.shuttletrack import build_model_from_config, extract_coordinates_from_heatmap
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils.eval import evaluate
import time
from tabulate import tabulate
from datetime import timedelta
import numpy as np

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
def compute_losses(pred_dict, target, config, epoch=None, max_epochs=None):
    """
    Compute all losses for the ShuttleTrack model.
    Fully implements Section 4: Loss Function Engineering with:
    - Focal Loss for visibility
    - Dynamic loss weighting
    - Uncertainty modeling
    - Smooth L1 Loss
    
    Args:
        pred_dict: Dictionary containing model predictions ('visibility', 'heatmap', 'uncertainty' if available)
        target: Target tensor with shape (B, T, 3) containing [visibility, x, y]
        config: Training configuration
        epoch: Current epoch (for dynamic loss weighting)
        max_epochs: Maximum number of epochs (for dynamic loss weighting)
    """
    # pred_dict contains 'visibility' and 'heatmap' keys
    # visibility: (B, T) logits
    # heatmap: (B, T, H, W) heatmaps
    # uncertainty: (B, T, 2) uncertainty values for x,y (if enabled)
    
    # target: (B, T, 3) [visibility, x, y]
    vis_true = target[..., 0]
    xy_true = target[..., 1:]  # Normalized coordinates [0,1]
    batch_size, seq_len = vis_true.shape
    
    # Check if we have uncertainty predictions
    has_uncertainty = 'uncertainty' in pred_dict
    
    # ======= 1. Visibility Loss (Focal BCE) =======
    vis_pred = pred_dict['visibility']
    
    # Calculate Binary Cross Entropy (raw)
    bce_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        vis_pred, vis_true, reduction='none'
    )
    
    # Apply focal loss to focus on hard examples
    # Using gamma=2 as in the original Focal Loss paper
    gamma = config.get('training', {}).get('focal_gamma', 2.0)
    focal_weight = (1 - torch.sigmoid(vis_pred) * vis_true - (1 - vis_true) * (1 - torch.sigmoid(vis_pred))) ** gamma
    focal_bce = (bce_loss_raw * focal_weight).mean()
    
    # ======= 2. Create target heatmaps for position prediction =======
    device = target.device
    heatmap_size = pred_dict['heatmap'].shape[-1]
    
    # Convert normalized target coordinates to heatmap pixel positions
    target_heatmaps = torch.zeros_like(pred_dict['heatmap'])  # (B, T, H, W)
    
    for b in range(batch_size):
        for t in range(seq_len):
            if vis_true[b, t] > 0:  # Only create target heatmap for visible frames
                # Convert normalized coordinates to heatmap pixel indices
                x_norm, y_norm = xy_true[b, t]
                cx = int(x_norm * heatmap_size)
                cy = int(y_norm * heatmap_size)
                
                # Keep coordinates within bounds
                cx = max(0, min(cx, heatmap_size - 1))
                cy = max(0, min(cy, heatmap_size - 1))
                
                # Set peak at target position
                target_heatmaps[b, t, cy, cx] = 1.0
                
                # Apply Gaussian blur (approximated with a simple kernel)
                sigma = 1.0
                radius = int(3 * sigma)
                for y in range(max(0, cy - radius), min(heatmap_size, cy + radius + 1)):
                    for x in range(max(0, cx - radius), min(heatmap_size, cx + radius + 1)):
                        if (x, y) != (cx, cy):
                            dist_squared = (x - cx)**2 + (y - cy)**2
                            weight = torch.exp(-dist_squared / (2 * sigma**2))
                            target_heatmaps[b, t, y, x] = weight
    
    # Normalize each heatmap
    max_values = target_heatmaps.reshape(batch_size, seq_len, -1).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    target_heatmaps = target_heatmaps / (max_values + 1e-8)
    
    # ======= 3. Heatmap Loss (MSE) =======
    heatmap_pred = pred_dict['heatmap']
    
    # Apply visibility mask to only compute loss for visible frames
    vis_mask = vis_true.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    masked_heatmap_pred = heatmap_pred * vis_mask
    masked_target_heatmaps = target_heatmaps * vis_mask
    
    mse = nn.MSELoss()(masked_heatmap_pred, masked_target_heatmaps)
    
    # ======= 4. Focal Loss for heatmap (to focus on hard examples) =======
    alpha = 2.0
    focal_weight = (1 - masked_heatmap_pred)**alpha * masked_target_heatmaps
    focal_loss = nn.MSELoss()(masked_heatmap_pred * focal_weight, masked_target_heatmaps * focal_weight)
    
    # ======= 5. Extract predicted coordinates for trajectory loss =======
    pred_coords, pred_conf = extract_coordinates_from_heatmap(heatmap_pred)  # (B, T, 2)
    
    # ======= 6. Smooth L1 Loss for direct coordinate regression =======
    # This helps with outliers in the coordinate predictions
    smooth_l1_loss = torch.nn.functional.smooth_l1_loss(
        pred_coords * vis_true.unsqueeze(-1),  # Only consider visible frames
        xy_true * vis_true.unsqueeze(-1),
        beta=0.1,  # Controls the smoothness, smaller values make it more like L1
        reduction='sum'
    ) / (torch.sum(vis_true) * 2 + 1e-8)  # Normalize by number of visible points
    
    # ======= 7. Trajectory continuity loss (L2 on differences) =======
    diff_pred = pred_coords[:, 1:, :] - pred_coords[:, :-1, :]
    diff_true = xy_true[:, 1:, :] - xy_true[:, :-1, :]
    
    # Mask for consecutive visible frames
    vis_pairs = vis_true[:, :-1] * vis_true[:, 1:]  # (B, T-1)
    
    smooth_loss = torch.sum((diff_pred - diff_true)**2 * vis_pairs.unsqueeze(-1)) 
    smooth_loss = smooth_loss / (torch.sum(vis_pairs) * 2 + 1e-8)  # Normalize by number of visible pairs
    
    # ======= 8. Uncertainty Loss (if enabled) =======
    uncertainty_loss = 0.0
    if has_uncertainty:
        uncertainty = pred_dict['uncertainty']  # (B, T, 2) - x,y uncertainty
        
        # Apply visibility mask to only consider visible points
        vis_mask_xy = vis_true.unsqueeze(-1)  # (B, T, 1)
        masked_uncertainty = uncertainty * vis_mask_xy
        
        # Calculate Euclidean distance between predicted and true coordinates
        coord_error = torch.sqrt(torch.sum((pred_coords - xy_true)**2, dim=-1, keepdim=True) + 1e-8)  # (B, T, 1)
        masked_coord_error = coord_error * vis_mask_xy
        
        # Implement uncertainty loss (negative log likelihood with uncertainty weighting)
        # Loss = error^2 / (2 * sigma^2) + log(sigma)
        # This encourages the model to predict higher uncertainty for points with high error
        nll_loss = (masked_coord_error**2 / (2 * masked_uncertainty + 1e-8) + torch.log(masked_uncertainty + 1e-8))
        uncertainty_loss = torch.sum(nll_loss) / (torch.sum(vis_mask_xy) + 1e-8)
    
    # ======= 9. Dynamic Loss Weighting (if epoch is provided) =======
    if epoch is not None and max_epochs is not None:
        # Implement dynamic loss weighting based on training progress
        progress = epoch / max_epochs
        
        # Gradually increase weight on trajectory smoothness as training progresses
        smooth_weight_factor = min(1.0, 0.5 + progress)  # Ranges from 0.5 to 1.0
        
        # Gradually decrease focal loss weight as training stabilizes
        focal_weight_factor = max(0.3, 1.0 - progress * 0.7)  # Ranges from 1.0 to 0.3
        
        # Gradually increase uncertainty weight as training progresses
        uncertainty_weight_factor = min(1.0, progress * 2)  # Ranges from 0 to 1.0
    else:
        # Use default factors if epoch information not provided
        smooth_weight_factor = 1.0
        focal_weight_factor = 1.0
        uncertainty_weight_factor = 1.0
    
    # ======= 10. Get loss weights from config =======
    loss_weights = config.get('training', {}).get('loss_weights', {
        'bce': 1.0, 
        'focal_bce': 1.0,
        'mse': 1.0, 
        'focal': 0.5, 
        'smooth': 0.1,
        'smooth_l1': 0.5,
        'uncertainty': 0.2
    })
    
    bce_weight = loss_weights.get('bce', 1.0)
    focal_bce_weight = loss_weights.get('focal_bce', 1.0) * focal_weight_factor
    mse_weight = loss_weights.get('mse', 1.0)
    focal_weight = loss_weights.get('focal', 0.5) * focal_weight_factor
    smooth_weight = loss_weights.get('smooth', 0.1) * smooth_weight_factor
    smooth_l1_weight = loss_weights.get('smooth_l1', 0.5)
    uncertainty_weight = loss_weights.get('uncertainty', 0.2) * uncertainty_weight_factor if has_uncertainty else 0.0
    
    # ======= 11. Calculate total loss =======
    total_loss = (
        focal_bce_weight * focal_bce +  # Focal loss for visibility
        mse_weight * mse +              # MSE loss for heatmap regression
        focal_weight * focal_loss +     # Focal loss for heatmaps
        smooth_weight * smooth_loss +   # Trajectory smoothness
        smooth_l1_weight * smooth_l1_loss + # Smooth L1 for direct coordinate regression
        uncertainty_weight * uncertainty_loss  # Uncertainty loss (if enabled)
    )
    
    return total_loss, focal_bce, mse, smooth_loss

# --- Training loop ---
def train_one_epoch(model, loader, optimizer, device, config, epoch=None, max_epochs=None, batch_losses=None, batch_indices=None):
    model.train()
    total_loss = 0
    total_bce, total_mse, total_smooth = 0, 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        flows = batch.get('flows')
        if flows is not None:
            flows = flows.to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        
        pred = model(frames, diffs, flows)
        loss, bce, mse, smooth = compute_losses(pred, labels, config, epoch, max_epochs)
        loss.backward()
        
        # Apply gradient clipping if enabled
        if 'gradient_clip_val' in config.get('training', {}):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
            
        optimizer.step()
        total_loss += loss.item() * frames.size(0)
        total_bce += bce.item() * frames.size(0)
        total_mse += mse.item() * frames.size(0)
        total_smooth += smooth.item() * frames.size(0)
        
        if batch_losses is not None and batch_indices is not None:
            batch_losses.append(loss.item())
            batch_indices.append(batch['index'].item())
    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_mse / n, total_smooth / n

def validate(model, loader, device, config, epoch=None, max_epochs=None):
    model.eval()
    total_loss = 0
    total_bce, total_mse, total_smooth = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid', leave=False):
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            flows = batch.get('flows')
            if flows is not None:
                flows = flows.to(device)
            labels = batch['labels'].to(device)
            
            pred = model(frames, diffs, flows)
            loss, bce, mse, smooth = compute_losses(pred, labels, config, epoch, max_epochs)
            total_loss += loss.item() * frames.size(0)
            total_bce += bce.item() * frames.size(0)
            total_mse += mse.item() * frames.size(0)
            total_smooth += smooth.item() * frames.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_mse / n, total_smooth / n

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# --- Main training script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--hard-mining', action='store_true', help='Enable hard example mining')
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

    # Set initial curriculum difficulty if enabled
    # Initially disabled for first run to ensure all samples are loaded
    initial_difficulty = None  # 'easy' if args.curriculum else None
    max_curriculum_epochs = config['training'].get('curriculum_epochs', 20) if args.curriculum else 0
    
    # Enhanced dataset with data augmentation techniques
    train_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'], 
        split='Train', 
        sequence_length=config['model']['sequence_length'], 
        augment=True, 
        input_size=input_size_tuple,
        use_optical_flow=config['model'].get('use_optical_flow', True),
        use_heatmaps=True,
        difficulty_level=initial_difficulty,
        mixup_prob=config['training'].get('mixup_prob', 0.3),
        blur_prob=config['training'].get('blur_prob', 0.3),
        cutout_prob=config['training'].get('cutout_prob', 0.2),
        curriculum_epoch=0 if args.curriculum else None,
        max_curriculum_epochs=max_curriculum_epochs
    )
    
    valid_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'], 
        split='valid', 
        sequence_length=config['model']['sequence_length'], 
        augment=False, 
        input_size=input_size_tuple,
        use_optical_flow=config['model'].get('use_optical_flow', True),
        use_heatmaps=True
    )
    
    # Create data loaders with custom samplers if needed
    if args.hard_mining:
        # Initialize with a uniform sampler, will be updated during training
        train_sampler = torch.utils.data.RandomSampler(train_set)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config['training']['batch_size'], 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=config['training']['num_workers'], 
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_set, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers'], 
        pin_memory=(device.type == 'cuda')
    )

    # Model
    model = build_model_from_config(config).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer with learning rate warmup and cosine annealing
    num_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/shuttletrack')

    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)

    # --- Resume logic ---
    start_epoch = 1
    checkpoint_path = 'checkpoints/checkpoint_last.pth'
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Restore hard mining state if available and enabled
        if args.hard_mining and 'hard_sample_indices' in checkpoint:
            train_set.hard_sample_indices = checkpoint.get('hard_sample_indices', [])
            train_set.hard_sample_losses = checkpoint.get('hard_sample_losses', [])
            train_set.use_hard_mining = len(train_set.hard_sample_indices) > 10
            
        print(f"[INFO] Resumed training from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
    else:
        best_val_loss = float('inf')
        print("[INFO] No checkpoint found or resume not requested. Starting training from scratch.")

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
    num_epochs = config['training']['epochs']

    # Record start time
    training_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}/{config["training"]["epochs"]}')
        
        # Update curriculum difficulty level if enabled
        if args.curriculum:
            train_set.set_curriculum_epoch(epoch)
            print(f"Curriculum learning: Using difficulty level '{train_set.difficulty_level or 'all'}'")
        
        # Training
        batch_losses = []  # For hard example mining
        batch_indices = []  # For hard example mining
        
        train_loss, train_bce, train_mse, train_smooth = train_one_epoch(
            model, train_loader, optimizer, device, config, 
            epoch, num_epochs, 
            batch_losses=batch_losses, 
            batch_indices=batch_indices
        )
        print("Finished training epoch, starting validation...")
        
        # Update hard example mining if enabled
        if args.hard_mining and batch_losses and batch_indices:
            train_set.update_hard_samples(batch_indices, batch_losses)
            print(f"Hard example mining: Tracking {len(train_set.hard_sample_indices)} hard samples")
        
        # Validation
        val_loss, val_bce, val_mse, val_smooth = validate(
            model, valid_loader, device, config, 
            epoch, num_epochs
        )
        print("Finished validation, starting evaluation on valid set...")
        
        val_metrics = evaluate(model, valid_loader, device)
        print("Finished all evaluations, proceeding to logging and checkpointing...")
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
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
        print(f'Learning Rate: {current_lr:.6f}')
        
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
        
        # Add heatmap metrics if available
        if 'heatmap_pck' in val_metrics:
            table_data.append(["Heatmap PCK", "-", f"{val_metrics['heatmap_pck']*100:.1f}%", f"{best_metrics.get('val_heatmap_pck', 0)*100:.1f}%"])
        
        # Print metrics table
        print(f"\nEpoch {epoch} Summary | Time: {format_time(epoch_time)} | Total: {format_time(total_time)}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print human-readable insights
        print("\n📊 Training Insights:")
        for insight in insights:
            print(f"  {insight}")
        print("")
        
        # Early stopping check
        early_stopping_patience = config['training'].get('early_stopping_patience', 15)
        if epochs_without_improvement >= early_stopping_patience:
            print(f"⛔ Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break
        
        # Save best model
        if is_best:
            # Save complete checkpoint with additional data
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'config': config,
                'metrics': val_metrics,
                'best_metrics': best_metrics
            }
            
            # Add hard sample data if enabled
            if args.hard_mining:
                checkpoint_data['hard_sample_indices'] = train_set.hard_sample_indices
                checkpoint_data['hard_sample_losses'] = train_set.hard_sample_losses
                
            torch.save(checkpoint_data, f'checkpoints/checkpoint_best.pth')
            print('  [*] Saved new best model!')

            # Also save a more lightweight model for deployment
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'metrics': val_metrics
            }, f'checkpoints/model_best.pth')

        # Save last model with complete training state
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss': val_loss,
            'config': config
        }
        
        # Add hard sample data if enabled
        if args.hard_mining:
            checkpoint_data['hard_sample_indices'] = train_set.hard_sample_indices
            checkpoint_data['hard_sample_losses'] = train_set.hard_sample_losses
            
        torch.save(checkpoint_data, f'checkpoints/checkpoint_last.pth')
        print(f'  [*] Saved checkpoint for epoch {epoch}.')
        
        # Save additional checkpoints periodically
        if epoch % 10 == 0:
            torch.save(checkpoint_data, f'checkpoints/checkpoint_epoch_{epoch}.pth')
            print(f'  [*] Saved additional checkpoint at epoch {epoch}.')

    print(f"Training completed after {format_time(time.time() - training_start_time)}.")
    print(f"Best metrics achieved: Within 5px: {best_metrics['val_within_5px']*100:.1f}%, F1: {best_metrics['val_visibility_f1']:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
