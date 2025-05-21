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
from utils.eval import evaluate # Assuming this does not need debugging for now
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
    if previous_metrics is None:
        insights.append("🚀 Training started! Initial metrics captured.")
        if 'train_loss' in current_metrics and current_metrics['train_loss'] > 1000:
             insights.append(f"⚠️ Initial train loss ({current_metrics['train_loss']:.2f}) is very high. Check loss calculation or learning rate.")
        if 'val_loss' in current_metrics and current_metrics['val_loss'] > 1000:
             insights.append(f"⚠️ Initial validation loss ({current_metrics['val_loss']:.2f}) is very high. Check loss calculation or data.")
        return insights
    improved_val_metrics = 0
    total_key_metrics = 0
    key_metric_weights = {
        'val_loss': 1.0, 'val_visibility_f1': 2.0,
        'val_within_5px': 2.0, 'val_distance_error': 1.5
    }
    for metric, weight in key_metric_weights.items():
        total_key_metrics += weight
        if metric not in current_metrics or metric not in previous_metrics: continue
        if 'f1' in metric or 'precision' in metric or 'recall' in metric or 'within' in metric:
            if current_metrics[metric] > previous_metrics[metric]: improved_val_metrics += weight
        else:
            if current_metrics[metric] < previous_metrics[metric]: improved_val_metrics += weight
    improvement_ratio = improved_val_metrics / total_key_metrics if total_key_metrics > 0 else 0
    if improvement_ratio >= 0.7: insights.append(f"✅ Good progress! Model improving on {int(improvement_ratio * 100)}% of key metrics.")
    elif improvement_ratio >= 0.5: insights.append(f"🔄 Moderate improvement on {int(improvement_ratio * 100)}% of key metrics.")
    elif improvement_ratio > 0: insights.append(f"⚠️ Limited improvement ({int(improvement_ratio * 100)}% of key metrics).")
    else: insights.append("❌ No improvement on validation metrics in this epoch.")

    if 'train_loss' in current_metrics and 'val_loss' in current_metrics and \
       'train_loss' in previous_metrics and 'val_loss' in previous_metrics:
        train_val_gap = current_metrics['train_loss'] - current_metrics['val_loss']
        prev_train_val_gap = previous_metrics['train_loss'] - previous_metrics['val_loss']
        if train_val_gap < 0 and abs(train_val_gap) > abs(prev_train_val_gap) and abs(train_val_gap) > 0.1 * current_metrics['train_loss']: # Check if val_loss significantly better
            insights.append("⚠️ Model might be struggling if validation loss consistently much lower than training loss and gap is widening.")
        elif current_metrics['train_loss'] < previous_metrics['train_loss'] and current_metrics['val_loss'] > previous_metrics['val_loss'] and \
             (current_metrics['val_loss'] - current_metrics['train_loss']) > 0.2 * current_metrics['train_loss']: # Check for significant overfitting gap
            insights.append("⚠️ Warning: Possible overfitting! Training loss ↓ but validation loss ↑ significantly.")

    if 'val_visibility_f1' in current_metrics and 'val_visibility_f1' in previous_metrics and \
        current_metrics['val_visibility_f1'] > previous_metrics['val_visibility_f1']:
        insights.append(f"👁️ Detection accuracy improved: {current_metrics['val_visibility_f1']:.1%} (↑{(current_metrics['val_visibility_f1']-previous_metrics['val_visibility_f1'])*100:.1f}%)")
    if 'val_within_5px' in current_metrics and 'val_within_5px' in previous_metrics and \
        current_metrics['val_within_5px'] > previous_metrics['val_within_5px']:
        insights.append(f"🎯 Position accuracy improved: {current_metrics['val_within_5px']*100:.1f}% within 5px (↑{(current_metrics['val_within_5px']-previous_metrics['val_within_5px'])*100:.1f}%)")
    if num_epochs_without_improvement > 0:
        if num_epochs_without_improvement >= 10: insights.append(f"⛔ No improvement for {num_epochs_without_improvement} epochs! Consider early stopping.")
        elif num_epochs_without_improvement >= 5: insights.append(f"⚠️ No improvement for {num_epochs_without_improvement} epochs. Consider adjusting LR.")
        else: insights.append(f"ℹ️ No improvement for {num_epochs_without_improvement} epochs.")
    if best_metrics:
        if 'val_visibility_f1' in best_metrics and 'val_visibility_f1' in current_metrics:
            best_f1_gap = best_metrics['val_visibility_f1'] - current_metrics['val_visibility_f1']
            if best_f1_gap > 0.001: insights.append(f"🔍 Still {best_f1_gap*100:.1f}% behind best detection accuracy.")
        if 'val_within_5px' in best_metrics:
            best_within_5px = best_metrics['val_within_5px']
            insights.append(f"📈 Current best position accuracy (within 5px): {best_within_5px*100:.1f}% (Target: >95%)")
    progress = epoch / num_epochs if num_epochs > 0 else 0
    if progress > 0.8: insights.append(f"🏁 Training {progress*100:.0f}% complete ({epoch}/{num_epochs} epochs)")
    return insights

# --- Loss functions ---
def compute_losses(pred_dict, target, config, epoch=None, max_epochs=None):
    vis_true = target[..., 0]
    xy_true = target[..., 1:]
    batch_size, seq_len = vis_true.shape
    device = target.device # Added device for tensor creation

    print(f"\n[DEBUG LOSS] --- Epoch: {epoch}, Batch Size: {batch_size}, Seq Len: {seq_len} ---")
    if batch_size > 0 and seq_len > 0:
        print(f"[DEBUG LOSS] vis_true shape: {vis_true.shape}, example (first item, first 5 ts): {vis_true[0, :min(5, seq_len)]}")
        print(f"[DEBUG LOSS] vis_true unique values: {torch.unique(vis_true)}")
        print(f"[DEBUG LOSS] xy_true shape: {xy_true.shape}, example (first item, first frame): {xy_true[0, 0]}")
    else:
        print("[DEBUG LOSS] Empty batch or sequence encountered!")
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)


    has_uncertainty = 'uncertainty' in pred_dict
    print(f"[DEBUG LOSS] Has uncertainty: {has_uncertainty}")

    # ======= 1. Visibility Loss (Focal BCE) =======
    vis_pred = pred_dict['visibility'] # Logits
    if batch_size > 0 and seq_len > 0:
        print(f"[DEBUG LOSS] vis_pred shape: {vis_pred.shape}, example (logits, first item, first 5 ts): {vis_pred[0, :min(5, seq_len)]}")
        print(f"[DEBUG LOSS] vis_pred stats: min={vis_pred.min().item():.4f}, max={vis_pred.max().item():.4f}, mean={vis_pred.mean().item():.4f}")

    # Check for NaNs in predictions early
    if torch.isnan(vis_pred).any(): print("[DEBUG LOSS ERROR] NaN in vis_pred!")
    if torch.isinf(vis_pred).any(): print("[DEBUG LOSS ERROR] Inf in vis_pred!")


    bce_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        vis_pred, vis_true, reduction='none'
    )
    if batch_size > 0: print(f"[DEBUG LOSS] bce_loss_raw (before focal) shape: {bce_loss_raw.shape}, mean: {bce_loss_raw.mean().item():.4f}")

    gamma = config.get('training', {}).get('focal_gamma', 2.0)
    pt_vis = torch.sigmoid(vis_pred) * vis_true + (1 - torch.sigmoid(vis_pred)) * (1 - vis_true)
    focal_weight_vis = (1 - pt_vis).pow(gamma) # (1-pt)^gamma
    
    focal_bce = (bce_loss_raw * focal_weight_vis).mean()
    if batch_size > 0: print(f"[DEBUG LOSS] focal_weight_vis mean: {focal_weight_vis.mean().item():.4f}")
    print(f"[DEBUG LOSS] >>> focal_bce (visibility loss) <<< : {focal_bce.item():.4f}")

    # ======= 2. Create target heatmaps for position prediction =======
    heatmap_size = pred_dict['heatmap'].shape[-1]
    target_heatmaps = torch.zeros_like(pred_dict['heatmap'])
    for b_idx in range(batch_size):
        for t_idx in range(seq_len):
            if vis_true[b_idx, t_idx] > 0:
                x_norm, y_norm = xy_true[b_idx, t_idx]
                cx = int(x_norm * heatmap_size)
                cy = int(y_norm * heatmap_size)
                cx = max(0, min(cx, heatmap_size - 1))
                cy = max(0, min(cy, heatmap_size - 1))
                if 0 <= cy < heatmap_size and 0 <= cx < heatmap_size: # Boundary check
                    target_heatmaps[b_idx, t_idx, cy, cx] = 1.0
                    sigma_g = 1.0 # Gaussian sigma
                    radius_g = int(3 * sigma_g)
                    for y_h_idx in range(max(0, cy - radius_g), min(heatmap_size, cy + radius_g + 1)):
                        for x_h_idx in range(max(0, cx - radius_g), min(heatmap_size, cx + radius_g + 1)):
                            if (x_h_idx, y_h_idx) != (cx, cy):
                                dist_sq = (x_h_idx - cx)**2 + (y_h_idx - cy)**2
                                weight_g = torch.exp(torch.tensor(-dist_sq / (2 * sigma_g**2), device=device))
                                target_heatmaps[b_idx, t_idx, y_h_idx, x_h_idx] = max(target_heatmaps[b_idx, t_idx, y_h_idx, x_h_idx].item(), weight_g.item())
    
    if batch_size > 0 and seq_len > 0: # Avoid div by zero if empty
        max_vals_heatmap = target_heatmaps.reshape(batch_size, seq_len, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        target_heatmaps = target_heatmaps / (max_vals_heatmap + 1e-8)
        print(f"[DEBUG LOSS] target_heatmaps (first item, first frame) stats: min={target_heatmaps[0,0].min().item():.4f}, max={target_heatmaps[0,0].max().item():.4f}, sum={target_heatmaps[0,0].sum().item():.4f}")

    # ======= 3. Heatmap Loss (MSE) =======
    heatmap_pred = pred_dict['heatmap']
    if torch.isnan(heatmap_pred).any(): print("[DEBUG LOSS ERROR] NaN in heatmap_pred!")
    if torch.isinf(heatmap_pred).any(): print("[DEBUG LOSS ERROR] Inf in heatmap_pred!")

    if batch_size > 0 and seq_len > 0:
        print(f"[DEBUG LOSS] heatmap_pred (first item, first frame) stats: min={heatmap_pred[0,0].min().item():.4f}, max={heatmap_pred[0,0].max().item():.4f}, sum={heatmap_pred[0,0].sum().item():.4f}")

    vis_mask_hm = vis_true.unsqueeze(-1).unsqueeze(-1)
    masked_heatmap_pred = heatmap_pred * vis_mask_hm
    masked_target_heatmaps = target_heatmaps * vis_mask_hm
    
    num_visible_pixels_for_mse = torch.sum(vis_mask_hm) * heatmap_size * heatmap_size # Total pixels in visible heatmaps
    # Alternative: sum over elements of vis_mask_hm that are active
    num_active_heatmaps_for_mse = torch.sum(vis_true)
    
    if num_active_heatmaps_for_mse > 0:
        # MSE should be per heatmap element, for active heatmaps
        mse_heatmap = nn.MSELoss(reduction='sum')(masked_heatmap_pred, masked_target_heatmaps) / (num_active_heatmaps_for_mse * heatmap_size * heatmap_size + 1e-8)
    else:
        mse_heatmap = torch.tensor(0.0, device=device)
    print(f"[DEBUG LOSS] >>> mse_heatmap <<< : {mse_heatmap.item():.4f}, num_active_heatmaps_for_mse: {num_active_heatmaps_for_mse.item()}")

    # ======= 4. Focal Loss for heatmap (MSE weighted by focal term) =======
    alpha_focal_hm = 2.0
    squared_error_hm = (masked_heatmap_pred - masked_target_heatmaps).pow(2)
    # Weight where target is high (peaks) and prediction is low, or target is low and prediction is high
    focal_weight_hm = (masked_target_heatmaps * (1 - masked_heatmap_pred).abs().pow(alpha_focal_hm) + \
                       (1 - masked_target_heatmaps) * masked_heatmap_pred.abs().pow(alpha_focal_hm)) * vis_mask_hm

    if num_active_heatmaps_for_mse > 0:
        focal_loss_heatmap = (squared_error_hm * focal_weight_hm).sum() / (num_active_heatmaps_for_mse * heatmap_size * heatmap_size + 1e-8)
    else:
        focal_loss_heatmap = torch.tensor(0.0, device=device)
    print(f"[DEBUG LOSS] >>> focal_loss_heatmap <<< : {focal_loss_heatmap.item():.4f}")

    # ======= 5. Extract predicted coordinates for trajectory loss =======
    pred_coords_hm, pred_conf_hm = extract_coordinates_from_heatmap(heatmap_pred.detach()) # Detach heatmap for coord extraction if loss on coords shouldn't affect heatmap directly
    if batch_size > 0 and seq_len > 0:
        print(f"[DEBUG LOSS] pred_coords_hm shape: {pred_coords_hm.shape}, example (first item, first frame): {pred_coords_hm[0,0]}")
        print(f"[DEBUG LOSS] pred_conf_hm shape: {pred_conf_hm.shape}, example (first item, first 5 ts): {pred_conf_hm[0,:min(5, seq_len)]}")

    # ======= 6. Smooth L1 Loss for direct coordinate regression =======
    num_visible_coords = torch.sum(vis_true)
    if num_visible_coords > 0:
        smooth_l1_coord_loss = torch.nn.functional.smooth_l1_loss(
            pred_coords_hm * vis_true.unsqueeze(-1),
            xy_true * vis_true.unsqueeze(-1),
            beta=0.1, reduction='sum'
        ) / (num_visible_coords * 2 + 1e-8) # *2 for x,y
    else:
        smooth_l1_coord_loss = torch.tensor(0.0, device=device)
    print(f"[DEBUG LOSS] >>> smooth_l1_coord_loss <<< : {smooth_l1_coord_loss.item():.4f}, num_visible_coords: {num_visible_coords.item()}")

    # ======= 7. Trajectory continuity loss (L2 on differences) =======
    diff_pred_coords = pred_coords_hm[:, 1:, :] - pred_coords_hm[:, :-1, :]
    diff_true_coords = xy_true[:, 1:, :] - xy_true[:, :-1, :]
    vis_pairs_traj = vis_true[:, :-1] * vis_true[:, 1:]
    
    num_visible_pairs_traj = torch.sum(vis_pairs_traj)
    if num_visible_pairs_traj > 0:
        smooth_loss_trajectory = torch.sum((diff_pred_coords - diff_true_coords).pow(2) * vis_pairs_traj.unsqueeze(-1))
        smooth_loss_trajectory = smooth_loss_trajectory / (num_visible_pairs_traj * 2 + 1e-8) # *2 for x,y
    else:
        smooth_loss_trajectory = torch.tensor(0.0, device=device)
    print(f"[DEBUG LOSS] >>> smooth_loss_trajectory <<< : {smooth_loss_trajectory.item():.4f}, num_visible_pairs_traj: {num_visible_pairs_traj.item()}")

    # ======= 8. Uncertainty Loss (if enabled) =======
    uncertainty_loss = torch.tensor(0.0, device=device)
    if has_uncertainty:
        uncertainty_pred_model = pred_dict['uncertainty'] # (B, T, 2) for sigma_x, sigma_y
        if torch.isnan(uncertainty_pred_model).any(): print("[DEBUG LOSS ERROR] NaN in uncertainty_pred_model!")
        if torch.isinf(uncertainty_pred_model).any(): print("[DEBUG LOSS ERROR] Inf in uncertainty_pred_model!")
        if batch_size > 0 and seq_len > 0:
            print(f"[DEBUG LOSS] uncertainty_pred_model shape: {uncertainty_pred_model.shape}, example (first item, first frame): {uncertainty_pred_model[0,0]}")
            print(f"[DEBUG LOSS] uncertainty_pred_model stats: min={uncertainty_pred_model.min().item():.4f}, max={uncertainty_pred_model.max().item():.4f}, mean={uncertainty_pred_model.mean().item():.4f}")

        # sigma_sq = uncertainty_pred_model.pow(2) # Assuming model outputs std dev (sigma)
        # Using Softplus in model, output is already sigma. If model outputs log_sigma_sq, then sigma_sq = exp(output)
        # Let's assume model output is sigma directly, so use uncertainty_pred_model.pow(2) for sigma_sq
        log_sigma_sq = 2 * torch.log(uncertainty_pred_model + 1e-8) # log(sigma^2) = 2*log(sigma)
        
        diff_x_sq = (pred_coords_hm.detach()[..., 0] - xy_true[..., 0]).pow(2)
        diff_y_sq = (pred_coords_hm.detach()[..., 1] - xy_true[..., 1]).pow(2)

        # Loss = 0.5 * exp(-log_sigma_sq) * diff_sq + 0.5 * log_sigma_sq (Aleatoric Uncertainty for regression)
        nll_x = 0.5 * torch.exp(-log_sigma_sq[..., 0]) * diff_x_sq + 0.5 * log_sigma_sq[..., 0]
        nll_y = 0.5 * torch.exp(-log_sigma_sq[..., 1]) * diff_y_sq + 0.5 * log_sigma_sq[..., 1]
        
        nll_combined = (nll_x + nll_y) * vis_true # Mask by visibility (B,T)
        
        if num_visible_coords > 0:
            uncertainty_loss = torch.sum(nll_combined) / num_visible_coords
        else:
            uncertainty_loss = torch.tensor(0.0, device=device)
        print(f"[DEBUG LOSS] >>> uncertainty_loss <<< : {uncertainty_loss.item():.4f}")


    # ======= 9. Dynamic Loss Weighting =======
    if epoch is not None and max_epochs is not None:
        progress = epoch / max_epochs if max_epochs > 0 else 0
        smooth_weight_factor = min(1.0, 0.5 + progress)
        focal_weight_factor = max(0.3, 1.0 - progress * 0.7)
        uncertainty_weight_factor = min(1.0, progress * 2.0) # Make it reach 1.0 by mid-training
    else:
        smooth_weight_factor = 1.0; focal_weight_factor = 1.0; uncertainty_weight_factor = 1.0
    print(f"[DEBUG LOSS] Dynamic Weights: smooth_wf={smooth_weight_factor:.2f}, focal_wf={focal_weight_factor:.2f}, uncert_wf={uncertainty_weight_factor:.2f}")

    # ======= 10. Get loss weights from config =======
    loss_weights_map = config.get('training', {}).get('loss_weights', {
        'focal_bce': 1.0, 'mse': 1.0, 'focal': 0.5,
        'smooth': 0.1, 'smooth_l1': 0.5, 'uncertainty': 0.2
    })

    w_focal_bce = loss_weights_map.get('focal_bce', 1.0) * focal_weight_factor
    w_mse_hm = loss_weights_map.get('mse', 1.0)
    w_focal_hm = loss_weights_map.get('focal', 0.5) * focal_weight_factor
    w_smooth_traj = loss_weights_map.get('smooth', 0.1) * smooth_weight_factor
    w_smooth_l1 = loss_weights_map.get('smooth_l1', 0.5)
    w_uncertainty = loss_weights_map.get('uncertainty', 0.2) * uncertainty_weight_factor if has_uncertainty else 0.0
    print(f"[DEBUG LOSS] Final Loss Weights: focal_bce_w={w_focal_bce:.3f}, mse_hm_w={w_mse_hm:.3f}, focal_hm_w={w_focal_hm:.3f}, smooth_traj_w={w_smooth_traj:.3f}, smooth_l1_w={w_smooth_l1:.3f}, uncert_w={w_uncertainty:.3f}")

    # ======= 11. Calculate total loss (Weighted Components) =======
    L_focal_bce_final = w_focal_bce * focal_bce
    L_mse_hm_final = w_mse_hm * mse_heatmap
    L_focal_hm_final = w_focal_hm * focal_loss_heatmap
    L_smooth_traj_final = w_smooth_traj * smooth_loss_trajectory
    L_smooth_l1_final = w_smooth_l1 * smooth_l1_coord_loss
    L_uncertainty_final = w_uncertainty * uncertainty_loss

    print(f"[DEBUG LOSS] Weighted Components: L_focal_bce={L_focal_bce_final.item():.4f}, L_mse_hm={L_mse_hm_final.item():.4f}, L_focal_hm={L_focal_hm_final.item():.4f}, L_smooth_traj={L_smooth_traj_final.item():.4f}, L_smooth_l1={L_smooth_l1_final.item():.4f}, L_uncertainty={L_uncertainty_final.item():.4f}")

    total_loss = (
        L_focal_bce_final + L_mse_hm_final + L_focal_hm_final +
        L_smooth_traj_final + L_smooth_l1_final + L_uncertainty_final
    )
    print(f"[DEBUG LOSS] >>> FINAL TOTAL LOSS <<< : {total_loss.item():.4f}\n")
    
    # For logging, return the main unweighted losses you want to track.
    # The reported BCE loss in your table was 0.0, MSE was 0.000x.
    # Let's return focal_bce, mse_heatmap, and smooth_loss_trajectory for consistency with your previous naming.
    return total_loss, focal_bce, mse_heatmap, smooth_loss_trajectory


# --- Training loop ---
# ... (train_one_epoch, validate, format_time remain largely the same, ensure they use the new return values from compute_losses)
def train_one_epoch(model, loader, optimizer, device, config, epoch=None, max_epochs=None, batch_losses=None, batch_indices=None):
    model.train()
    running_total_loss = 0.0 # Use running sum for precision
    running_bce_loss = 0.0
    running_mse_loss = 0.0
    running_smooth_loss = 0.0
    num_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)):
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        flows = batch.get('flows')
        if flows is not None:
            flows = flows.to(device)
        labels = batch['labels'].to(device)
        
        current_batch_size = frames.size(0)
        if current_batch_size == 0: continue # Skip empty batches if they somehow occur

        optimizer.zero_grad()
        
        pred = model(frames, diffs, flows)
        
        # Check for NaNs in model predictions before loss calculation
        for key, value in pred.items():
            if torch.isnan(value).any():
                print(f"[TRAIN LOOP ERROR] NaN detected in model prediction: {key} in batch {batch_idx}")
            if torch.isinf(value).any():
                print(f"[TRAIN LOOP ERROR] Inf detected in model prediction: {key} in batch {batch_idx}")


        loss, bce, mse, smooth = compute_losses(pred, labels, config, epoch, max_epochs)
        
        if torch.isnan(loss):
            print(f"[TRAIN LOOP ERROR] NaN detected in total_loss for batch {batch_idx}. Skipping backward pass for this batch.")
            # Optionally, log more details about pred and labels here if loss is NaN
            continue # Skip backprop if loss is NaN

        loss.backward()
        
        if 'gradient_clip_val' in config.get('training', {}):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
            
        optimizer.step()
        
        running_total_loss += loss.item() * current_batch_size
        running_bce_loss += bce.item() * current_batch_size
        running_mse_loss += mse.item() * current_batch_size
        running_smooth_loss += smooth.item() * current_batch_size
        num_samples += current_batch_size
        
        if batch_losses is not None and batch_indices is not None: # For hard mining
            batch_losses.append(loss.item())
            # Ensure batch['index'] is correctly handled
            idx_val = batch['index']
            if torch.is_tensor(idx_val) and idx_val.numel() == 1:
                batch_indices.append(idx_val.item())
            elif isinstance(idx_val, (list, tuple)) and len(idx_val) > 0 and isinstance(idx_val[0], torch.Tensor):
                 batch_indices.append(idx_val[0].item()) # Assuming first element if list of tensors
            elif isinstance(idx_val, (list, tuple)) and len(idx_val) > 0:
                 batch_indices.append(idx_val[0]) # Assuming first element if list of numbers
            else: # Fallback or log an error
                 print(f"[TRAIN LOOP WARNING] Unexpected type for batch['index']: {type(idx_val)}. Skipping hard mining for this item.")


    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0 # Avoid division by zero
    return running_total_loss / num_samples, running_bce_loss / num_samples, \
           running_mse_loss / num_samples, running_smooth_loss / num_samples


def validate(model, loader, device, config, epoch=None, max_epochs=None):
    model.eval()
    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_mse_loss = 0.0
    running_smooth_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid', leave=False):
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            flows = batch.get('flows')
            if flows is not None:
                flows = flows.to(device)
            labels = batch['labels'].to(device)
            current_batch_size = frames.size(0)
            if current_batch_size == 0: continue

            pred = model(frames, diffs, flows)
            loss, bce, mse, smooth = compute_losses(pred, labels, config, epoch, max_epochs) # No epoch/max_epochs for val if dynamic weighting is train-only

            if torch.isnan(loss):
                print(f"[VALIDATE ERROR] NaN detected in validation loss. Skipping this batch's contribution.")
                continue

            running_total_loss += loss.item() * current_batch_size
            running_bce_loss += bce.item() * current_batch_size
            running_mse_loss += mse.item() * current_batch_size
            running_smooth_loss += smooth.item() * current_batch_size
            num_samples += current_batch_size
    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0
    return running_total_loss / num_samples, running_bce_loss / num_samples, \
           running_mse_loss / num_samples, running_smooth_loss / num_samples

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

    input_size_cfg = config['model']['input_size']
    input_size_tuple = tuple(input_size_cfg) if isinstance(input_size_cfg, list) else (input_size_cfg, input_size_cfg)

    initial_difficulty = None 
    max_curriculum_epochs = config['training'].get('curriculum_epochs', 20) if args.curriculum else 0
    
    train_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'], split='Train', 
        sequence_length=config['model']['sequence_length'], augment=True, input_size=input_size_tuple,
        use_optical_flow=config['model'].get('use_optical_flow', True), use_heatmaps=True,
        difficulty_level=initial_difficulty,
        mixup_prob=config['training'].get('mixup_prob', 0.3), blur_prob=config['training'].get('blur_prob', 0.3),
        cutout_prob=config['training'].get('cutout_prob', 0.2),
        curriculum_epoch=0 if args.curriculum else None, max_curriculum_epochs=max_curriculum_epochs
    )
    valid_set = ShuttleTrackDataset(
        config['data']['processed_dataset_path'], split='valid', 
        sequence_length=config['model']['sequence_length'], augment=False, input_size=input_size_tuple,
        use_optical_flow=config['model'].get('use_optical_flow', True), use_heatmaps=True
    )
    
    train_sampler = torch.utils.data.RandomSampler(train_set) if args.hard_mining else None
    train_loader = DataLoader(
        train_set, batch_size=config['training']['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=config['training']['num_workers'], pin_memory=(device.type == 'cuda'), drop_last=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=config['training']['batch_size'], shuffle=False, 
        num_workers=config['training']['num_workers'], pin_memory=(device.type == 'cuda')
    )

    model = build_model_from_config(config).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    num_epochs_cfg = config['training']['epochs']
    warmup_epochs_cfg = config['training'].get('warmup_epochs', 5)
    optimizer = optim.AdamW(
        model.parameters(), lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'], betas=(0.9, 0.999), eps=1e-8
    )
    
    def lr_lambda_func(epoch_idx): # Renamed to avoid conflict
        if epoch_idx < warmup_epochs_cfg:
            return float(epoch_idx + 1) / float(warmup_epochs_cfg) # Ensure float division and 1-based epoch for warmup
        progress_val = float(epoch_idx - warmup_epochs_cfg) / float(max(1, num_epochs_cfg - warmup_epochs_cfg))
        return 0.5 * (1.0 + np.cos(np.pi * progress_val))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_func)

    writer = SummaryWriter(log_dir='runs/shuttletrack')
    os.makedirs('checkpoints', exist_ok=True)

    start_epoch = 1
    checkpoint_path = 'checkpoints/checkpoint_last.pth'
    best_val_loss_overall = float('inf') # Renamed to avoid conflict with per-epoch val_loss

    if args.resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss_overall = checkpoint.get('loss', float('inf')) # Use .get for safety
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.hard_mining and 'hard_sample_indices' in checkpoint:
            train_set.hard_sample_indices = checkpoint.get('hard_sample_indices', [])
            train_set.hard_sample_losses = checkpoint.get('hard_sample_losses', [])
            train_set.use_hard_mining = len(train_set.hard_sample_indices) > 10
        print(f"[INFO] Resumed from epoch {start_epoch}. Best val loss so far: {best_val_loss_overall:.4f}")
    else:
        print("[INFO] No checkpoint or not resuming. Starting fresh.")

    best_metrics_overall = { # Renamed
        'val_loss': float('inf'), 'val_bce': float('inf'), 'val_mse': float('inf'), 
        'val_distance_error': float('inf'), 'val_visibility_f1': 0, 'val_precision': 0,
        'val_recall': 0, 'val_within_5px': 0, 'val_within_10px': 0, 'epoch': 0
    }

    previous_metrics_epoch = None # Renamed
    epochs_without_improvement_count = 0 # Renamed
    
    training_start_time = time.time()

    for epoch_num in range(start_epoch, num_epochs_cfg + 1): # Renamed
        epoch_start_time_loop = time.time() # Renamed
        print(f'\nEpoch {epoch_num}/{num_epochs_cfg}')
        
        if args.curriculum:
            train_set.set_curriculum_epoch(epoch_num)
            print(f"Curriculum: Difficulty '{train_set.difficulty_level or 'all'}'")
        
        batch_losses_list = [] # Renamed
        batch_indices_list = [] # Renamed
        
        train_loss_epoch, train_bce_epoch, train_mse_epoch, train_smooth_epoch = train_one_epoch(
            model, train_loader, optimizer, device, config, 
            epoch_num, num_epochs_cfg, 
            batch_losses=batch_losses_list, batch_indices=batch_indices_list
        )
        print("Train epoch done. Validating...")
        
        if args.hard_mining and batch_losses_list and batch_indices_list:
            train_set.update_hard_samples(batch_indices_list, batch_losses_list)
            print(f"Hard mining: Tracking {len(train_set.hard_sample_indices)} samples")
        
        val_loss_epoch, val_bce_epoch, val_mse_epoch, val_smooth_epoch = validate(
            model, valid_loader, device, config, epoch_num, num_epochs_cfg
        )
        print("Validation done. Evaluating on valid set...")
        
        val_metrics_epoch = evaluate(model, valid_loader, device) # Renamed
        print("Evaluation done. Logging & checkpointing...")
        
        scheduler.step() # Step scheduler after optimizer.step() per epoch
        current_lr_val = scheduler.get_last_lr()[0] # Renamed
        writer.add_scalar('Learning_Rate', current_lr_val, epoch_num)
        
        is_best_epoch = False # Renamed
        if val_loss_epoch < best_val_loss_overall:
            best_val_loss_overall = val_loss_epoch
            epochs_without_improvement_count = 0
            is_best_epoch = True # This flag should be based on primary metric like val_loss
        else:
            epochs_without_improvement_count += 1
            
        # Update best_metrics_overall more robustly
        if val_loss_epoch < best_metrics_overall['val_loss']:
            best_metrics_overall['val_loss'] = val_loss_epoch
            best_metrics_overall['epoch'] = epoch_num
            # is_best_epoch = True # Already handled by best_val_loss_overall
        if val_bce_epoch < best_metrics_overall.get('val_bce', float('inf')): best_metrics_overall['val_bce'] = val_bce_epoch
        if val_mse_epoch < best_metrics_overall.get('val_mse', float('inf')): best_metrics_overall['val_mse'] = val_mse_epoch

        for key_m, value_m in val_metrics_epoch.items(): # Renamed
            val_key_m = f'val_{key_m}'
            current_best_val = best_metrics_overall.get(val_key_m, -float('inf') if any(k in key_m for k in ['f1', 'within', 'precision', 'recall']) else float('inf'))
            
            if any(k_metric in key_m for k_metric in ['f1', 'within', 'precision', 'recall']): # Higher is better
                if value_m > current_best_val:
                    best_metrics_overall[val_key_m] = value_m
                    # is_best_epoch = True # Consider if this metric should also trigger "best model" save
            elif 'error' in key_m: # Lower is better
                if value_m < current_best_val:
                    best_metrics_overall[val_key_m] = value_m
                    # is_best_epoch = True
        
        current_metrics_for_insights = { # Renamed
            'train_loss': train_loss_epoch, 'val_loss': val_loss_epoch,
            'val_visibility_f1': val_metrics_epoch.get('visibility_f1', 0),
            'val_within_5px': val_metrics_epoch.get('within_5px', 0),
            'val_within_10px': val_metrics_epoch.get('within_10px', 0),
            'val_distance_error': val_metrics_epoch.get('distance_error', float('inf'))
        }
        
        insights_list = generate_human_insights( # Renamed
            current_metrics_for_insights, previous_metrics_epoch, 
            best_metrics_overall, epoch_num, num_epochs_cfg, epochs_without_improvement_count
        )
        previous_metrics_epoch = current_metrics_for_insights
        
        epoch_time_val = time.time() - epoch_start_time_loop # Renamed
        total_time_val = time.time() - training_start_time # Renamed
        
        writer.add_scalar('Loss/train_total', train_loss_epoch, epoch_num)
        writer.add_scalar('Loss/train_BCE', train_bce_epoch, epoch_num)
        writer.add_scalar('Loss/train_MSE', train_mse_epoch, epoch_num)
        writer.add_scalar('Loss/train_Smooth', train_smooth_epoch, epoch_num)
        writer.add_scalar('Loss/val_total', val_loss_epoch, epoch_num)
        writer.add_scalar('Loss/val_BCE', val_bce_epoch, epoch_num)
        writer.add_scalar('Loss/val_MSE', val_mse_epoch, epoch_num)
        writer.add_scalar('Loss/val_Smooth', val_smooth_epoch, epoch_num)
        for key, value in val_metrics_epoch.items(): writer.add_scalar(f'Metrics/val_{key}', value, epoch_num)
        
        print(f'Train Loss: {train_loss_epoch:.4f} (BCE: {train_bce_epoch:.4f}, MSE: {train_mse_epoch:.4f}, Smooth: {train_smooth_epoch:.4f})')
        print(f'Valid Loss: {val_loss_epoch:.4f} (BCE: {val_bce_epoch:.4f}, MSE: {val_mse_epoch:.4f}, Smooth: {val_smooth_epoch:.4f})')
        print(f'Learning Rate: {current_lr_val:.6f}')
        
        headers_table = ["Metric", "Train", "Valid", "Best Overall"] # Renamed
        table_data_list = [ # Renamed
            ["Total Loss", f"{train_loss_epoch:.4f}", f"{val_loss_epoch:.4f}", f"{best_metrics_overall['val_loss']:.4f}"],
            ["BCE Loss", f"{train_bce_epoch:.4f}", f"{val_bce_epoch:.4f}", f"{best_metrics_overall.get('val_bce', float('inf')):.4f}"],
            ["MSE Loss", f"{train_mse_epoch:.4f}", f"{val_mse_epoch:.4f}", f"{best_metrics_overall.get('val_mse', float('inf')):.4f}"],
            ["Dist Error", "-", f"{val_metrics_epoch.get('distance_error', float('inf')):.4f}", f"{best_metrics_overall.get('val_distance_error', float('inf')):.4f}"],
            ["Vis F1", "-", f"{val_metrics_epoch.get('visibility_f1', 0):.4f}", f"{best_metrics_overall.get('val_visibility_f1', 0):.4f}"],
            ["Precision", "-", f"{val_metrics_epoch.get('precision', 0):.4f}", f"{best_metrics_overall.get('val_precision', 0):.4f}"],
            ["Recall", "-", f"{val_metrics_epoch.get('recall', 0):.4f}", f"{best_metrics_overall.get('val_recall', 0):.4f}"],
            ["Within 5px", "-", f"{val_metrics_epoch.get('within_5px', 0)*100:.1f}%", f"{best_metrics_overall.get('val_within_5px', 0)*100:.1f}%"],
            ["Within 10px", "-", f"{val_metrics_epoch.get('within_10px', 0)*100:.1f}%", f"{best_metrics_overall.get('val_within_10px', 0)*100:.1f}%"]
        ]
        if 'heatmap_pck' in val_metrics_epoch:
            table_data_list.append(["Heatmap PCK", "-", f"{val_metrics_epoch['heatmap_pck']*100:.1f}%", f"{best_metrics_overall.get('val_heatmap_pck', 0)*100:.1f}%"])
        
        print(f"\nEpoch {epoch_num} Summary | Time: {format_time(epoch_time_val)} | Total: {format_time(total_time_val)}")
        metrics_table_str = tabulate(table_data_list, headers=headers_table, tablefmt="grid") # Renamed
        print(metrics_table_str)
        
        with open(f'checkpoints/epoch_metrics_log.txt', 'a') as f_log: # Renamed
            f_log.write(f"\nEpoch {epoch_num}/{num_epochs_cfg} Summary | Time: {format_time(epoch_time_val)} | Total: {format_time(total_time_val)}\n")
            f_log.write(metrics_table_str)
            f_log.write("\n\n📊 Training Insights:\n"); [f_log.write(f"  {insight}\n") for insight in insights_list]
            f_log.write("\n" + "="*80 + "\n")
        
        print("\n📊 Training Insights:"); [print(f"  {insight}") for insight in insights_list]; print("")
        
        early_stopping_patience_val = config['training'].get('early_stopping_patience', 15) # Renamed
        if epochs_without_improvement_count >= early_stopping_patience_val:
            print(f"⛔ Early stopping after {epochs_without_improvement_count} epochs without improvement on val_loss.")
            break
        
        # Primary condition for saving "best" model should be the overall validation loss improvement.
        if is_best_epoch: # If val_loss_epoch < best_val_loss_overall
            checkpoint_data_save = { # Renamed
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch_num, 'loss': val_loss_epoch,
                'config': config, 'metrics': val_metrics_epoch, 'best_metrics': best_metrics_overall
            }
            if args.hard_mining:
                checkpoint_data_save['hard_sample_indices'] = train_set.hard_sample_indices
                checkpoint_data_save['hard_sample_losses'] = train_set.hard_sample_losses
            torch.save(checkpoint_data_save, f'checkpoints/checkpoint_best.pth')
            print('  [*] Saved new best model (based on val_loss)!')
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'metrics': val_metrics_epoch}, f'checkpoints/model_best_lightweight.pth')

        last_checkpoint_data = { # Renamed
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch_num, 'loss': val_loss_epoch, 'config': config
        }
        if args.hard_mining:
            last_checkpoint_data['hard_sample_indices'] = train_set.hard_sample_indices
            last_checkpoint_data['hard_sample_losses'] = train_set.hard_sample_losses
        torch.save(last_checkpoint_data, f'checkpoints/checkpoint_last.pth')
        # print(f'  [*] Saved checkpoint for epoch {epoch_num}.') # Less verbose

        if epoch_num % 10 == 0:
            torch.save(last_checkpoint_data, f'checkpoints/checkpoint_epoch_{epoch_num}.pth')
            print(f'  [*] Saved additional checkpoint at epoch {epoch_num}.')

    print(f"Training completed. Total time: {format_time(time.time() - training_start_time)}.")
    print(f"Best metrics: Within 5px: {best_metrics_overall.get('val_within_5px',0)*100:.1f}%, F1: {best_metrics_overall.get('val_visibility_f1',0):.4f}, Loss: {best_metrics_overall['val_loss']:.4f}")
    writer.close()

if __name__ == '__main__':
    main()