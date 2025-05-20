import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from models.shuttletrack import extract_coordinates_from_heatmap


def compute_distance_error(pred_xy, true_xy, visibility=None):
    """
    Compute mean Euclidean distance error between predicted and true (x, y).
    Optionally mask by visibility.
    Args:
        pred_xy: (N, T, 2) predicted x, y (normalized)
        true_xy: (N, T, 2) ground truth x, y (normalized)
        visibility: (N, T) binary mask (optional)
    Returns:
        mean_distance: float
    """
    dist = np.linalg.norm(pred_xy - true_xy, axis=-1)  # (N, T)
    if visibility is not None:
        dist = dist * visibility
        total = np.sum(visibility)
        if total == 0:
            return 0.0
        return np.sum(dist) / total
    return np.mean(dist)


def compute_visibility_f1(pred_vis, true_vis, threshold=0.5):
    """
    Compute F1-score for shuttlecock visibility detection.
    Args:
        pred_vis: (N, T) predicted logits or probabilities
        true_vis: (N, T) ground truth (0 or 1)
    Returns:
        f1: float
    """
    pred_bin = (pred_vis > threshold).astype(np.uint8)
    return f1_score(true_vis.flatten(), pred_bin.flatten())


def compute_precision_recall(pred_vis, true_vis, threshold=0.5):
    """
    Compute precision and recall for shuttlecock visibility detection.
    Args:
        pred_vis: (N, T) predicted logits or probabilities
        true_vis: (N, T) ground truth (0 or 1)
    Returns:
        precision: float, recall: float
    """
    pred_bin = (pred_vis > threshold).astype(np.uint8)
    precision = precision_score(true_vis.flatten(), pred_bin.flatten(), zero_division=0)
    recall = recall_score(true_vis.flatten(), pred_bin.flatten(), zero_division=0)
    return precision, recall


def compute_within_distance(pred_xy, true_xy, thresholds=[5, 10], img_size=224, visibility=None):
    """
    Compute percentage of predictions within specified pixel thresholds.
    Args:
        pred_xy: (N, T, 2) predicted x, y (normalized 0-1)
        true_xy: (N, T, 2) ground truth x, y (normalized 0-1)
        thresholds: List of pixel thresholds to check
        img_size: Size of the image (used to convert from normalized to pixels)
        visibility: (N, T) binary mask (optional)
    Returns:
        dict: Percentage of predictions within each threshold
    """
    # Convert normalized coordinates to pixel distance
    dist = np.linalg.norm(pred_xy - true_xy, axis=-1) * img_size  # (N, T) in pixels
    
    results = {}
    for threshold in thresholds:
        within = (dist <= threshold).astype(np.float32)  # (N, T)
        
        if visibility is not None:
            # Only count visible points
            within = within * visibility
            total = np.sum(visibility)
            if total == 0:
                results[f'within_{threshold}px'] = 0.0
            else:
                results[f'within_{threshold}px'] = np.sum(within) / total
        else:
            results[f'within_{threshold}px'] = np.mean(within)
    
    return results


def compute_heatmap_metrics(pred_heatmaps, true_heatmaps, visibility=None):
    """
    Compute metrics specifically for heatmap predictions.
    Args:
        pred_heatmaps: (N, T, H, W) predicted heatmaps
        true_heatmaps: (N, T, H, W) ground truth heatmaps
        visibility: (N, T) binary mask (optional)
    Returns:
        dict: Heatmap-specific metrics
    """
    N, T, H, W = pred_heatmaps.shape
    
    # Reshape for easier computation
    pred_flat = pred_heatmaps.reshape(N*T, H*W)
    true_flat = true_heatmaps.reshape(N*T, H*W)
    
    if visibility is not None:
        # Only consider visible frames
        vis_mask = visibility.reshape(N*T)
        valid_indices = np.where(vis_mask > 0)[0]
        if len(valid_indices) == 0:
            return {'heatmap_mse': 0.0, 'heatmap_pck': 0.0}
            
        pred_flat = pred_flat[valid_indices]
        true_flat = true_flat[valid_indices]
    
    # Mean squared error
    mse = np.mean((pred_flat - true_flat) ** 2)
    
    # Peak accuracy - percentage of predictions where max peak is within 3 pixels of ground truth peak
    pred_peaks = np.argmax(pred_flat, axis=1)
    true_peaks = np.argmax(true_flat, axis=1)
    
    # Convert indices to 2D coordinates
    pred_y, pred_x = pred_peaks // W, pred_peaks % W
    true_y, true_x = true_peaks // W, true_peaks % W
    
    # Compute distance between peaks
    peak_distance = np.sqrt((pred_y - true_y) ** 2 + (pred_x - true_x) ** 2)
    
    # Percentage of Correct Keypoints (PCK) - within 3 pixels
    pck = np.mean(peak_distance <= 3)
    
    return {'heatmap_mse': mse, 'heatmap_pck': pck}


def evaluate(model, dataloader, device):
    print("[EVAL] Starting evaluation...")
    model.eval()
    all_pred_xy, all_true_xy = [], []
    all_pred_vis, all_true_vis = [], []
    all_pred_heatmaps, all_true_heatmaps = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"[EVAL] Processing batch {i+1}/{len(dataloader)}")
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            
            # Get optical flow if available
            flows = batch.get('flows')
            if flows is not None:
                flows = flows.to(device)
                
            # Get ground truth heatmaps if available
            gt_heatmaps = batch.get('heatmaps')
                
            labels = batch['labels'].to(device)
            
            # Forward pass through the model
            pred = model(frames, diffs, flows)  # Returns dict with 'visibility' and 'heatmap'
            
            # Extract visibility predictions
            pred_vis = torch.sigmoid(pred['visibility']).cpu().numpy()
            true_vis = labels[..., 0].cpu().numpy()
            
            # Extract coordinates from heatmaps
            pred_heatmaps = pred['heatmap'].cpu().numpy() 
            pred_coords, _ = extract_coordinates_from_heatmap(pred['heatmap'])
            pred_coords = pred_coords.cpu().numpy()
            
            # Ground truth coordinates
            true_xy = labels[..., 1:].cpu().numpy()
            
            # Store predictions and ground truth
            all_pred_xy.append(pred_coords)
            all_true_xy.append(true_xy)
            all_pred_vis.append(pred_vis)
            all_true_vis.append(true_vis)
            all_pred_heatmaps.append(pred_heatmaps)
            
            # Store ground truth heatmaps if available
            if gt_heatmaps is not None:
                all_true_heatmaps.append(gt_heatmaps.numpy())
            
    print("[EVAL] Finished all batches, concatenating results and computing metrics...")
    all_pred_xy = np.concatenate(all_pred_xy, axis=0)
    all_true_xy = np.concatenate(all_true_xy, axis=0)
    all_pred_vis = np.concatenate(all_pred_vis, axis=0)
    all_true_vis = np.concatenate(all_true_vis, axis=0)
    all_pred_heatmaps = np.concatenate(all_pred_heatmaps, axis=0)
    
    # Calculate core metrics
    dist_err = compute_distance_error(all_pred_xy, all_true_xy, visibility=all_true_vis)
    f1 = compute_visibility_f1(all_pred_vis, all_true_vis)
    precision, recall = compute_precision_recall(all_pred_vis, all_true_vis)
    within_dist = compute_within_distance(all_pred_xy, all_true_xy, thresholds=[5, 10], visibility=all_true_vis)
    
    metrics = {
        'distance_error': dist_err, 
        'visibility_f1': f1,
        'precision': precision,
        'recall': recall,
        **within_dist  # Add within_5px and within_10px
    }
    
    # Calculate heatmap-specific metrics if ground truth heatmaps are available
    if len(all_true_heatmaps) > 0:
        all_true_heatmaps = np.concatenate(all_true_heatmaps, axis=0)
        heatmap_metrics = compute_heatmap_metrics(all_pred_heatmaps, all_true_heatmaps, all_true_vis)
        metrics.update(heatmap_metrics)
    
    print("[EVAL] Metrics computed.")
    return metrics 