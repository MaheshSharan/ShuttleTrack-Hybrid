import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


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


def evaluate(model, dataloader, device):
    print("[EVAL] Starting evaluation...")
    model.eval()
    all_pred_xy, all_true_xy = [], []
    all_pred_vis, all_true_vis = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"[EVAL] Processing batch {i+1}/{len(dataloader)}")
            frames = batch['frames'].to(device)
            diffs = batch['diffs'].to(device)
            labels = batch['labels'].to(device)
            pred = model(frames, diffs)  # (B, T, 3)
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            # Visibility (after sigmoid)
            pred_vis = 1 / (1 + np.exp(-pred[..., 0]))
            true_vis = labels[..., 0]
            # (x, y)
            pred_xy = pred[..., 1:]
            true_xy = labels[..., 1:]
            all_pred_xy.append(pred_xy)
            all_true_xy.append(true_xy)
            all_pred_vis.append(pred_vis)
            all_true_vis.append(true_vis)
    print("[EVAL] Finished all batches, concatenating results and computing metrics...")
    all_pred_xy = np.concatenate(all_pred_xy, axis=0)
    all_true_xy = np.concatenate(all_true_xy, axis=0)
    all_pred_vis = np.concatenate(all_pred_vis, axis=0)
    all_true_vis = np.concatenate(all_true_vis, axis=0)
    
    # Calculate metrics
    dist_err = compute_distance_error(all_pred_xy, all_true_xy, visibility=all_true_vis)
    f1 = compute_visibility_f1(all_pred_vis, all_true_vis)
    precision, recall = compute_precision_recall(all_pred_vis, all_true_vis)
    within_dist = compute_within_distance(all_pred_xy, all_true_xy, thresholds=[5, 10], visibility=all_true_vis)
    print("[EVAL] Metrics computed.")
    return {
        'distance_error': dist_err, 
        'visibility_f1': f1,
        'precision': precision,
        'recall': recall,
        **within_dist  # Add within_5px and within_10px
    } 