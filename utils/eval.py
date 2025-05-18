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


def heatmap_to_xy(heatmap):
    # heatmap: (B, T, 1, H, W) or (B, T, H, W)
    # Returns: (B, T, 2) in normalized [0, 1] coordinates
    if heatmap.ndim == 5:
        heatmap = heatmap.squeeze(2)  # (B, T, H, W)
    B, T, H, W = heatmap.shape
    flat = heatmap.reshape(B, T, -1)
    idx = flat.argmax(-1)  # (B, T)
    y = idx // W
    x = idx % W
    x = x.float() / (W - 1)
    y = y.float() / (H - 1)
    xy = torch.stack([x, y], dim=-1)  # (B, T, 2)
    return xy


class Kalman2D:
    """Simple 2D constant velocity Kalman filter for trajectory smoothing."""
    def __init__(self, process_var=1e-3, measurement_var=1e-2):
        # State: [x, y, vx, vy]
        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Q = process_var * np.eye(4, dtype=np.float32)
        self.R = measurement_var * np.eye(2, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False
    def reset(self):
        self.P = np.eye(4, dtype=np.float32)
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False
    def step(self, z):
        # z: [x, y] measurement
        if not self.initialized:
            self.x[:2, 0] = z
            self.x[2:, 0] = 0
            self.initialized = True
        # Predict
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        # Update
        y = z.reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2, 0].copy()

def rectify_trajectory_kalman(xy_seq, vis_seq=None):
    """Apply Kalman filter to a sequence of (x, y) predictions. Only smooth visible frames if vis_seq is given."""
    N, T, _ = xy_seq.shape
    xy_rect = np.zeros_like(xy_seq)
    for i in range(N):
        kf = Kalman2D()
        for t in range(T):
            if vis_seq is not None and vis_seq[i, t] < 0.5:
                xy_rect[i, t] = xy_seq[i, t]
            else:
                xy_rect[i, t] = kf.step(xy_seq[i, t])
    return xy_rect

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
            true_vis = batch['visibility'].cpu().numpy()
            true_heatmap = batch['heatmap'].cpu().numpy()
            pred = model(frames, diffs)
            pred_heatmap = pred['heatmap'].cpu()
            pred_vis = torch.sigmoid(pred['visibility']).cpu().numpy()
            # Convert heatmap to (x, y)
            pred_xy = heatmap_to_xy(pred['heatmap']).cpu().numpy()  # (B, T, 2)
            # True (x, y) from heatmap (use argmax)
            true_xy = heatmap_to_xy(torch.from_numpy(true_heatmap)).cpu().numpy()
            all_pred_xy.append(pred_xy)
            all_true_xy.append(true_xy)
            all_pred_vis.append(pred_vis)
            all_true_vis.append(true_vis)
    print("[EVAL] Finished all batches, concatenating results and computing metrics...")
    all_pred_xy = np.concatenate(all_pred_xy, axis=0)
    all_true_xy = np.concatenate(all_true_xy, axis=0)
    all_pred_vis = np.concatenate(all_pred_vis, axis=0)
    all_true_vis = np.concatenate(all_true_vis, axis=0)
    # --- Kalman rectification ---
    all_pred_xy_rect = rectify_trajectory_kalman(all_pred_xy, all_pred_vis)
    # Calculate metrics (rectified)
    dist_err = compute_distance_error(all_pred_xy_rect, all_true_xy, visibility=all_true_vis)
    f1 = compute_visibility_f1(all_pred_vis, all_true_vis)
    precision, recall = compute_precision_recall(all_pred_vis, all_true_vis)
    within_dist = compute_within_distance(all_pred_xy_rect, all_true_xy, thresholds=[5, 10], visibility=all_true_vis)
    print("[EVAL] Metrics computed (with Kalman rectification).")
    return {
        'distance_error': dist_err, 
        'visibility_f1': f1,
        'precision': precision,
        'recall': recall,
        **within_dist  # Add within_5px and within_10px
    } 