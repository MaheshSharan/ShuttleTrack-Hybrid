import yaml
import torch
import cv2
import numpy as np
from collections import deque
from utils.post_processing import create_trajectory_processor

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

def preprocess_frame(frame, size=224):
    """Preprocess a single frame for inference (RGB conversion and resize)"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if isinstance(size, tuple):
        frame = cv2.resize(frame, (size[1], size[0]))  # cv2.resize takes (width, height)
    else:
        frame = cv2.resize(frame, (size, size))
    frame = frame.astype(float) / 255.0
    return frame

def compute_median_background(frames, max_frames=30):
    """Compute median background from a sequence of frames
    
    Args:
        frames: List of frames to compute median from
        max_frames: Maximum number of frames to use for median
        
    Returns:
        median_frame: Median frame
    """
    # Sample frames if there are too many
    if len(frames) > max_frames:
        # Use evenly spaced frames for more representative sampling
        sample_indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames_to_use = [frames[i] for i in sample_indices]
    else:
        frames_to_use = frames
    
    # Stack frames and compute median
    stacked_frames = np.stack(frames_to_use, axis=0)
    median_frame = np.median(stacked_frames, axis=0)
    
    return median_frame

def create_median_subtracted_diff(frame, median_bg):
    """Create median-subtracted diff from frame and median background
    
    Args:
        frame: Current frame
        median_bg: Median background frame
        
    Returns:
        diff: Absolute difference between frame and median background
    """
    # Ensure both are float arrays with same range [0-1]
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32)
    if median_bg.dtype != np.float32:
        median_bg = median_bg.astype(np.float32)
        
    # Calculate absolute difference
    diff = np.abs(frame - median_bg)
    
    return diff

def compute_optical_flow(prev_frame, curr_frame):
    """Compute optical flow between two frames using Farneback method.
    
    Args:
        prev_frame: Previous frame (H, W, C)
        curr_frame: Current frame (H, W, C)
        
    Returns:
        flow: Optical flow map (H, W, 2)
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    
    # Calculate flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    return flow

def process_video_for_inference(video_path, input_size=224, compute_flow=False):
    """Process a video for inference, returning frames and median-subtracted diffs
    
    Args:
        video_path: Path to video file
        input_size: Size to resize frames to
        compute_flow: Whether to compute optical flow
        
    Returns:
        frames: Processed frames
        diffs: Median-subtracted diffs
        flows: Optional optical flow maps (if compute_flow=True)
    """
    # Read video frames
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    
    # Preprocess frames
    processed_frames = [preprocess_frame(frame, input_size) for frame in raw_frames]
    
    # Compute median background
    median_bg = compute_median_background(processed_frames)
    
    # Create median-subtracted diffs
    diffs = [create_median_subtracted_diff(frame, median_bg) for frame in processed_frames]
    
    # Compute optical flow if requested
    if compute_flow:
        flows = [np.zeros((processed_frames[0].shape[0], processed_frames[0].shape[1], 2), dtype=np.float32)]
        for i in range(1, len(processed_frames)):
            flow = compute_optical_flow(
                (processed_frames[i-1] * 255).astype(np.uint8),
                (processed_frames[i] * 255).astype(np.uint8)
            )
            flows.append(flow)
        return np.stack(processed_frames), np.stack(diffs), np.stack(flows)
    
    return np.stack(processed_frames), np.stack(diffs)

def apply_post_processing(predictions, config):
    """Apply post-processing to model predictions.
    
    Args:
        predictions: Dictionary with keys 'visibility' and 'coordinates'
        config: Dictionary with post-processing configuration
        
    Returns:
        Processed predictions
    """
    # Create trajectory processor
    processor = create_trajectory_processor(config)
    
    # Reshape to add batch dimension
    coords = predictions['coordinates'].reshape(1, -1, 2)
    vis = predictions['visibility'].reshape(1, -1)
    
    # Apply post-processing
    processed_coords, processed_vis = processor.process_predictions(coords, vis)
    
    # Update predictions and return
    result = predictions.copy()
    result['coordinates'] = processed_coords[0]  # Remove batch dimension
    result['visibility'] = processed_vis[0]      # Remove batch dimension
    
    return result

def save_predictions_to_csv(predictions, output_path):
    """Save predictions to CSV file.
    
    Args:
        predictions: Dictionary with keys 'visibility' and 'coordinates'
        output_path: Path to save CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'visibility': predictions['visibility'],
        'x': predictions['coordinates'][:, 0],
        'y': predictions['coordinates'][:, 1]
    })
    
    df.to_csv(output_path, index=False)
    return df 