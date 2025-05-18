import yaml
import torch
import cv2
import numpy as np
from collections import deque

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

def process_video_for_inference(video_path, input_size=224):
    """Process a video for inference, returning frames and median-subtracted diffs
    
    Args:
        video_path: Path to video file
        input_size: Size to resize frames to
        
    Returns:
        frames: Processed frames
        diffs: Median-subtracted diffs
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
    
    return np.stack(processed_frames), np.stack(diffs) 