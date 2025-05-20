#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
from utils.post_processing import create_trajectory_processor
from models.shuttletrack import build_model_from_config, extract_coordinates_from_heatmap
from torch.utils.data import DataLoader
from utils.dataset import ShuttleTrackDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_config(path):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def visualize_trajectories(frames, original_coords, processed_coords, 
                         original_vis, processed_vis, output_path):
    """Visualize original and processed trajectories on frames.
    
    Args:
        frames: List of frames (T, H, W, C)
        original_coords: Original coordinates (T, 2)
        processed_coords: Processed coordinates (T, 2)
        original_vis: Original visibility (T,)
        processed_vis: Processed visibility (T,)
        output_path: Path to save visualization
    """
    T = len(frames)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create copies of the first frame for visualization
    orig_img = frames[0].copy()
    proc_img = frames[0].copy()
    
    # Draw trajectories
    for t in range(T):
        # Scale coordinates to pixel values
        H, W = orig_img.shape[:2]
        orig_x, orig_y = int(original_coords[t, 0] * W), int(original_coords[t, 1] * H)
        proc_x, proc_y = int(processed_coords[t, 0] * W), int(processed_coords[t, 1] * H)
        
        # Draw original trajectory
        alpha = min(1.0, original_vis[t] + 0.3)  # Adjust visibility for better visualization
        color = (0, int(255 * (1-alpha)), int(255 * alpha))  # Red->Green based on visibility
        cv2.circle(orig_img, (orig_x, orig_y), 3, color, -1)
        
        # Connect points
        if t > 0:
            prev_x, prev_y = int(original_coords[t-1, 0] * W), int(original_coords[t-1, 1] * H)
            cv2.line(orig_img, (prev_x, prev_y), (orig_x, orig_y), color, 1)
        
        # Draw processed trajectory
        alpha = min(1.0, processed_vis[t] + 0.3)
        color = (0, int(255 * (1-alpha)), int(255 * alpha))
        cv2.circle(proc_img, (proc_x, proc_y), 3, color, -1)
        
        # Connect points
        if t > 0:
            prev_x, prev_y = int(processed_coords[t-1, 0] * W), int(processed_coords[t-1, 1] * H)
            cv2.line(proc_img, (prev_x, prev_y), (proc_x, proc_y), color, 1)
    
    # Show images
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Trajectory')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Processed Trajectory')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")

def process_single_sequence(model, processor, dataloader, sequence_idx, device, output_dir):
    """Process a single sequence from the dataset.
    
    Args:
        model: The trained model
        processor: The trajectory processor
        dataloader: DataLoader containing the sequence
        sequence_idx: Index of the sequence to process
        device: Device to run inference on
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch the sequence
    for i, batch in enumerate(dataloader):
        if i != sequence_idx:
            continue
            
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        flows = batch.get('flows')
        if flows is not None:
            flows = flows.to(device)
        ground_truth = batch['labels'].cpu().numpy()
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(frames, diffs, flows)
            
        # Extract coordinates and visibility
        pred_heatmaps = predictions['heatmap']
        pred_vis = torch.sigmoid(predictions['visibility']).cpu().numpy()
        pred_coords, _ = extract_coordinates_from_heatmap(pred_heatmaps)
        pred_coords = pred_coords.cpu().numpy()
        
        # Get ground truth
        gt_vis = ground_truth[0, :, 0]  # (T,)
        gt_coords = ground_truth[0, :, 1:3]  # (T, 2)
        
        # Apply post-processing
        processed_coords, processed_vis = processor.process_predictions(
            pred_coords, pred_vis)
        
        # Convert frames to numpy for visualization
        frame_images = []
        for t in range(frames.shape[1]):
            # Convert from tensor [C,H,W] to numpy [H,W,C]
            frame = frames[0, t].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize and convert to uint8
            frame = frame * 255.0
            frame = frame.astype(np.uint8)
            
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_images.append(frame)
        
        # First visualization: original vs. processed predictions
        visualize_trajectories(
            frame_images,
            pred_coords[0],  # First batch item
            processed_coords[0],  # First batch item
            pred_vis[0],  # First batch item
            processed_vis[0],  # First batch item
            os.path.join(output_dir, f'sequence_{sequence_idx}_pred_vs_processed.png')
        )
        
        # Second visualization: ground truth vs. processed predictions
        visualize_trajectories(
            frame_images,
            gt_coords,
            processed_coords[0],  # First batch item
            gt_vis,
            processed_vis[0],  # First batch item
            os.path.join(output_dir, f'sequence_{sequence_idx}_gt_vs_processed.png')
        )
        
        # Save numerical results
        np.savez(
            os.path.join(output_dir, f'sequence_{sequence_idx}_results.npz'),
            frames=frames.cpu().numpy(),
            original_coords=pred_coords[0],
            processed_coords=processed_coords[0],
            original_vis=pred_vis[0],
            processed_vis=processed_vis[0],
            gt_coords=gt_coords,
            gt_vis=gt_vis
        )
        
        return
        
    print(f"Sequence {sequence_idx} not found in dataset.")

def process_all_sequences(model, processor, dataloader, device, output_dir, num_examples=5):
    """Process multiple sequences and generate visualizations.
    
    Args:
        model: The trained model
        processor: The trajectory processor
        dataloader: DataLoader containing the sequences
        device: Device to run inference on
        output_dir: Directory to save outputs
        num_examples: Number of examples to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process specified number of sequences
    for i, batch in enumerate(dataloader):
        if i >= num_examples:
            break
            
        frames = batch['frames'].to(device)
        diffs = batch['diffs'].to(device)
        flows = batch.get('flows')
        if flows is not None:
            flows = flows.to(device)
        ground_truth = batch['labels'].cpu().numpy()
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(frames, diffs, flows)
            
        # Extract coordinates and visibility
        pred_heatmaps = predictions['heatmap']
        pred_vis = torch.sigmoid(predictions['visibility']).cpu().numpy()
        pred_coords, _ = extract_coordinates_from_heatmap(pred_heatmaps)
        pred_coords = pred_coords.cpu().numpy()
        
        # Apply post-processing
        processed_coords, processed_vis = processor.process_predictions(
            pred_coords, pred_vis)
        
        # Convert frames to numpy for visualization
        frame_images = []
        for t in range(frames.shape[1]):
            # Convert from tensor [C,H,W] to numpy [H,W,C]
            frame = frames[0, t].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize and convert to uint8
            frame = frame * 255.0
            frame = frame.astype(np.uint8)
            
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_images.append(frame)
        
        # Visualize original vs. processed predictions
        visualize_trajectories(
            frame_images,
            pred_coords[0],  # First batch item
            processed_coords[0],  # First batch item
            pred_vis[0],  # First batch item
            processed_vis[0],  # First batch item
            os.path.join(output_dir, f'sequence_{i}_pred_vs_processed.png')
        )
        
        print(f"Processed sequence {i}")

def main():
    parser = argparse.ArgumentParser(description='Apply post-processing to model predictions')
    parser.add_argument('--config', default='config/shuttletrack.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='results/post_processing', help='Output directory')
    parser.add_argument('--sequence_idx', type=int, default=None, help='Process specific sequence index')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to process')
    parser.add_argument('--dataset_split', default='valid', choices=['Train', 'valid', 'test'], 
                        help='Dataset split to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = build_model_from_config(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded model from {args.checkpoint}')
    
    # Create trajectory processor
    processor = create_trajectory_processor(config)
    print(f'Created trajectory processor with mode: {processor.mode}')
    
    # Create dataset and dataloader
    input_size_cfg = config['model']['input_size']
    if isinstance(input_size_cfg, int):
        input_size_tuple = (input_size_cfg, input_size_cfg)
    else:
        input_size_tuple = tuple(input_size_cfg)
        
    dataset = ShuttleTrackDataset(
        config['data']['processed_dataset_path'], 
        split=args.dataset_split, 
        sequence_length=config['model']['sequence_length'], 
        augment=False, 
        input_size=input_size_tuple,
        use_optical_flow=config['model'].get('use_optical_flow', True),
        use_heatmaps=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Process one sequence at a time 
        shuffle=False, 
        num_workers=1, 
        pin_memory=(device.type == 'cuda')
    )
    
    # Process sequences
    if args.sequence_idx is not None:
        # Process specific sequence
        process_single_sequence(model, processor, dataloader, args.sequence_idx, device, args.output_dir)
    else:
        # Process multiple sequences
        process_all_sequences(model, processor, dataloader, device, args.output_dir, args.num_examples)
    
    print('Post-processing complete!')

if __name__ == '__main__':
    main() 