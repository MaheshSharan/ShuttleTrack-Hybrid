import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from utils.infer_utils import load_config, load_checkpoint, process_video_for_inference
from models.shuttletrack import build_model_from_config, extract_coordinates_from_heatmap
from utils.post_processing import create_trajectory_processor

def run_inference(model, frames, diffs, flows=None, device=None, seq_len=5):
    """Run inference on frames and diffs using sliding window approach
    
    Args:
        model: ShuttleTrack model
        frames: Processed frames array (N, H, W, C)
        diffs: Median-subtracted diffs array (N, H, W, C)  
        flows: Optional optical flow array (N, H, W, 2)
        device: Device to run inference on
        seq_len: Sequence length for model input
        
    Returns:
        Dictionary with visibility, coordinates, and heatmaps
    """
    visibility_preds = []
    coords_preds = []
    heatmap_preds = []
    
    with torch.no_grad():
        # Run inference on sliding windows
        for i in tqdm(range(len(frames) - seq_len + 1), desc="Running inference"):
            # Extract sequence
            frame_seq = torch.tensor(frames[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)
            diff_seq = torch.tensor(diffs[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            # Handle optical flow if available
            flow_seq = None
            if flows is not None:
                flow_seq = torch.tensor(flows[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            # Get predictions
            pred_dict = model(frame_seq, diff_seq, flow_seq)
            
            # Extract visibility and convert to probability
            vis_probs = torch.sigmoid(pred_dict['visibility']).cpu().numpy()[0]  # (T,)
            
            # Extract coordinates from heatmap
            coords, _ = extract_coordinates_from_heatmap(pred_dict['heatmap'])
            coords = coords.cpu().numpy()[0]  # (T, 2)
            
            # Store heatmaps for later visualization
            heatmaps = pred_dict['heatmap'].cpu().numpy()[0]  # (T, H, W)
            
            # Store predictions for each frame in the sequence
            for t in range(seq_len):
                if i + t >= len(visibility_preds):
                    visibility_preds.append(vis_probs[t])
                    coords_preds.append(coords[t])
                    heatmap_preds.append(heatmaps[t])
    
    # Handle any remaining frames at the end with last valid prediction
    if len(frames) > len(visibility_preds):
        remaining = len(frames) - len(visibility_preds)
        last_vis = visibility_preds[-1] if visibility_preds else 0
        last_coord = coords_preds[-1] if coords_preds else np.zeros(2)
        last_heatmap = heatmap_preds[-1] if heatmap_preds else np.zeros_like(heatmaps[0])
        
        for _ in range(remaining):
            visibility_preds.append(last_vis)
            coords_preds.append(last_coord)
            heatmap_preds.append(last_heatmap)
    
    return {
        'visibility': np.array(visibility_preds),  # (N,)
        'coordinates': np.array(coords_preds),     # (N, 2)
        'heatmaps': np.array(heatmap_preds)        # (N, H, W)
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ShuttleTrack Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output predictions CSV')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pth', help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions with trajectory overlay')
    # Post-processing arguments
    parser.add_argument('--post-process', action='store_true', help='Apply post-processing')
    parser.add_argument('--smoothing', type=str, default='kalman', choices=['kalman', 'savgol', 'moving_avg'],
                       help='Trajectory smoothing method')
    parser.add_argument('--window-size', type=int, default=5, help='Window size for smoothing')
    parser.add_argument('--inpaint-method', type=str, default='spline', choices=['linear', 'spline', 'pchip'],
                       help='Trajectory inpainting method')
    parser.add_argument('--vis-threshold', type=float, default=0.5, help='Visibility threshold')
    args = parser.parse_args()

    # Load config and model
    config = load_config('config/shuttletrack.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    model = build_model_from_config(config).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    seq_len = config['model']['sequence_length']
    input_size = config['model']['input_size']
    if isinstance(input_size, int):
        input_size_int = input_size
    else:
        input_size_int = input_size[0]  # Use height if it's a tuple

    print(f"Processing video: {args.video}")
    print(f"Using enhanced preprocessing with median background subtraction")
    
    # Process video to get frames and median-subtracted diffs
    frames, diffs = process_video_for_inference(args.video, input_size_int)
    
    # Run inference
    print(f"Running inference with sequence length {seq_len}")
    predictions = run_inference(model, frames, diffs, None, device, seq_len)
    
    # Apply post-processing if requested
    if args.post_process:
        print(f"Applying post-processing: {args.smoothing} smoothing with window size {args.window_size}")
        
        # Update config with command line arguments
        post_config = {
            'post_processing': {
                'smooth_mode': args.smoothing,
                'smooth_window_size': args.window_size,
                'inpaint_method': args.inpaint_method,
                'visibility_threshold': args.vis_threshold,
                'visibility_smooth_window': 3
            }
        }
        
        # Create trajectory processor
        processor = create_trajectory_processor({**config, **post_config})
        
        # Apply post-processing
        coords = predictions['coordinates'].reshape(1, -1, 2)  # Add batch dimension
        vis = predictions['visibility'].reshape(1, -1)         # Add batch dimension
        
        processed_coords, processed_vis = processor.process_predictions(coords, vis)
        
        # Update predictions
        predictions['coordinates'] = processed_coords[0]  # Remove batch dimension
        predictions['visibility'] = processed_vis[0]      # Remove batch dimension
    
    # Save predictions to CSV
    df = pd.DataFrame({
        'visibility': predictions['visibility'],
        'x': predictions['coordinates'][:, 0],
        'y': predictions['coordinates'][:, 1]
    })
    df.to_csv(args.output, index=False)
    print(f'Saved predictions to {args.output}')
    
    # Visualize predictions if requested
    if args.visualize:
        from scripts.visualize import draw_fading_trajectory, draw_modern_shuttlecock
        from collections import deque
        
        print("Creating visualization...")
        output_video = args.output.replace('.csv', '_visualized.mp4')
        
        # Open video for visualization
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Set up trajectory buffer
        buffer_len = int(fps) if fps > 0 else 20  # 1 second buffer
        trajectory = deque(maxlen=buffer_len)
        
        # Process each frame
        for i, row in tqdm(enumerate(df.iterrows()), total=len(df), desc="Creating visualization"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add point to trajectory if visible
            if row[1]['visibility'] > 0.5:
                x = int(row[1]['x'] * width)
                y = int(row[1]['y'] * height)
                trajectory.append((x, y))
                draw_modern_shuttlecock(frame, x, y)
                
            # Draw trajectory
            draw_fading_trajectory(frame, list(trajectory))
            
            # Add prediction confidence and post-processing info
            confidence_text = f"Confidence: {row[1]['visibility']:.2f}"
            cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Add post-processing info if used
            if args.post_process:
                pp_text = f"Post-processing: {args.smoothing}"
                cv2.putText(frame, pp_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"Saved visualization to {output_video}")

if __name__ == '__main__':
    main() 