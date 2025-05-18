import torch
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from utils.infer_utils import load_config, load_checkpoint, process_video_for_inference
from models.shuttletrack import build_model_from_config

def run_inference(model, frames, diffs, device, seq_len):
    """Run inference on frames and diffs using sliding window approach
    
    Args:
        model: ShuttleTrack model
        frames: Processed frames array (N, H, W, C)
        diffs: Median-subtracted diffs array (N, H, W, C)  
        device: Device to run inference on
        seq_len: Sequence length for model input
        
    Returns:
        predictions: Array of predictions (N, 3)
    """
    all_preds = []
    with torch.no_grad():
        # Run inference on sliding windows
        for i in tqdm(range(len(frames) - seq_len + 1), desc="Running inference"):
            # Extract sequence
            frame_seq = torch.tensor(frames[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)
            diff_seq = torch.tensor(diffs[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            # Get predictions
            pred = model(frame_seq, diff_seq).cpu().numpy()[0]  # (T, 3)
            
            # Store predictions
            for t in range(seq_len):
                if i + t >= len(all_preds):
                    all_preds.append(pred[t])
    
    # Handle any remaining frames at the end with last sequence
    if len(frames) > len(all_preds):
        remaining = len(frames) - len(all_preds)
        last_pred = all_preds[-1] if all_preds else np.zeros(3)
        for _ in range(remaining):
            all_preds.append(last_pred)
    
    return np.array(all_preds)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ShuttleTrack Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output predictions CSV')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pth', help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions with trajectory overlay')
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
    predictions = run_inference(model, frames, diffs, device, seq_len)
    
    # Convert sigmoid output to probability for visibility
    predictions[:, 0] = 1 / (1 + np.exp(-predictions[:, 0]))
    
    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=['visibility', 'x', 'y'])
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
            
            # Add prediction confidence
            confidence_text = f"Confidence: {row[1]['visibility']:.2f}"
            cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 2, cv2.LINE_AA)
            
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"Saved visualization to {output_video}")

if __name__ == '__main__':
    main() 