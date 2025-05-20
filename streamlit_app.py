import streamlit as st
import tempfile
import os
import torch
import yaml
import cv2
import numpy as np
from models.shuttletrack import build_model_from_config, extract_coordinates_from_heatmap
from utils.infer_utils import load_config, load_checkpoint, preprocess_frame
from utils.post_processing import create_trajectory_processor
import shutil
from collections import deque
# Import drawing functions from visualize.py
from scripts.visualize import draw_fading_trajectory, draw_modern_shuttlecock

st.set_page_config(page_title='ShuttleTrack Inference & Visualization', layout='wide')
st.title('🏸 ShuttleTrack: Shuttlecock Detection & Trajectory Tracking')

st.markdown('''
Upload your trained model checkpoint and a badminton video. The app will run inference, track the shuttlecock, and overlay the trajectory on the video. Logs and progress will be shown below.
''')

# Upload model checkpoint and video
ckpt_file = st.file_uploader('Upload trained model checkpoint (.pth)', type=['pth'])
video_file = st.file_uploader('Upload video file', type=['mp4', 'avi', 'mov'])

# Post-processing options
st.sidebar.header("Post-Processing Options")
enable_post_processing = st.sidebar.checkbox("Enable Post-Processing", value=True)

if enable_post_processing:
    # Smoothing method
    smoothing_method = st.sidebar.selectbox(
        "Trajectory Smoothing Method",
        ["kalman", "savgol", "moving_avg"],
        help="Kalman: motion model, SavGol: polynomial fit, Moving Avg: simple averaging"
    )
    
    # Window size
    window_size = st.sidebar.slider(
        "Smoothing Window Size", 
        min_value=3, 
        max_value=11, 
        value=5, 
        step=2,
        help="Larger window = smoother trajectory but potentially less accurate"
    )
    
    # Inpainting method
    inpainting_method = st.sidebar.selectbox(
        "Trajectory Inpainting Method", 
        ["spline", "linear", "pchip"],
        help="Method to fill in missing points during occlusions"
    )
    
    # Visibility threshold
    visibility_threshold = st.sidebar.slider(
        "Visibility Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Threshold to determine if shuttlecock is visible"
    )
    
    # Visibility smoothing window
    vis_smooth_window = st.sidebar.slider(
        "Visibility Smoothing Window",
        min_value=3,
        max_value=9,
        value=3,
        step=2,
        help="Window size for temporal smoothing of visibility predictions"
    )

if ckpt_file and video_file:
    if st.button('Track Shuttlecock'):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'model.pth')
            video_path = os.path.join(tmpdir, 'input_video.mp4')
            with open(ckpt_path, 'wb') as f:
                f.write(ckpt_file.read())
            with open(video_path, 'wb') as f:
                f.write(video_file.read())
            # Load config
            config = load_config('config/shuttletrack.yaml')
            device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
            model = build_model_from_config(config).to(device)
            model = load_checkpoint(model, ckpt_path, device)
            model.eval()
            seq_len = config['model']['sequence_length']
            input_size = config['model']['input_size']
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(preprocess_frame(frame, input_size))
            cap.release()
            frames = np.stack(frames, axis=0)
            # Compute frame diffs
            diffs = [np.zeros_like(frames[0])]
            for i in range(1, len(frames)):
                diffs.append(np.abs(frames[i] - frames[i-1]))
            diffs = np.stack(diffs, axis=0)
            
            # Run inference in sliding windows
            visibility_preds = []
            coords_preds = []
            heatmap_preds = []
            
            progress = st.progress(0, text='Running inference...')
            for i in range(len(frames) - seq_len + 1):
                frame_seq = torch.tensor(frames[i:i+seq_len], dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                diff_seq = torch.tensor(diffs[i:i+seq_len], dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_dict = model(frame_seq, diff_seq)
                
                # Extract visibility (sigmoid applied) and coordinates
                vis_probs = torch.sigmoid(pred_dict['visibility']).cpu().numpy()[0]  # (T,)
                coords, _ = extract_coordinates_from_heatmap(pred_dict['heatmap'])
                coords = coords.cpu().numpy()[0]  # (T, 2)
                heatmaps = pred_dict['heatmap'].cpu().numpy()[0]  # (T, H, W)
                
                for t in range(seq_len):
                    if i + t >= len(visibility_preds):
                        visibility_preds.append(vis_probs[t])
                        coords_preds.append(coords[t])
                        heatmap_preds.append(heatmaps[t])
                        st.write(f"Frame {i+t}: visibility={vis_probs[t]:.3f}, x={coords[t, 0]:.3f}, y={coords[t, 1]:.3f}")
                
                progress.progress((i+1)/(len(frames)-seq_len+1), text=f'Inference: {i+1}/{len(frames)-seq_len+1}')
            
            # Handle remaining frames
            if len(frames) > len(visibility_preds):
                remaining = len(frames) - len(visibility_preds)
                last_vis = visibility_preds[-1] if visibility_preds else 0
                last_coord = coords_preds[-1] if coords_preds else np.zeros(2)
                last_heatmap = heatmap_preds[-1] if heatmap_preds else np.zeros_like(heatmaps[0])
                
                for _ in range(remaining):
                    visibility_preds.append(last_vis)
                    coords_preds.append(last_coord)
                    heatmap_preds.append(last_heatmap)
            
            # Package predictions
            predictions = {
                'visibility': np.array(visibility_preds),
                'coordinates': np.array(coords_preds),
                'heatmaps': np.array(heatmap_preds)
            }
            
            # Apply post-processing if enabled
            if enable_post_processing:
                st.info(f"Applying post-processing: {smoothing_method} smoothing with window size {window_size}")
                
                # Create trajectory processor with selected options
                post_config = {
                    'post_processing': {
                        'smooth_mode': smoothing_method,
                        'smooth_window_size': window_size,
                        'inpaint_method': inpainting_method,
                        'visibility_threshold': visibility_threshold,
                        'visibility_smooth_window': vis_smooth_window
                    }
                }
                
                # Create trajectory processor
                processor = create_trajectory_processor({**config, **post_config})
                
                # Apply post-processing (add batch dimension, then remove it)
                coords = predictions['coordinates'].reshape(1, -1, 2)
                vis = predictions['visibility'].reshape(1, -1)
                
                processed_coords, processed_vis = processor.process_predictions(coords, vis)
                
                # Update predictions
                predictions['coordinates'] = processed_coords[0]
                predictions['visibility'] = processed_vis[0]
            
            st.success('Inference complete!')
            
            # Overlay trajectory
            output_path = os.path.join(tmpdir, 'output_with_trajectory.mp4')
            # Rewind video
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            idx = 0
            # Use a buffer for fading trajectory (1 second or 20 frames if fps unknown)
            buffer_len = int(fps) if fps > 0 else 20
            trajectory = deque(maxlen=buffer_len)
            
            # Create progress bar for visualization
            vis_progress = st.progress(0, text='Creating visualization...')
            total_frames = len(frames)
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or idx >= len(predictions['visibility']):
                    break
                
                # Get predictions for current frame
                visibility = predictions['visibility'][idx]
                coords = predictions['coordinates'][idx]
                
                if visibility > visibility_threshold:
                    x = int(coords[0] * width)
                    y = int(coords[1] * height)
                    trajectory.append((x, y))
                    draw_modern_shuttlecock(frame, x, y)
                
                # Draw fading trajectory
                draw_fading_trajectory(frame, list(trajectory))
                
                # Add prediction confidence
                confidence_text = f"Confidence: {visibility:.2f}"
                cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Add post-processing info if used
                if enable_post_processing:
                    pp_text = f"Post-processing: {smoothing_method}"
                    cv2.putText(frame, pp_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                out.write(frame)
                idx += 1
                
                # Update progress
                vis_progress.progress((frame_idx+1)/total_frames, text=f'Visualizing: {frame_idx+1}/{total_frames}')
            
            cap.release()
            out.release()
            
            # Save output video to results folder
            os.makedirs('results', exist_ok=True)
            shutil.copy(output_path, 'results/output_with_trajectory.mp4')
            
            st.video(output_path)
            st.success('Output video with trajectory overlay is shown above.')
            
            # Display post-processing summary if enabled
            if enable_post_processing:
                st.info(f"""
                Post-processing applied:
                - Method: {smoothing_method}
                - Window size: {window_size}
                - Inpainting: {inpainting_method}
                - Visibility threshold: {visibility_threshold}
                """)
            
            st.info('Output video also saved to results/output_with_trajectory.mp4') 