import streamlit as st
import tempfile
import os
import torch
import yaml
import cv2
import numpy as np
from models.shuttletrack import build_model_from_config
from utils.infer_utils import load_config, load_checkpoint, preprocess_frame

st.set_page_config(page_title='ShuttleTrack Inference & Visualization', layout='wide')
st.title('🏸 ShuttleTrack: Shuttlecock Detection & Trajectory Tracking')

st.markdown('''
Upload your trained model checkpoint and a badminton video. The app will run inference, track the shuttlecock, and overlay the trajectory on the video. Logs and progress will be shown below.
''')

# Upload model checkpoint and video
ckpt_file = st.file_uploader('Upload trained model checkpoint (.pth)', type=['pth'])
video_file = st.file_uploader('Upload video file', type=['mp4', 'avi', 'mov'])

if ckpt_file and video_file:
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
        preds = []
        progress = st.progress(0, text='Running inference...')
        for i in range(len(frames) - seq_len + 1):
            frame_seq = torch.tensor(frames[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            diff_seq = torch.tensor(diffs[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(frame_seq, diff_seq).cpu().numpy()[0]
            for t in range(seq_len):
                preds.append(pred[t])
            progress.progress((i+1)/(len(frames)-seq_len+1), text=f'Inference: {i+1}/{len(frames)-seq_len+1}')
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
        trajectory = []
        for _ in range(len(frames)):
            ret, frame = cap.read()
            if not ret or idx >= len(preds):
                break
            row = preds[idx]
            if row[0] > 0.5:
                x = int(row[1] * width)
                y = int(row[2] * height)
                trajectory.append((x, y))
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
            out.write(frame)
            idx += 1
        cap.release()
        out.release()
        st.video(output_path)
        st.success('Output video with trajectory overlay is shown above.') 