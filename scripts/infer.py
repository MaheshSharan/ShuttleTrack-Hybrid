import torch
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from utils.infer_utils import load_config, load_checkpoint, preprocess_frame
from models.shuttletrack import build_model_from_config

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ShuttleTrack Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output predictions CSV')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pth', help='Path to model checkpoint')
    args = parser.parse_args()

    config = load_config('config/shuttletrack.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    model = build_model_from_config(config).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    seq_len = config['model']['sequence_length']
    input_size = config['model']['input_size']

    # Read video
    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, input_size))
    cap.release()
    frames = np.stack(frames, axis=0)  # (N, H, W, C)
    # Compute frame diffs
    diffs = [np.zeros_like(frames[0])]
    for i in range(1, len(frames)):
        diffs.append(np.abs(frames[i] - frames[i-1]))
    diffs = np.stack(diffs, axis=0)
    # Run inference in sliding windows
    preds = []
    for i in tqdm(range(len(frames) - seq_len + 1)):
        frame_seq = torch.tensor(frames[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)
        diff_seq = torch.tensor(diffs[i:i+seq_len]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(frame_seq, diff_seq).cpu().numpy()[0]  # (T, 3)
        for t in range(seq_len):
            preds.append(pred[t])
    # Save predictions
    import pandas as pd
    df = pd.DataFrame(preds, columns=['visibility', 'x', 'y'])
    df.to_csv(args.output, index=False)
    print(f'Saved predictions to {args.output}') 