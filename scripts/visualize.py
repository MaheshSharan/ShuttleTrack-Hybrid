import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay shuttlecock trajectory on video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--output', type=str, default='output_with_trajectory.mp4', help='Path to output video')
    args = parser.parse_args()

    # Load predictions
    df = pd.read_csv(args.predictions)
    # Open video
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    idx = 0
    trajectory = []
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(df):
            break
        row = df.iloc[idx]
        if row['visibility'] > 0.5:
            x = int(row['x'] * width)
            y = int(row['y'] * height)
            trajectory.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red dot for shuttlecock
        # Draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)  # Green line
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f'Saved output video with trajectory overlay to {args.output}') 