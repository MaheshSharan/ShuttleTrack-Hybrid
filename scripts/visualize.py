import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from collections import deque

def draw_fading_trajectory(frame, trajectory, color_start=(0,255,255), color_end=(0,128,255)):
    n = len(trajectory)
    for i in range(1, n):
        alpha = i / n
        color = (
            int(color_start[0] * (1-alpha) + color_end[0] * alpha),
            int(color_start[1] * (1-alpha) + color_end[1] * alpha),
            int(color_start[2] * (1-alpha) + color_end[2] * alpha),
        )
        overlay = frame.copy()
        fade_strength = 0.2 + 0.8 * (i / n)
        cv2.line(overlay, trajectory[i-1], trajectory[i], color, 16, cv2.LINE_AA)
        cv2.addWeighted(overlay, fade_strength, frame, 1-fade_strength, 0, frame)
        cv2.line(frame, trajectory[i-1], trajectory[i], color, 6, cv2.LINE_AA)

def draw_modern_shuttlecock(frame, x, y):
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), 18, (255,255,255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
    cv2.circle(frame, (x, y), 10, (255,255,255), 2, cv2.LINE_AA)

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
    buffer_len = int(fps) if fps > 0 else 20  # 1 second buffer, fallback to 20 if fps unknown
    trajectory = deque(maxlen=buffer_len)
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Overlaying trajectory")
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(df):
            break
        row = df.iloc[idx]
        if row['visibility'] > 0.5:
            x = int(row['x'] * width)
            y = int(row['y'] * height)
            trajectory.append((x, y))
            draw_modern_shuttlecock(frame, x, y)
        # Draw only the fading trajectory
        draw_fading_trajectory(frame, list(trajectory))
        out.write(frame)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    print(f'Saved output video with trajectory overlay to {args.output}') 