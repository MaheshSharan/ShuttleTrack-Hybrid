import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque

# Set your paths for preprocessed data (test version)
frames_dir = r"processed_data_test_match1/Train/match1/1_01_00/frames"
labels_path = r"processed_data_test_match1/Train/match1/1_01_00/labels.npy"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "annotation_visualization_from_preprocessed.mp4")

# Load labels
labels = np.load(labels_path)  # shape: (N, 3)
frame_files = sorted(os.listdir(frames_dir))

# Prepare video writer
sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
h, w, _ = sample_frame.shape
print(f"Frame shape: {sample_frame.shape}")
print(f"Num frames: {len(frame_files)}")
print(f"Num labels: {len(labels)}")
print(f"First 5 labels: {labels[:5]}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, 20, (w, h))

# Modern trajectory and shuttlecock drawing functions
def draw_fading_trajectory(frame, trajectory, color_start=(0,255,255), color_end=(0,128,255)):
    n = len(trajectory)
    for i in range(1, n):
        alpha = i / n
        color = (
            int(color_start[0] * (1-alpha) + color_end[0] * alpha),
            int(color_start[1] * (1-alpha) + color_end[1] * alpha),
            int(color_start[2] * (1-alpha) + color_end[2] * alpha),
        )
        # Fade effect: older segments are more transparent
        overlay = frame.copy()
        fade_strength = 0.2 + 0.8 * (i / n)  # Newer segments are more visible
        cv2.line(overlay, trajectory[i-1], trajectory[i], color, 16, cv2.LINE_AA)
        cv2.addWeighted(overlay, fade_strength, frame, 1-fade_strength, 0, frame)
        cv2.line(frame, trajectory[i-1], trajectory[i], color, 6, cv2.LINE_AA)

def draw_modern_shuttlecock(frame, x, y):
    # Outer glow
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), 18, (255,255,255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    # Main circle
    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)  # Bright yellow
    # White border
    cv2.circle(frame, (x, y), 10, (255,255,255), 2, cv2.LINE_AA)
    # Optional: add a small shadow or label

trajectory = deque(maxlen=20)  # Only keep the last 20 points (1 second at 20 FPS)

for i, frame_file in enumerate(tqdm(frame_files, total=len(frame_files), desc="Visualizing frames")):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    vis, x_norm, y_norm = labels[i]
    x = int(x_norm * w)
    y = int(y_norm * h)
    if vis > 0.5:
        trajectory.append((x, y))
        draw_modern_shuttlecock(frame, x, y)
    # Draw only the fading trajectory
    draw_fading_trajectory(frame, list(trajectory))
    out.write(frame)
out.release()
print(f"Saved annotation visualization video to {out_path}")