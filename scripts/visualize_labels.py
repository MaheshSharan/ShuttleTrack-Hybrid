import os
import numpy as np
import cv2
import random
from glob import glob
from tqdm import tqdm

# Parameters
SPLITS = ['Train', 'valid']
N_SAMPLES = 20
RESULTS_DIR = 'results/visualized_labels'
HEATMAP_ALPHA = 0.4  # Transparency for heatmap overlay

os.makedirs(RESULTS_DIR, exist_ok=True)

PROCESSED_ROOT = 'processed_data'

for split in SPLITS:
    split_dir = os.path.join(PROCESSED_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    matches = [d for d in os.listdir(split_dir) if d.startswith('match')]
    for match in matches:
        match_dir = os.path.join(split_dir, match)
        segments = os.listdir(match_dir)
        for segment in segments:
            seg_dir = os.path.join(match_dir, segment)
            frames_dir = os.path.join(seg_dir, 'frames')
            heatmaps_dir = os.path.join(seg_dir, 'heatmaps')
            if not (os.path.exists(frames_dir) and os.path.exists(heatmaps_dir)):
                continue
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            if len(frame_files) == 0:
                continue
            # Randomly sample up to N_SAMPLES frames from this segment
            sample_files = random.sample(frame_files, min(N_SAMPLES, len(frame_files)))
            for frame_file in tqdm(sample_files, desc=f"{split}/{match}/{segment}"):
                frame_path = os.path.join(frames_dir, frame_file)
                heatmap_path = os.path.join(heatmaps_dir, os.path.splitext(frame_file)[0] + '.npz')
                if not (os.path.exists(frame_path) and os.path.exists(heatmap_path)):
                    continue
                # Load RGB frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                # Load heatmap
                try:
                    heatmap = np.load(heatmap_path)['heatmap']
                except Exception as e:
                    print(f"[WARN] Could not load heatmap: {heatmap_path}")
                    continue
                # Resize heatmap to frame size
                heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Find peak (ground truth position)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
                # Overlay heatmap as color mask
                heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, heatmap_color, HEATMAP_ALPHA, 0)
                # Draw ground truth position
                cv2.circle(overlay, max_loc, 10, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                # Save visualization
                out_name = f"{split}_{match}_{segment}_{os.path.splitext(frame_file)[0]}.jpg"
                out_path = os.path.join(RESULTS_DIR, out_name)
                cv2.imwrite(out_path, overlay) 