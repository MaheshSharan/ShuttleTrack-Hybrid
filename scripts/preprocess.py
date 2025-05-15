import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_config, ensure_dir

# Load config
config = load_config()
DATASET_ROOT = config.get('data', {}).get('raw_dataset_path', 'ShuttleCockFrameDataset')
OUTPUT_ROOT = config.get('data', {}).get('processed_dataset_path', 'processed_data')
EXCLUDE_MATCHES = config.get('data', {}).get('exclude_matches', [])  # NEW
SPLITS = ['Train', 'valid']

def process_segment(frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file):
    df = pd.read_csv(csv_file)
    df = df.dropna().reset_index(drop=True)

    labels = []
    for i, row in df.iterrows():
        frame_file = os.path.join(frames_dir, f"{int(row['frame'])}.jpg")
        if not os.path.exists(frame_file):
            continue

        frame = cv2.imread(frame_file)
        next_frame_file = os.path.join(frames_dir, f"{int(row['frame']) + 1}.jpg")
        if not os.path.exists(next_frame_file):
            continue

        next_frame = cv2.imread(next_frame_file)
        diff = cv2.absdiff(next_frame, frame)

        frame_out_path = os.path.join(out_frames_dir, f"{i:04d}.jpg")
        diff_out_path = os.path.join(out_diffs_dir, f"{i:04d}.jpg")
        cv2.imwrite(frame_out_path, frame)
        cv2.imwrite(diff_out_path, diff)

        labels.append([row['x'], row['y']])

    if labels:
        np.save(out_labels_file, np.array(labels, dtype=np.float32))

def process_split(split):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]

    # Exclude matches defined in config
    matches = [m for m in matches if m not in EXCLUDE_MATCHES]
    print(f"[INFO] Excluding matches in '{split}': {EXCLUDE_MATCHES}")

    with tqdm(matches, desc=f'Processing {split}', position=0, leave=True) as match_bar:
        for match in match_bar:
            match_path = os.path.join(split_path, match)
            segments = os.listdir(os.path.join(match_path, 'frames'))
            with tqdm(segments, desc=f'  {match}', position=1, leave=False) as segment_bar:
                for segment in segment_bar:
                    frames_dir = os.path.join(match_path, 'frames', segment)
                    csv_file = os.path.join(match_path, 'csv', f'{segment}_ball.csv')
                    out_frames_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'frames')
                    out_diffs_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'diffs')
                    out_labels_file = os.path.join(OUTPUT_ROOT, split, match, segment, 'labels.npy')
                    ensure_dir(out_frames_dir)
                    ensure_dir(out_diffs_dir)
                    process_segment(frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file)

def main():
    for split in SPLITS:
        process_split(split)

if __name__ == '__main__':
    main()
