import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_ROOT = 'ShuttleCockFrameDataset'
OUTPUT_ROOT = 'processed_data'
SPLITS = ['Train', 'valid']
IMG_SIZE = 256


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_segment(frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file):
    frame_files = sorted(os.listdir(frames_dir))
    df = pd.read_csv(csv_file)
    prev_img = None
    labels = []
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        img = img.astype(np.float32) / 255.0
        # Frame difference
        if prev_img is not None:
            diff = cv2.absdiff(img, prev_img)
        else:
            diff = np.zeros_like(img)
        prev_img = img
        # Label
        row = df.iloc[i]
        original_h, original_w = img.shape[0], img.shape[1]
        label = [row['Visibility'], row['X'] / original_w, row['Y'] / original_h]
        labels.append(label)
        # Save processed frame and diff as images
        ensure_dir(out_frames_dir)
        ensure_dir(out_diffs_dir)
        cv2.imwrite(os.path.join(out_frames_dir, frame_file), (img * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_diffs_dir, frame_file), (diff * 255).astype(np.uint8))
        # No need to store img or diff in a list!
    np.save(out_labels_file, np.array(labels, dtype=np.float32))


def process_split(split):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]
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
    print('Preprocessing complete!')

if __name__ == '__main__':
    main() 