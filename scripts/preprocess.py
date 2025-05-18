import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor

# Load config
def load_config(path='config/shuttletrack.yaml'):
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Warning: Could not load {path}, using default paths")
        return {}

# Get paths from config or use defaults
config = load_config()
DATASET_ROOT = config.get('data', {}).get('raw_dataset_path', 'ShuttleCockFrameDataset')
OUTPUT_ROOT = config.get('data', {}).get('processed_dataset_path', 'processed_data')
EXCLUDE_MATCHES = set(config.get('data', {}).get('exclude_matches', []))
SPLITS = ['Train', 'valid']

print(f"Using raw dataset: {DATASET_ROOT}")
print(f"Output will be saved to: {OUTPUT_ROOT}")
print(f"Excluded matches: {EXCLUDE_MATCHES}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_median_image(frames_dir, frame_files):
    print(f"  [INFO] Computing median image for {frames_dir} ...")
    imgs = []
    for frame_file in tqdm(frame_files, desc="    Loading frames for median", leave=False):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        if img is not None:
            imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError(f"No frames found in {frames_dir} for median computation.")
    median_img = np.median(np.stack(imgs, axis=0), axis=0)
    print(f"  [INFO] Median image computed for {frames_dir}.")
    return median_img

def process_segment(frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file):
    frame_files = sorted(os.listdir(frames_dir))
    df = pd.read_csv(csv_file)
    labels = []
    ensure_dir(out_frames_dir)
    ensure_dir(out_diffs_dir)
    # Compute median image for this segment
    median_img = compute_median_image(frames_dir, frame_files)
    sample_diff_stats = None
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        img = img.astype(np.float32) / 255.0
        # Median-subtracted diff
        diff = np.abs(img - (median_img / 255.0))
        # Label
        row = df.iloc[i]
        original_h, original_w = img.shape[0], img.shape[1]
        label = [row['Visibility'], row['X'] / original_w, row['Y'] / original_h]
        labels.append(label)
        # Save processed frame and diff as images
        out_frame_path = os.path.join(out_frames_dir, frame_file)
        out_diff_path = os.path.join(out_diffs_dir, frame_file)
        cv2.imwrite(out_frame_path, (img * 255).astype(np.uint8))
        cv2.imwrite(out_diff_path, (diff * 255).astype(np.uint8))
        # Log stats for the first diff image
        if i == 0:
            sample_diff_stats = {
                'path': out_diff_path,
                'min': float(diff.min()),
                'max': float(diff.max()),
                'mean': float(diff.mean()),
                'std': float(diff.std())
            }
    np.save(out_labels_file, np.array(labels, dtype=np.float32))
    print(f"  [INFO] Processed {len(frame_files)} frames in {frames_dir}.")
    if sample_diff_stats:
        print(f"  [DEBUG] Sample diff image: {sample_diff_stats['path']} | min: {sample_diff_stats['min']:.4f}, max: {sample_diff_stats['max']:.4f}, mean: {sample_diff_stats['mean']:.4f}, std: {sample_diff_stats['std']:.4f}")

def process_split(split):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]
    matches = [m for m in matches if m not in EXCLUDE_MATCHES]

    with tqdm(matches, desc=f'Processing {split}', position=0, leave=True) as match_bar:
        for match in match_bar:
            match_path = os.path.join(split_path, match)
            segments = os.listdir(os.path.join(match_path, 'frames'))
            segment_args = []
            for segment in segments:
                frames_dir = os.path.join(match_path, 'frames', segment)
                csv_file = os.path.join(match_path, 'csv', f'{segment}_ball.csv')
                out_frames_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'frames')
                out_diffs_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'diffs')
                out_labels_file = os.path.join(OUTPUT_ROOT, split, match, segment, 'labels.npy')
                ensure_dir(out_frames_dir)
                ensure_dir(out_diffs_dir)
                segment_args.append((frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file))
            # Parallel processing of segments
            with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
                list(tqdm(executor.map(lambda args: process_segment(*args), segment_args), total=len(segment_args), desc=f'  {match}', position=1, leave=False))

def main():
    for split in SPLITS:
        process_split(split)
    print('Preprocessing complete!')

if __name__ == '__main__':
    main()
