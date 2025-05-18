import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

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
    # Use a reasonable subset of frames for median computation to save memory
    max_frames_for_median = 30
    
    if len(frame_files) > max_frames_for_median:
        # Use evenly spaced frames for representative sampling
        indices = np.linspace(0, len(frame_files)-1, max_frames_for_median, dtype=int)
        frame_files_subset = [frame_files[i] for i in indices]
    else:
        frame_files_subset = frame_files
        
    for frame_file in tqdm(frame_files_subset, desc="    Loading frames for median", leave=False):
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
    
    # Process frames with progress bar
    for i, frame_file in enumerate(tqdm(frame_files, desc=f"  Processing {os.path.basename(frames_dir)}", leave=False)):
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

    for match in tqdm(matches, desc=f'Processing {split}', position=0, leave=True):
        match_path = os.path.join(split_path, match)
        segments = os.listdir(os.path.join(match_path, 'frames'))
        
        print(f"\n[INFO] Processing match: {match} ({len(segments)} segments)")
        
        # Process each segment sequentially to avoid memory issues
        for segment in tqdm(segments, desc=f'  Segments in {match}', position=1, leave=True):
            frames_dir = os.path.join(match_path, 'frames', segment)
            csv_file = os.path.join(match_path, 'csv', f'{segment}_ball.csv')
            out_frames_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'frames')
            out_diffs_dir = os.path.join(OUTPUT_ROOT, split, match, segment, 'diffs')
            out_labels_file = os.path.join(OUTPUT_ROOT, split, match, segment, 'labels.npy')
            
            # Process this segment
            ensure_dir(out_frames_dir)
            ensure_dir(out_diffs_dir)
            try:
                process_segment(frames_dir, csv_file, out_frames_dir, out_diffs_dir, out_labels_file)
            except Exception as e:
                print(f"  [ERROR] Failed to process segment {segment}: {str(e)}")
                continue
        
        print(f"[INFO] Completed match: {match}\n")

def main():
    for split in SPLITS:
        process_split(split)
    print('\n[SUCCESS] Preprocessing complete!')

if __name__ == '__main__':
    main()
