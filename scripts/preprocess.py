import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import shutil
from PIL import Image
from scipy.ndimage import gaussian_filter

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

def generate_heatmap(x, y, H, W, sigma=2):
    # x, y are normalized [0, 1] coordinates
    cx = int(x * W)
    cy = int(y * H)
    heatmap = np.zeros((H, W), dtype=np.float32)
    if 0 <= cx < W and 0 <= cy < H:
        heatmap[cy, cx] = 1.0
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        heatmap /= (heatmap.max() + 1e-8)
    return heatmap

def compute_median_image(frames_dir, frame_files, max_frames=30):
    imgs = []
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files)-1, max_frames, dtype=int)
        frame_files_subset = [frame_files[i] for i in indices]
    else:
        frame_files_subset = frame_files
    for frame_file in frame_files_subset:
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        if img is not None:
            imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError(f"No frames found in {frames_dir} for median computation.")
    median_img = np.median(np.stack(imgs, axis=0), axis=0)
    return median_img

def compute_farneback_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Normalize flow to [-1, 1]
    flow = np.clip(flow / 20.0, -1, 1)
    return flow.astype(np.float32)  # shape: (H, W, 2)

def downsample_flow(flow, target_size=(640, 360)):
    # flow: (H, W, 2), target_size: (W, H)
    H, W = flow.shape[:2]
    new_W, new_H = target_size
    flow_resized = np.zeros((new_H, new_W, 2), dtype=flow.dtype)
    for i in range(2):
        flow_resized[..., i] = cv2.resize(flow[..., i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        # Scale flow values to account for resizing
        flow_resized[..., i] *= (new_W / W if i == 0 else new_H / H)
    return flow_resized

def process_segment(frames_dir, csv_file, out_dir, heatmap_size=64, stack_len=5, median_max_frames=30, flow_size=(640, 360)):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    df = pd.read_csv(csv_file)
    ensure_dir(out_dir)
    # Compute median background
    median_img = compute_median_image(frames_dir, frame_files, max_frames=median_max_frames)
    # Prepare output dirs
    rgb_dir = os.path.join(out_dir, 'frames')
    diff_dir = os.path.join(out_dir, 'diffs')
    flow_dir = os.path.join(out_dir, 'flows')
    heatmap_dir = os.path.join(out_dir, 'heatmaps')
    for d in [rgb_dir, diff_dir, flow_dir, heatmap_dir]:
        ensure_dir(d)
    
    # Precompute all frames, diffs, flows, heatmaps
    imgs = []
    diffs = []
    # Create a zero flow for the first frame - we'll use this placeholder
    zero_flow = np.zeros((median_img.shape[0], median_img.shape[1], 2), dtype=np.float32)
    
    # Process each frame
    for i, frame_file in enumerate(tqdm(frame_files, desc=f"Processing {os.path.basename(frames_dir)}")):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        img_f = img.astype(np.float32) / 255.0
        
        # Save the RGB frame
        cv2.imwrite(os.path.join(rgb_dir, frame_file), img)
        
        # Compute and save median-subtracted diff
        diff = np.abs(img_f - (median_img / 255.0))
        cv2.imwrite(os.path.join(diff_dir, frame_file), (diff * 255).astype(np.uint8))
        
        # Compute and save optical flow
        if i == 0:
            # For the first frame, use a zero flow
            flow = zero_flow
        else:
            flow = compute_farneback_flow(imgs[-1], img)
        
        # Downsample and convert to float16
        flow_ds = downsample_flow(flow, target_size=flow_size).astype(np.float16)
        flow_filename = os.path.splitext(frame_file)[0] + '.npz'
        np.savez_compressed(os.path.join(flow_dir, flow_filename), flow=flow_ds)
        
        # Compute and save heatmap
        row = df.iloc[i]
        x_norm = row['X'] / img.shape[1]
        y_norm = row['Y'] / img.shape[0]
        heatmap = generate_heatmap(x_norm, y_norm, heatmap_size, heatmap_size, sigma=2)
        heatmap_filename = os.path.splitext(frame_file)[0] + '.npz'
        np.savez_compressed(os.path.join(heatmap_dir, heatmap_filename), heatmap=heatmap.astype(np.float32))
        
        # Keep the current frame for next iteration's flow calculation
        imgs.append(img)
        
    # Save stacked sample indices for efficient loading
    stack_indices = []
    for i in range(len(frame_files) - stack_len + 1):
        stack_indices.append(list(range(i, i+stack_len)))
    np.save(os.path.join(out_dir, 'stack_indices.npy'), np.array(stack_indices, dtype=np.int32))
    
    # Save a metadata file with information about the processed data
    with open(os.path.join(out_dir, 'metadata.yaml'), 'w') as f:
        metadata = {
            'num_frames': len(frame_files),
            'heatmap_size': heatmap_size,
            'stack_len': stack_len,
            'flow_size': flow_size,
            'data_types': {
                'frames': 'RGB images',
                'diffs': 'Median-subtracted difference images',
                'flows': 'Farneback optical flow maps (.npz, float16, downsampled)',
                'heatmaps': 'Gaussian heatmaps for position (.npz, float32)',
                'stack_indices': 'Indices for frame stacking'
            }
        }
        yaml.dump(metadata, f)
    
    return len(frame_files)

def verify_processed_segment(out_dir, frames_dir):
    try:
        # Check all expected directories exist
        for subdir in ['frames', 'diffs', 'flows', 'heatmaps']:
            if not os.path.exists(os.path.join(out_dir, subdir)):
                print(f"[VERIFY] Missing directory: {os.path.join(out_dir, subdir)}")
                return False
        
        # Check all expected files exist
        raw_frames = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        processed_frames = [f for f in os.listdir(os.path.join(out_dir, 'frames')) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(raw_frames) != len(processed_frames):
            print(f"[VERIFY] Frame count mismatch: raw={len(raw_frames)}, processed={len(processed_frames)}")
            return False
        
        # Check stack_indices.npy exists
        if not os.path.exists(os.path.join(out_dir, 'stack_indices.npy')):
            print(f"[VERIFY] Missing stack_indices.npy")
            return False
        
        # Check metadata exists
        if not os.path.exists(os.path.join(out_dir, 'metadata.yaml')):
            print(f"[VERIFY] Missing metadata.yaml")
            return False
        
        return True
    except Exception as e:
        print(f"[VERIFY] Error verifying segment: {e}")
        return False

def process_split(split):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]
    matches = [m for m in matches if m not in EXCLUDE_MATCHES]

    for match in tqdm(matches, desc=f'Processing {split}', position=0, leave=True):
        match_path = os.path.join(split_path, match)
        segments_dir = os.path.join(match_path, 'frames')
        if not os.path.exists(segments_dir):
            continue
            
        segments = os.listdir(segments_dir)
        print(f"\n[INFO] Processing match: {match} ({len(segments)} segments)")
        total_frames = 0
        
        for segment in tqdm(segments, desc=f'  Segments in {match}', position=1, leave=True):
            frames_dir = os.path.join(match_path, 'frames', segment)
            csv_file = os.path.join(match_path, 'csv', f'{segment}_ball.csv')
            if not os.path.exists(csv_file):
                print(f"[WARNING] Missing CSV file for segment {segment}, skipping...")
                continue
                
            out_dir = os.path.join(OUTPUT_ROOT, split, match, segment)
            
            # Check if already processed
            if os.path.exists(out_dir):
                if verify_processed_segment(out_dir, frames_dir):
                    print(f"[INFO] Segment {segment} already processed, skipping...")
                    with open(os.path.join(out_dir, 'metadata.yaml'), 'r') as f:
                        metadata = yaml.safe_load(f)
                        total_frames += metadata.get('num_frames', 0)
                    continue
                else:
                    print(f"[WARNING] Segment {segment} verification failed, reprocessing...")
                    shutil.rmtree(out_dir, ignore_errors=True)
            
            try:
                num_frames = process_segment(frames_dir, csv_file, out_dir)
                total_frames += num_frames
            except Exception as e:
                print(f"[ERROR] Failed to process segment {segment}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"[INFO] Completed match: {match} with {total_frames} total frames\n")

def main():
    for split in SPLITS:
        process_split(split)
    print('\n[SUCCESS] Preprocessing complete!')

if __name__ == '__main__':
    main()
