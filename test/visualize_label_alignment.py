import os
import torch
from utils.dataset import ShuttleTrackDataset
import cv2
import numpy as np
from tqdm import tqdm

# Settings
BATCH_SIZE = 4
NUM_BATCHES = 3  # Number of batches to visualize
SPLIT = 'Train'  # or 'valid'
SAVE_DIR = 'results/label_alignment_vis'
INPUT_SIZE = 224  # or tuple if needed

os.makedirs(SAVE_DIR, exist_ok=True)

def denormalize_coords(x, y, img_w, img_h):
    return int(x * img_w), int(y * img_h)

def main():
    dataset = ShuttleTrackDataset(
        root_dir='processed_data',
        split=SPLIT,
        sequence_length=5,
        augment=False,
        input_size=INPUT_SIZE
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    for batch_idx, batch in enumerate(loader):
        frames = batch['frames']  # (B, T, C, H, W)
        labels = batch['labels']  # (B, T, 3)
        B, T, C, H, W = frames.shape
        for b in range(B):
            for t in range(T):
                img = frames[b, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                img = (img * 255).astype(np.uint8)
                vis, x_norm, y_norm = labels[b, t].cpu().numpy()
                x_px, y_px = denormalize_coords(x_norm, y_norm, W, H)
                # Draw label
                color = (0, 255, 0) if vis > 0.5 else (0, 0, 255)
                img = cv2.circle(img, (x_px, y_px), 5, color, -1)
                out_path = os.path.join(SAVE_DIR, f'b{batch_idx}_s{b}_t{t}.jpg')
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if batch_idx + 1 >= NUM_BATCHES:
            break
    print(f"Saved label alignment visualizations to {SAVE_DIR}")

if __name__ == '__main__':
    main() 