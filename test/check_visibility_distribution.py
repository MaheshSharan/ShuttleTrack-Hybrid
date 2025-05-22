# scripts/check_visibility_distribution.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.dataset import ShuttleTrackDataset

def main():
    dataset = ShuttleTrackDataset(
        root_dir='processed_data',  # or your processed_dataset_path
        split='Train',  # or 'valid'
        sequence_length=5,
        augment=False,
        use_optical_flow=True,
        use_heatmaps=True
    )

    num_batches = 10
    batch_size = 4
    visible_count = 0
    invisible_count = 0
    total = 0

    for i in range(num_batches):
        sample = dataset[i]
        vis = sample['labels'][:, 0]
        visible_count += (vis > 0.5).sum().item()
        invisible_count += (vis <= 0.5).sum().item()
        total += len(vis)
        print(f"Batch {i+1}: Visible: {(vis > 0.5).sum().item()}, Invisible: {(vis <= 0.5).sum().item()}, Total: {len(vis)}")

    print(f"\nTotal frames checked: {total}")
    print(f"Visible frames: {visible_count} ({visible_count/total:.1%})")
    print(f"Invisible frames: {invisible_count} ({invisible_count/total:.1%})")

if __name__ == '__main__':
    main()