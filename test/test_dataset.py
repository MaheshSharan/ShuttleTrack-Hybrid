import sys
import os
import torch
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import ShuttleTrackDataset

if __name__ == '__main__':
    dataset = ShuttleTrackDataset(root_dir='processed_data', split='Train', sequence_length=5, augment=False)
    print(f'Total samples: {len(dataset)}')
    sample = dataset[0]
    print('frames:', sample['frames'].shape)
    print('diffs:', sample['diffs'].shape)
    print('labels:', sample['labels'].shape)
    # Visualize the first frame and diff
    frame = sample['frames'][0].permute(1, 2, 0).numpy()
    diff = sample['diffs'][0].numpy()  # Already (H, W, C)
    print('diff image shape:', diff.shape)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(frame)
    axs[0].set_title('Frame 0')
    axs[1].imshow(diff)
    axs[1].set_title('Diff 0')
    plt.show()
