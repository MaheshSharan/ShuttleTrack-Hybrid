import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import ShuttleTrackDataset

def visualize_sequence_with_labels(frames, diffs, labels):
    """Visualize a sequence of frames and diffs with labels"""
    seq_len = frames.shape[0]
    fig, axs = plt.subplots(2, seq_len, figsize=(seq_len * 4, 8))
    
    # Helper function to convert tensor to numpy for visualization
    def tensor_to_np_img(tensor):
        return tensor.permute(1, 2, 0).numpy()
    
    for t in range(seq_len):
        # Get image dimensions for plotting coordinates
        h, w = frames[t].shape[1], frames[t].shape[2]
        
        # Plot frame
        frame_np = tensor_to_np_img(frames[t])
        axs[0, t].imshow(frame_np)
        axs[0, t].set_title(f'Frame {t}')
        
        # Plot diff
        diff_np = tensor_to_np_img(diffs[t])
        axs[1, t].imshow(diff_np)
        axs[1, t].set_title(f'Diff {t}')
        
        # Plot shuttlecock position if visible
        if labels[t, 0] > 0.5:
            x, y = labels[t, 1] * w, labels[t, 2] * h
            axs[0, t].scatter([x], [y], c='lime', s=100, marker='o')
            axs[0, t].text(10, 30, f'Visible: Yes', color='lime', fontsize=12)
            axs[1, t].scatter([x], [y], c='lime', s=100, marker='o')
        else:
            axs[0, t].text(10, 30, f'Visible: No', color='red', fontsize=12)
        
        # Turn off axis labels
        axs[0, t].set_axis_off()
        axs[1, t].set_axis_off()
    
    plt.tight_layout()
    plt.show()

def analyze_sample_statistics(sample):
    """Analyze and print statistics about a sample"""
    frames = sample['frames']  # (T, C, H, W)
    diffs = sample['diffs']    # (T, C, H, W)
    labels = sample['labels']  # (T, 3)
    
    print("\n--- Sample Statistics ---")
    print(f"Frames shape: {frames.shape}")
    print(f"Diffs shape: {diffs.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Frame statistics
    print("\nFrame pixel statistics:")
    print(f"  Mean: {frames.mean().item():.4f}")
    print(f"  Std: {frames.std().item():.4f}")
    print(f"  Min: {frames.min().item():.4f}")
    print(f"  Max: {frames.max().item():.4f}")
    
    # Diff statistics
    print("\nDiff pixel statistics:")
    print(f"  Mean: {diffs.mean().item():.4f}")
    print(f"  Std: {diffs.std().item():.4f}")
    print(f"  Min: {diffs.min().item():.4f}")
    print(f"  Max: {diffs.max().item():.4f}")
    
    # Count visible frames
    visible_count = (labels[:, 0] > 0.5).sum().item()
    print(f"\nVisible frames: {visible_count}/{labels.shape[0]} ({visible_count/labels.shape[0]*100:.1f}%)")

if __name__ == '__main__':
    # Create dataset for both train and valid splits
    train_dataset = ShuttleTrackDataset(root_dir='processed_data', split='Train', sequence_length=5, augment=False)
    valid_dataset = ShuttleTrackDataset(root_dir='processed_data', split='valid', sequence_length=5, augment=False)
    
    print(f'Total train samples: {len(train_dataset)}')
    print(f'Total validation samples: {len(valid_dataset)}')
    
    # Get one sample from train and one from valid
    train_sample = train_dataset[0]
    valid_sample = valid_dataset[0] if len(valid_dataset) > 0 else None
    
    # Analyze train sample
    print("\n=== Training Sample ===")
    analyze_sample_statistics(train_sample)
    visualize_sequence_with_labels(
        train_sample['frames'], 
        train_sample['diffs'], 
        train_sample['labels']
    )
    
    # Analyze valid sample if available
    if valid_sample is not None:
        print("\n=== Validation Sample ===")
        analyze_sample_statistics(valid_sample)
        visualize_sequence_with_labels(
            valid_sample['frames'], 
            valid_sample['diffs'], 
            valid_sample['labels']
        )
