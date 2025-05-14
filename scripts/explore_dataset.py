import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

DATASET_ROOT = 'ShuttleCockFrameDataset'
SPLITS = ['Train', 'valid']


def print_dataset_structure():
    print('Dataset Structure:')
    for split in SPLITS:
        split_path = os.path.join(DATASET_ROOT, split)
        if not os.path.exists(split_path):
            continue
        print(f'  {split}/')
        matches = [d for d in os.listdir(split_path) if d.startswith('match')]
        print(f'    Matches: {len(matches)}')
        for match in matches:
            match_path = os.path.join(split_path, match)
            segments = os.listdir(os.path.join(match_path, 'frames'))
            print(f'      {match}/: {len(segments)} segments')


def summarize_annotations(split='Train'):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]
    total_frames = 0
    visible_frames = 0
    for match in matches:
        csv_dir = os.path.join(split_path, match, 'csv')
        for csv_file in os.listdir(csv_dir):
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            total_frames += len(df)
            visible_frames += df['Visibility'].sum()
    print(f'[{split}] Total frames: {total_frames}, Visible shuttlecock: {visible_frames} ({visible_frames/total_frames:.2%})')


def visualize_sample(split='Train', match_idx=0, segment_idx=0, num_frames=5):
    split_path = os.path.join(DATASET_ROOT, split)
    matches = [d for d in os.listdir(split_path) if d.startswith('match')]
    match = matches[match_idx]
    match_path = os.path.join(split_path, match)
    segments = os.listdir(os.path.join(match_path, 'frames'))
    segment = segments[segment_idx]
    frames_dir = os.path.join(match_path, 'frames', segment)
    csv_file = os.path.join(match_path, 'csv', f'{segment}_ball.csv')
    df = pd.read_csv(csv_file)
    frame_files = sorted(os.listdir(frames_dir))[:num_frames]
    plt.figure(figsize=(15, 3))
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row = df.iloc[i]
        plt.subplot(1, num_frames, i+1)
        plt.imshow(img)
        if row['Visibility'] == 1:
            plt.scatter([row['X']], [row['Y']], c='red', s=40, label='Shuttlecock')
        plt.title(f'Frame {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    print_dataset_structure()
    for split in SPLITS:
        summarize_annotations(split)
    print('\nVisualizing sample frames from Train set:')
    visualize_sample('Train', match_idx=0, segment_idx=0, num_frames=5)

if __name__ == '__main__':
    main() 