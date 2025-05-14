import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ShuttleTrackDataset(Dataset):
    def __init__(self, root_dir, split='Train', sequence_length=5, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.augment = augment
        self.samples = self._gather_samples()
        self.transform = self._build_transform()

    def _gather_samples(self):
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        for match in os.listdir(split_dir):
            match_dir = os.path.join(split_dir, match)
            for segment in os.listdir(match_dir):
                seg_dir = os.path.join(match_dir, segment)
                frames_dir = os.path.join(seg_dir, 'frames')
                diffs_dir = os.path.join(seg_dir, 'diffs')
                labels_path = os.path.join(seg_dir, 'labels.npy')
                frame_files = sorted(os.listdir(frames_dir))
                labels = np.load(labels_path)
                # Only use sequences where all frames are available
                for i in range(len(frame_files) - self.sequence_length + 1):
                    samples.append({
                        'frames_dir': frames_dir,
                        'diffs_dir': diffs_dir,
                        'labels_path': labels_path,
                        'start_idx': i,
                        'frame_files': frame_files[i:i+self.sequence_length],
                        'label_indices': list(range(i, i+self.sequence_length)),
                    })
        return samples

    def _build_transform(self):
        if self.augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(224, 224, p=0.8),
                A.ColorJitter(p=0.5),
                A.GaussNoise(p=0.2),
                A.Resize(224, 224),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.CenterCrop(224, 224),
                A.Resize(224, 224),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        diffs = []
        for frame_file in sample['frame_files']:
            frame = np.uint8(np.round(np.load(os.path.join(sample['frames_dir'], frame_file.replace('.jpg', '.npy'))) * 255)) if frame_file.endswith('.npy') else cv2.imread(os.path.join(sample['frames_dir'], frame_file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            diff = cv2.imread(os.path.join(sample['diffs_dir'], frame_file))
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            diffs.append(diff)
        # Stack frames and diffs along channel axis
        frames = np.stack(frames, axis=0)  # (T, H, W, C)
        diffs = np.stack(diffs, axis=0)
        # Temporal dropout (randomly drop a frame in the sequence)
        if self.augment and random.random() < 0.2:
            drop_idx = random.randint(0, self.sequence_length-1)
            frames[drop_idx] = 0
            diffs[drop_idx] = 0
        # Apply spatial augmentations to each frame
        frames_aug = []
        diffs_aug = []
        for f, d in zip(frames, diffs):
            augmented = self.transform(image=f, mask=d)
            frames_aug.append(augmented['image'])
            mask = augmented['mask']
            if isinstance(mask, np.ndarray):
                if mask.ndim == 3 and mask.shape[2] == 3:
                    mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)))
                mask = mask.float() / 255.0
            elif torch.is_tensor(mask):
                if mask.ndim == 3 and mask.shape[2] == 3:
                    mask = mask.permute(2, 0, 1)
                mask = mask.float() / 255.0
            diffs_aug.append(mask)
        frames_aug = torch.stack(frames_aug, dim=0)  # (T, C, H, W)
        diffs_aug = torch.stack(diffs_aug, dim=0)
        # Labels
        labels = np.load(sample['labels_path'])[sample['label_indices']]
        labels = torch.tensor(labels, dtype=torch.float32)
        return {
            'frames': frames_aug,  # (T, C, H, W)
            'diffs': diffs_aug,    # (T, C, H, W)
            'labels': labels       # (T, 3)
        } 