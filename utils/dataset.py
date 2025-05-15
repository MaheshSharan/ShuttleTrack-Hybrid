import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ShuttleTrackDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=5, augment=True, input_size=224):
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.augment = augment
        self.samples = self._gather_samples()
        
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size) # (height, width)
        else:
            self.input_size = tuple(input_size) # (height, width)
        
        # cv2.resize expects dsize as (width, height)
        self.cv_resize_dsize = (self.input_size[1], self.input_size[0])

        self.transform = self._build_transform()

    def _gather_samples(self):
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        for match in os.listdir(split_dir):
            match_dir = os.path.join(split_dir, match)
            if not os.path.isdir(match_dir):
                continue
            for segment in os.listdir(match_dir):
                seg_dir = os.path.join(match_dir, segment)
                if not os.path.isdir(seg_dir):
                    continue

                frames_dir = os.path.join(seg_dir, 'frames')
                diffs_dir = os.path.join(seg_dir, 'diffs')
                labels_path = os.path.join(seg_dir, 'labels.npy')

                if not (os.path.isdir(frames_dir) and os.path.isdir(diffs_dir) and os.path.isfile(labels_path)):
                    continue
                    
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if not frame_files:
                    continue

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
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.GaussNoise(p=0.2),
                ToTensorV2() # Converts HWC numpy to CHW tensor, divides by 255
            ], additional_targets={'diff_image': 'image'})
        else:
            transform = A.Compose([
                ToTensorV2()
            ], additional_targets={'diff_image': 'image'})
        return transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames_resized_list = []
        diffs_resized_list = []

        for frame_filename in sample['frame_files']:
            frame_path = os.path.join(sample['frames_dir'], frame_filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                raise IOError(f"Could not read frame image: {frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            frames_resized_list.append(frame_resized)

            diff_path = os.path.join(sample['diffs_dir'], frame_filename) # Assuming diffs have same filenames
            diff = cv2.imread(diff_path)
            if diff is None:
                raise IOError(f"Could not read diff image: {diff_path}")
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            diff_resized = cv2.resize(diff, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            diffs_resized_list.append(diff_resized)
        
        if self.augment and random.random() < 0.2: # Temporal dropout
            drop_idx = random.randint(0, self.sequence_length - 1)
            frames_resized_list[drop_idx] = np.zeros_like(frames_resized_list[drop_idx])
            diffs_resized_list[drop_idx] = np.zeros_like(diffs_resized_list[drop_idx])
            
        final_frames_tensors = []
        final_diffs_tensors = []

        for f_resized, d_resized in zip(frames_resized_list, diffs_resized_list):
            augmented = self.transform(image=f_resized, diff_image=d_resized)
            final_frames_tensors.append(augmented['image'])
            final_diffs_tensors.append(augmented['diff_image'])

        frames_stacked = torch.stack(final_frames_tensors, dim=0)
        diffs_stacked = torch.stack(final_diffs_tensors, dim=0)
        
        all_labels_np = np.load(sample['labels_path'])
        labels_for_sequence_np = all_labels_np[sample['label_indices']]
        labels_tensor = torch.tensor(labels_for_sequence_np, dtype=torch.float32)

        return {
            'frames': frames_stacked,
            'diffs': diffs_stacked,
            'labels': labels_tensor
        } 
