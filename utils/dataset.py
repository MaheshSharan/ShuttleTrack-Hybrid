import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time

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
        t0 = time.time()
        sample = self.samples[idx]
        t1 = time.time()
        frames_resized_list = []
        diffs_resized_list = []
        image_load_time = 0
        resize_time = 0
        for frame_filename in sample['frame_files']:
            t_img_start = time.time()
            frame_path = os.path.join(sample['frames_dir'], frame_filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                raise IOError(f"Could not read frame image: {frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_img_end = time.time()
            image_load_time += t_img_end - t_img_start
            t_resize_start = time.time()
            frame_resized = cv2.resize(frame, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            frames_resized_list.append(frame_resized)
            t_resize_end = time.time()
            resize_time += t_resize_end - t_resize_start

            diff_path = os.path.join(sample['diffs_dir'], frame_filename)
            diff = cv2.imread(diff_path)
            if diff is None:
                raise IOError(f"Could not read diff image: {diff_path}")
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            t_img_end2 = time.time()
            image_load_time += t_img_end2 - t_resize_end
            t_resize_start2 = time.time()
            diff_resized = cv2.resize(diff, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            diffs_resized_list.append(diff_resized)
            t_resize_end2 = time.time()
            resize_time += t_resize_end2 - t_resize_start2
        t2 = time.time()
        aug_time = 0
        if self.augment and random.random() < 0.2:
            drop_idx = random.randint(0, self.sequence_length - 1)
            frames_resized_list[drop_idx] = np.zeros_like(frames_resized_list[drop_idx])
            diffs_resized_list[drop_idx] = np.zeros_like(diffs_resized_list[drop_idx])
        final_frames_tensors = []
        final_diffs_tensors = []
        for f_resized, d_resized in zip(frames_resized_list, diffs_resized_list):
            t_aug_start = time.time()
            augmented = self.transform(image=f_resized, diff_image=d_resized)
            t_aug_end = time.time()
            aug_time += t_aug_end - t_aug_start
            final_frames_tensors.append(augmented['image'])
            final_diffs_tensors.append(augmented['diff_image'])
        t3 = time.time()
        label_load_start = time.time()
        all_labels_np = np.load(sample['labels_path'])
        labels_for_sequence_np = all_labels_np[sample['label_indices']]
        labels_tensor = torch.tensor(labels_for_sequence_np, dtype=torch.float32)
        label_load_end = time.time()
        t4 = time.time()
        return {
            'frames': torch.stack(final_frames_tensors, dim=0),
            'diffs': torch.stack(final_diffs_tensors, dim=0),
            'labels': labels_tensor
        } 
