import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time
import math
from albumentations import RandomBrightnessContrast, Blur, GaussNoise, RandomResizedCrop, RandomGamma, RandomShadow, RandomFog, RandomRain, RandomSnow, RandomSunFlare, RandomBrightness, RandomContrast, RandomErasing

class ShuttleTrackDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=5, augment=True, input_size=224, augmentation_config=None):
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

        self.augmentation_config = augmentation_config or {}
        self.transform = self._build_transform()
        # Advanced aug flags
        self.use_mixup = self.augmentation_config.get('mixup', False)
        self.use_cutmix = self.augmentation_config.get('cutmix', False)
        self.use_random_erasing = self.augmentation_config.get('random_erasing', False)
        self.use_color_jitter = self.augmentation_config.get('color_jitter', False)
        self.use_blur = self.augmentation_config.get('blur', False)
        self.use_noise = self.augmentation_config.get('noise', False)

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
        transforms = []
        if self.augment:
            transforms.append(A.HorizontalFlip(p=0.5))
            if self.use_color_jitter:
                transforms.append(A.ColorJitter(p=0.5))
            if self.use_blur:
                transforms.append(A.Blur(p=0.2))
            if self.use_noise:
                transforms.append(A.GaussNoise(p=0.2))
            if self.use_random_erasing:
                transforms.append(A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.3))
        transforms.append(ToTensorV2())
        return A.Compose(transforms, additional_targets={'diff_image': 'image'})

    def __len__(self):
        return len(self.samples)

    def _generate_heatmap(self, x, y, visible, size=56, sigma=2):
        """Generate a 2D Gaussian heatmap centered at (x, y) in [0,1] coords."""
        if not visible:
            return np.zeros((size, size), dtype=np.float32)
        cx = int(x * (size - 1))
        cy = int(y * (size - 1))
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        heatmap = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.astype(np.float32)

    def _apply_mixup(self, frames, diffs, vis, heatmap):
        # Mixup with another random sample in the batch
        lam = np.random.beta(0.4, 0.4)
        idx = random.randint(0, len(self.samples) - 1)
        sample2 = self.__getitem__(idx)
        frames2, diffs2 = sample2['frames'], sample2['diffs']
        vis2, heatmap2 = sample2['visibility'], sample2['heatmap']
        frames = lam * frames + (1 - lam) * frames2
        diffs = lam * diffs + (1 - lam) * diffs2
        vis = lam * vis + (1 - lam) * vis2
        heatmap = lam * heatmap + (1 - lam) * heatmap2
        return frames, diffs, vis, heatmap

    def _apply_cutmix(self, frames, diffs, vis, heatmap):
        # CutMix with another random sample in the batch
        lam = np.random.beta(1.0, 1.0)
        idx = random.randint(0, len(self.samples) - 1)
        sample2 = self.__getitem__(idx)
        frames2, diffs2 = sample2['frames'], sample2['diffs']
        vis2, heatmap2 = sample2['visibility'], sample2['heatmap']
        _, H, W = frames.shape
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cut_w = np.random.randint(W // 4, W // 2)
        cut_h = np.random.randint(H // 4, H // 2)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        frames[:, y1:y2, x1:x2] = frames2[:, y1:y2, x1:x2]
        diffs[:, y1:y2, x1:x2] = diffs2[:, y1:y2, x1:x2]
        heatmap[:, y1:y2, x1:x2] = heatmap2[:, y1:y2, x1:x2]
        # For visibility, use lam
        vis = lam * vis + (1 - lam) * vis2
        return frames, diffs, vis, heatmap

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
        # labels_for_sequence_np: (T, 3) [visibility, x, y]
        vis_list = []
        heatmap_list = []
        for label in labels_for_sequence_np:
            vis, x, y = label
            vis_list.append(vis)
            heatmap = self._generate_heatmap(x, y, vis, size=56, sigma=2)
            heatmap_list.append(torch.tensor(heatmap, dtype=torch.float32))
        vis_tensor = torch.tensor(vis_list, dtype=torch.float32)  # (T,)
        heatmap_tensor = torch.stack(heatmap_list, dim=0)  # (T, H, W)
        label_load_end = time.time()
        t4 = time.time()
        # Apply advanced augmentations (only during training)
        if self.augment:
            if self.use_mixup and random.random() < 0.5:
                frames, diffs, vis_tensor, heatmap_tensor = self._apply_mixup(
                    torch.stack(final_frames_tensors, dim=0),
                    torch.stack(final_diffs_tensors, dim=0),
                    vis_tensor, heatmap_tensor)
            elif self.use_cutmix and random.random() < 0.5:
                frames, diffs, vis_tensor, heatmap_tensor = self._apply_cutmix(
                    torch.stack(final_frames_tensors, dim=0),
                    torch.stack(final_diffs_tensors, dim=0),
                    vis_tensor, heatmap_tensor)
            else:
                frames = torch.stack(final_frames_tensors, dim=0)
                diffs = torch.stack(final_diffs_tensors, dim=0)
        else:
            frames = torch.stack(final_frames_tensors, dim=0)
            diffs = torch.stack(final_diffs_tensors, dim=0)
        return {
            'frames': frames,
            'diffs': diffs,
            'visibility': vis_tensor,
            'heatmap': heatmap_tensor
        } 
