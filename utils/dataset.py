import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time
from scipy.ndimage import gaussian_filter
import yaml

class ShuttleTrackDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=5, augment=True, input_size=224, 
                 use_optical_flow=True, use_heatmaps=True, difficulty_level=None, 
                 mixup_prob=0.3, blur_prob=0.3, cutout_prob=0.2,
                 curriculum_epoch=None, max_curriculum_epochs=20,
                 exclude_folders=None):
        """
        Enhanced ShuttleTrack Dataset with advanced augmentation techniques.
        
        Args:
            root_dir: Root directory for processed data
            split: Data split ('Train', 'valid', 'test')
            sequence_length: Number of frames in each sequence
            augment: Whether to apply data augmentation
            input_size: Image dimensions (height, width) or single value for both
            use_optical_flow: Whether to include optical flow data
            use_heatmaps: Whether to include heatmap data
            difficulty_level: For curriculum learning (None, 'easy', 'medium', 'hard')
            mixup_prob: Probability of applying mixup augmentation
            blur_prob: Probability of applying motion blur augmentation
            cutout_prob: Probability of applying random cutout augmentation
            curriculum_epoch: Current epoch for curriculum learning
            max_curriculum_epochs: Number of epochs for curriculum progression
            exclude_folders: List of folders to exclude from training
        """
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.augment = augment
        self.use_optical_flow = use_optical_flow
        self.use_heatmaps = use_heatmaps
        self.mixup_prob = mixup_prob if augment else 0.0
        self.blur_prob = blur_prob if augment else 0.0
        self.cutout_prob = cutout_prob if augment else 0.0
        self.curriculum_epoch = curriculum_epoch
        self.max_curriculum_epochs = max_curriculum_epochs
        self.difficulty_level = difficulty_level
        self.exclude_folders = exclude_folders if exclude_folders is not None else []
        
        # Get samples based on difficulty if in curriculum learning mode
        self.samples = self._gather_samples()
        
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size) # (height, width)
        else:
            self.input_size = tuple(input_size) # (height, width)
        
        # cv2.resize expects dsize as (width, height)
        self.cv_resize_dsize = (self.input_size[1], self.input_size[0])

        self.transform = self._build_transform()
        
        # Hard sample indices (initialize empty, will be updated during training)
        self.hard_sample_indices = []
        self.hard_sample_losses = []
        self.use_hard_mining = False
        
        # Cache for mixup samples
        self._mixup_cache = []
        self._update_mixup_cache()

    def _update_mixup_cache(self):
        """Prepare a cache of samples for mixup augmentation."""
        if not self.augment or self.mixup_prob <= 0:
            return
            
        # Randomly select 100 samples for the mixup cache
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        self._mixup_cache = [self.samples[i] for i in indices[:100]]
    
    def _estimate_sample_difficulty(self, sample):
        """Estimate difficulty level of a sample based on heuristics.
        
        Returns:
            str: 'easy', 'medium', or 'hard'
        """
        if self.split != 'Train' or not os.path.exists(sample['metadata_path']):
            return 'medium'  # Default for validation/test
        
        # Load metadata to analyze
        try:
            with open(sample['metadata_path'], 'r') as f:
                metadata = yaml.safe_load(f)
            
            # Extract metrics to determine difficulty
            if 'visibility_rate' in metadata:
                vis_percentage = metadata['visibility_rate']
            else:
                vis_percentage = 0.8  # Default if not available
                
            # Use other metadata if available, such as:
            # - shuttlecock speed
            # - occlusions
            # - motion blur
            
            # Simple heuristic based on visibility
            if vis_percentage > 0.9:
                return 'easy'
            elif vis_percentage < 0.7:
                return 'hard'
            else:
                return 'medium'
        except:
            return 'medium'  # Default if analysis fails

    def _gather_samples(self):
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        for match in os.listdir(split_dir):
            if match in self.exclude_folders:
                continue
            match_dir = os.path.join(split_dir, match)
            if not os.path.isdir(match_dir):
                continue
            for segment in os.listdir(match_dir):
                seg_dir = os.path.join(match_dir, segment)
                if not os.path.isdir(seg_dir):
                    continue

                frames_dir = os.path.join(seg_dir, 'frames')
                diffs_dir = os.path.join(seg_dir, 'diffs')
                flows_dir = os.path.join(seg_dir, 'flows') if self.use_optical_flow else None
                heatmaps_dir = os.path.join(seg_dir, 'heatmaps') if self.use_heatmaps else None
                metadata_path = os.path.join(seg_dir, 'metadata.yaml')
                stack_indices_path = os.path.join(seg_dir, 'stack_indices.npy')

                # Check if all required directories exist
                required_dirs = [frames_dir, diffs_dir]
                if self.use_optical_flow:
                    required_dirs.append(flows_dir)
                
                # Check if all required directories and files exist
                if not all(os.path.isdir(d) for d in required_dirs) or not os.path.isfile(stack_indices_path):
                    continue
                    
                # Get frame files
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if not frame_files:
                    continue

                # Load stack indices to get mapping between frames and labels
                if os.path.exists(stack_indices_path):
                    stack_indices = np.load(stack_indices_path)
                else:
                    # If no stack indices, use frame indices directly
                    stack_indices = np.arange(len(frame_files))

                # Process sequences
                for i in range(len(frame_files) - self.sequence_length + 1):
                    seq_frame_files = frame_files[i:i+self.sequence_length]
                    seq_indices = stack_indices[i:i+self.sequence_length] if len(stack_indices) >= i+self.sequence_length else np.arange(i, i+self.sequence_length)
                    
                    sample = {
                        'frames_dir': frames_dir,
                        'diffs_dir': diffs_dir,
                        'flows_dir': flows_dir,
                        'heatmaps_dir': heatmaps_dir,
                        'metadata_path': metadata_path,
                        'stack_indices_path': stack_indices_path,
                        'start_idx': i,
                        'frame_files': seq_frame_files,
                        'stack_indices': seq_indices,
                        'match': match,
                        'segment': segment
                    }
                    
                    # Estimate difficulty for curriculum learning
                    if self.difficulty_level is not None:
                        difficulty = self._estimate_sample_difficulty(sample)
                        sample['difficulty'] = difficulty
                        
                        # Only add samples that match the current difficulty level
                        if difficulty == self.difficulty_level:
                            samples.append(sample)
                    else:
                        samples.append(sample)
        
        return samples
    
    def update_hard_samples(self, indices, losses):
        """Update hard sample indices based on losses.
        
        Args:
            indices: List of sample indices
            losses: Corresponding loss values
        """
        # Combine with existing hard samples
        combined_indices = self.hard_sample_indices + indices
        combined_losses = self.hard_sample_losses + losses
        
        # Sort by loss in descending order
        sorted_pairs = sorted(zip(combined_indices, combined_losses), 
                             key=lambda x: x[1], reverse=True)
        
        # Keep top 20% of samples
        keep_count = min(len(sorted_pairs), int(0.2 * len(self.samples)))
        
        self.hard_sample_indices = [idx for idx, _ in sorted_pairs[:keep_count]]
        self.hard_sample_losses = [loss for _, loss in sorted_pairs[:keep_count]]
        
        # Enable hard mining after we have enough samples
        self.use_hard_mining = len(self.hard_sample_indices) > 10
    
    def set_curriculum_epoch(self, epoch):
        """Update curriculum learning parameters based on current epoch."""
        self.curriculum_epoch = epoch
        
        # Update difficulty level based on curriculum progress
        if self.max_curriculum_epochs > 0:
            progress = min(1.0, epoch / self.max_curriculum_epochs)
            
            if progress < 0.3:
                self.difficulty_level = 'easy'
            elif progress < 0.7:
                self.difficulty_level = 'medium'
            else:
                self.difficulty_level = None  # Use all samples
                
            # Re-gather samples if needed
            if self.difficulty_level is not None:
                self.samples = self._gather_samples()

    def _build_transform(self):
        """Build augmentation pipeline."""
        if not self.augment:
            transform = A.Compose([
                ToTensorV2()
            ], additional_targets={'diff_image': 'image'})
            return transform
            
        # Advanced augmentation pipeline
        transform = A.Compose([
            # Spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                height=self.input_size[0], 
                width=self.input_size[1],
                scale=(0.9, 1.1),
                ratio=(0.9, 1.1),
                p=0.3
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=5, 
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5)
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=(3, 7))
            ], p=0.3),
            
            # Cutout/dropout
            A.CoarseDropout(
                max_holes=5, 
                max_height=int(self.input_size[0]*0.1),
                max_width=int(self.input_size[1]*0.1),
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),
            
            # Normalization and conversion to tensor
            ToTensorV2()
        ], additional_targets={'diff_image': 'image'})
        
        return transform

    def apply_motion_blur(self, image, angle=None, direction=None):
        """Apply realistic motion blur to an image.
        
        Args:
            image: Input image (H, W, C)
            angle: Motion blur angle (degrees)
            direction: Direction of motion (None for random)
        
        Returns:
            Blurred image
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Random angle if not specified
        if angle is None:
            angle = random.uniform(0, 180)
            
        # Random direction if not specified
        if direction is None:
            direction = random.choice([-1, 1])
            
        # Calculate kernel size based on image size
        kernel_size = random.randint(7, 15)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Plot line on kernel
        for i in range(kernel_size):
            x = i - center
            y = direction * int(np.round(np.tan(angle_rad) * x))
            if abs(y) < center:
                kernel[center + y, i] = 1
                
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Apply to each channel
        blurred = np.zeros_like(img_float)
        for c in range(img_float.shape[2]):
            blurred[..., c] = cv2.filter2D(img_float[..., c], -1, kernel)
            
        # Convert back to original dtype
        blurred = (np.clip(blurred, 0, 1) * 255).astype(image.dtype)
        return blurred

    def mixup_sample(self, sample1, sample2, alpha=0.2):
        """Apply mixup augmentation between two samples.
        
        Args:
            sample1: First sample dict
            sample2: Second sample dict
            alpha: Mixup interpolation factor
        
        Returns:
            Mixed sample
        """
        # Interpolate between samples
        lam = np.random.beta(alpha, alpha)
        lam = max(0.2, min(0.8, lam))  # Keep reasonable range
        
        # Mix images
        mixed_frames = []
        mixed_diffs = []
        
        for i in range(self.sequence_length):
            # Get paths
            frame1_path = os.path.join(sample1['frames_dir'], sample1['frame_files'][i])
            frame2_path = os.path.join(sample2['frames_dir'], sample2['frame_files'][i])
            diff1_path = os.path.join(sample1['diffs_dir'], sample1['frame_files'][i])
            diff2_path = os.path.join(sample2['diffs_dir'], sample2['frame_files'][i])
            
            # Load images
            frame1 = cv2.imread(frame1_path)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            
            frame2 = cv2.imread(frame2_path)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2 = cv2.resize(frame2, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            
            diff1 = cv2.imread(diff1_path)
            diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2RGB)
            diff1 = cv2.resize(diff1, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            
            diff2 = cv2.imread(diff2_path)
            diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2RGB)
            diff2 = cv2.resize(diff2, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
            
            # Mix frames and diffs
            mixed_frame = cv2.addWeighted(frame1, lam, frame2, 1-lam, 0)
            mixed_diff = cv2.addWeighted(diff1, lam, diff2, 1-lam, 0)
            
            mixed_frames.append(mixed_frame)
            mixed_diffs.append(mixed_diff)
        
        # Extract labels from heatmaps for both samples
        heatmaps1 = self._load_heatmaps(sample1)
        heatmaps2 = self._load_heatmaps(sample2)
        
        # Mix heatmaps
        mixed_heatmaps = []
        for i in range(self.sequence_length):
            mixed_heatmap = heatmaps1[i] * lam + heatmaps2[i] * (1-lam)
            mixed_heatmaps.append(mixed_heatmap)
        
        # Create mixed labels from heatmaps
        mixed_labels = self._create_labels_from_heatmaps(mixed_heatmaps)
        
        return mixed_frames, mixed_diffs, mixed_labels

    def _load_heatmaps(self, sample):
        """Load heatmaps for a sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            List of heatmaps
        """
        heatmaps = []
        
        if sample['heatmaps_dir'] and os.path.isdir(sample['heatmaps_dir']):
            for frame_file in sample['frame_files']:
                heatmap_file = os.path.splitext(frame_file)[0] + '.npz'
                heatmap_path = os.path.join(sample['heatmaps_dir'], heatmap_file)
                
                try:
                    if os.path.exists(heatmap_path):
                        heatmap = np.load(heatmap_path)['heatmap']
                    else:
                        # Default empty heatmap
                        heatmap = np.zeros((64, 64), dtype=np.float32)
                except:
                    # Default empty heatmap
                    heatmap = np.zeros((64, 64), dtype=np.float32)
                    
                heatmaps.append(heatmap)
        else:
            # If no heatmaps directory, create empty heatmaps
            for _ in range(len(sample['frame_files'])):
                heatmaps.append(np.zeros((64, 64), dtype=np.float32))
        
        return heatmaps
        
    def _create_labels_from_heatmaps(self, heatmaps):
        """Create labels from heatmaps.
        
        Args:
            heatmaps: List of heatmaps
            
        Returns:
            Labels array of shape (T, 3) with [visibility, x, y]
        """
        T = len(heatmaps)
        labels = np.zeros((T, 3), dtype=np.float32)
        
        for t, heatmap in enumerate(heatmaps):
            # Check if the heatmap has any signal
            if np.max(heatmap) > 0:
                # Get visibility based on peak value
                labels[t, 0] = min(1.0, np.max(heatmap) * 2.0)  # Scale up, but cap at 1.0
                
                # Get peak position
                peak_idx = np.argmax(heatmap)
                cy, cx = np.unravel_index(peak_idx, heatmap.shape)
                
                # Convert to normalized coordinates
                labels[t, 1] = cx / heatmap.shape[1]  # x
                labels[t, 2] = cy / heatmap.shape[0]  # y
            else:
                # No shuttlecock detected
                labels[t, 0] = 0.0
                labels[t, 1] = 0.5  # Default to center
                labels[t, 2] = 0.5  # Default to center
                
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample with advanced augmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Sample data
        """
        # Hard example mining: randomly switch to a hard sample
        if self.use_hard_mining and self.hard_sample_indices and random.random() < 0.4:
            idx = random.choice(self.hard_sample_indices)
            
        sample = self.samples[idx]
        frames_resized_list = []
        diffs_resized_list = []
        flows_list = []
        heatmaps_list = []
        
        # Apply mixup augmentation
        mixed_sample = False
        mixed_frames = None
        mixed_diffs = None
        mixed_labels = None
        
        if self.augment and random.random() < self.mixup_prob and self._mixup_cache:
            # Get a random sample from the cache
            mix_sample = random.choice(self._mixup_cache)
            
            # Apply mixup
            mixed_frames, mixed_diffs, mixed_labels = self.mixup_sample(sample, mix_sample)
            mixed_sample = True
        
        # Load frames and apply augmentations
        if mixed_sample:
            # Use the pre-mixed frames
            frames_resized_list = mixed_frames
            diffs_resized_list = mixed_diffs
        else:
            # Process each frame normally
            for frame_filename in sample['frame_files']:
                # Load and process RGB frame
                frame_path = os.path.join(sample['frames_dir'], frame_filename)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise IOError(f"Could not read frame image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame_resized = cv2.resize(frame, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
                
                # Load and process difference image
                diff_path = os.path.join(sample['diffs_dir'], frame_filename)
                diff = cv2.imread(diff_path)
                if diff is None:
                    raise IOError(f"Could not read diff image: {diff_path}")
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                
                diff_resized = cv2.resize(diff, self.cv_resize_dsize, interpolation=cv2.INTER_LINEAR)
                
                # Apply synthetic motion blur if needed
                if self.augment and random.random() < self.blur_prob:
                    # Generate consistent blur parameters for both frame and diff
                    blur_angle = random.uniform(0, 180)
                    blur_direction = random.choice([-1, 1])
                    
                    # Apply blur
                    frame_resized = self.apply_motion_blur(
                        frame_resized, angle=blur_angle, direction=blur_direction
                    )
                    diff_resized = self.apply_motion_blur(
                        diff_resized, angle=blur_angle, direction=blur_direction
                    )
                
                frames_resized_list.append(frame_resized)
                diffs_resized_list.append(diff_resized)
                
                # Load optical flow if enabled
                if self.use_optical_flow and sample['flows_dir'] and os.path.isdir(sample['flows_dir']):
                    flow_filename = os.path.splitext(frame_filename)[0] + '.npz'
                    flow_path = os.path.join(sample['flows_dir'], flow_filename)
                    try:
                        if os.path.exists(flow_path):
                            flow = np.load(flow_path)['flow']
                            
                            # Flow is stored as [H, W, 2] with downsampled size (e.g., 640x360)
                            # Resize to match the input size
                            flow_resized = np.zeros((self.input_size[0], self.input_size[1], 2), dtype=np.float32)
                            for i in range(2):
                                flow_resized[..., i] = cv2.resize(
                                    flow[..., i], 
                                    self.cv_resize_dsize, 
                                    interpolation=cv2.INTER_LINEAR
                                )
                                
                                # Scale flow values to account for resizing
                                flow_resized[..., i] *= (self.input_size[1] / flow.shape[1] if i == 0 else 
                                                     self.input_size[0] / flow.shape[0])
                            
                            flows_list.append(flow_resized)
                        else:
                            # Zero flow if file doesn't exist
                            flows_list.append(np.zeros((self.input_size[0], self.input_size[1], 2), dtype=np.float32))
                    except Exception as e:
                        # If flow loading fails, use zero flow
                        print(f"Warning: Could not load flow for {flow_path}: {e}")
                        flows_list.append(np.zeros((self.input_size[0], self.input_size[1], 2), dtype=np.float32))
                elif self.use_optical_flow:
                    # Add zero flow if flows_dir doesn't exist but optical flow is enabled
                    flows_list.append(np.zeros((self.input_size[0], self.input_size[1], 2), dtype=np.float32))
                
                # Load heatmaps if enabled
                if self.use_heatmaps and sample['heatmaps_dir'] and os.path.isdir(sample['heatmaps_dir']):
                    heatmap_filename = os.path.splitext(frame_filename)[0] + '.npz'
                    heatmap_path = os.path.join(sample['heatmaps_dir'], heatmap_filename)
                    try:
                        if os.path.exists(heatmap_path):
                            heatmap = np.load(heatmap_path)['heatmap']
                            heatmaps_list.append(heatmap)
                        else:
                            # Zero heatmap if file doesn't exist
                            heatmaps_list.append(np.zeros((64, 64), dtype=np.float32))
                    except Exception as e:
                        print(f"Warning: Could not load heatmap for {heatmap_path}: {e}")
                        # For missing heatmaps, add zero heatmap
                        heatmaps_list.append(np.zeros((64, 64), dtype=np.float32))  # Default 64x64 size
                elif self.use_heatmaps:
                    # Add zero heatmap if heatmaps_dir doesn't exist but heatmaps are enabled
                    heatmaps_list.append(np.zeros((64, 64), dtype=np.float32))
        
        # Apply random frame dropout for data augmentation
        if self.augment and random.random() < 0.2:
            drop_idx = random.randint(0, self.sequence_length - 1)
            frames_resized_list[drop_idx] = np.zeros_like(frames_resized_list[drop_idx])
            diffs_resized_list[drop_idx] = np.zeros_like(diffs_resized_list[drop_idx])
            if self.use_optical_flow and flows_list:
                flows_list[drop_idx] = np.zeros_like(flows_list[drop_idx])
            
        # Apply spatial augmentations to frames and diffs
        final_frames_tensors = []
        final_diffs_tensors = []
        
        for f_resized, d_resized in zip(frames_resized_list, diffs_resized_list):
            augmented = self.transform(image=f_resized, diff_image=d_resized)
            final_frames_tensors.append(augmented['image'])
            final_diffs_tensors.append(augmented['diff_image'])
        
        # Get labels from heatmaps if mixed sample not used
        if not mixed_sample:
            # Load or create heatmaps if not already loaded
            if not heatmaps_list and self.use_heatmaps:
                heatmaps_list = self._load_heatmaps(sample)
                
            # Create labels from heatmaps
            labels_np = self._create_labels_from_heatmaps(heatmaps_list)
            labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
        else:
            # Use the pre-mixed labels
            labels_tensor = torch.tensor(mixed_labels, dtype=torch.float32)
        
        # Create the return dictionary
        result = {
            'frames': torch.stack(final_frames_tensors, dim=0),
            'diffs': torch.stack(final_diffs_tensors, dim=0),
            'labels': labels_tensor,
            'index': idx  # Include sample index for hard mining
        }
        
        # Add optical flow if enabled and available
        if self.use_optical_flow:
            if len(flows_list) > 0:
                # Convert flow arrays to tensors
                flow_tensors = [torch.from_numpy(flow).permute(2, 0, 1) for flow in flows_list]
                result['flows'] = torch.stack(flow_tensors, dim=0)
            else:
                # Create empty flows tensor if no flows available
                dummy_flow = torch.zeros(self.sequence_length, 2, self.input_size[0], self.input_size[1])
                result['flows'] = dummy_flow
        
        # Add heatmaps if enabled
        if self.use_heatmaps:
            if len(heatmaps_list) > 0:
                # Convert heatmap arrays to tensors
                heatmap_tensors = [torch.from_numpy(heatmap) for heatmap in heatmaps_list]
                result['heatmaps'] = torch.stack(heatmap_tensors, dim=0)
            else:
                # Create empty heatmaps tensor if no heatmaps available
                dummy_heatmaps = torch.zeros(self.sequence_length, 64, 64)
                result['heatmaps'] = dummy_heatmaps
            
        return result 
