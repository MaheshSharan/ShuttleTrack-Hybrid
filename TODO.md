# 🚦 ShuttleTrack Project: Systematic TODO Roadmap

## 1. Dataset Understanding & Exploration
- [x] Inspect dataset folder structure and file types.
- [x] Write a script to:
  - Load a sample sequence (frames + corresponding CSV annotation).
  - Visualize frames with shuttlecock position overlay (if visible).
  - Summarize dataset stats (number of matches, frames, annotation coverage).

## 2. Project Skeleton & Environment Setup
- [x] Create the project directory structure as per README.
- [x] Set up a Python virtual environment.
- [x] Create a requirements.txt with all dependencies.
- [x] Initialize a git repository (if not already).

## 3. Data Preprocessing Pipeline
- [x] Implement a preprocessing script to:
  - Read frames and CSVs, align them.
  - Generate frame difference maps and/or optical flow.
  - Normalize, resize, and augment data (temporal dropout, mixup, jittering).
  - Save processed data for efficient loading.

## 4. Data Loader & Augmentation
- [x] Build a PyTorch Dataset class to:
  - Load sequences of frames and corresponding labels.
  - Apply augmentations on-the-fly.
  - Support both training and validation splits.

## 5. Model Development
- [x] Implement the hybrid CNN + Transformer model:
  - Start with a simple CNN or ViT baseline for pipeline testing.
  - Add temporal fusion and frame difference modules.
  - Integrate trajectory inpainting and output head.

## 6. Training Pipeline
- [x] Write a training script to:
  - Load data using the DataLoader.
  - Train the model with appropriate losses (localization, trajectory continuity).
  - Log metrics and save checkpoints.

## 7. Validation & Evaluation
- [x] Implement validation metrics:
  - Frame-level accuracy (IoU, distance error).
  - Sequence-level trajectory error (L2, F1-score).
- [x] Visualize predictions vs. ground truth.

## 8. Inference & Visualization
- [x] Write an inference script for raw videos.
- [x] Implement trajectory overlay visualization on output videos.

## 9. Benchmarking & Reporting
- [ ] Compare results to baseline (TrackNetV2, etc.).
- [ ] Document results, create plots, and prepare for publication.

## 10. Future Extensions (after MVP)
- [ ] Multi-object detection (players + shuttlecock).
- [ ] 3D projection using court geometry.
- [ ] Real-time deployment.

## 9. 🚀 Advanced Performance Boosting (Apply if accuracy is low)
- [ ] **Data Quality & Augmentation**
    - Clean and consistent annotations
    - Strong spatial and temporal augmentations (crop, jitter, occlusion, MixUp/CutMix)
- [ ] **Model Architecture Tweaks**
    - Longer temporal context (increase sequence length)
    - Deeper Transformer (more layers/heads)
    - Better CNN backbone (ResNet34/50, EfficientNet)
    - Multi-scale feature fusion
- [ ] **Loss Functions & Training Tricks**
    - Focal loss for visibility
    - Smooth L1 loss for coordinates
    - Auxiliary losses (intermediate supervision)
    - Label smoothing
- [ ] **Post-Processing**
    - Trajectory inpainting
    - Kalman filter or spline smoothing
- [ ] **Ensemble & Test-Time Tricks**
    - Model ensembling
    - Test-time augmentation
- [ ] **Hyperparameter Tuning**
    - Learning rate schedules (cosine, one-cycle)
    - Batch size, weight decay, dropout
- [ ] **Advanced Ideas**
    - Attention visualization
    - Self-supervised pretraining
    - Domain adaptation
- [ ] **Evaluation & Error Analysis**
    - Visualize failure cases
    - Per-sequence metrics 