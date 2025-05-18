# ShuttleTrack: High-Precision Shuttlecock Detection and Trajectory Tracking

## Project Overview

**Objective:**  
To develop a state-of-the-art deep learning system for detecting and tracking the shuttlecock in broadcast badminton videos, with the goal of achieving or surpassing the accuracy of leading models such as TrackNetV3 (F1: 98.56%).

**Motivation:**  
Accurate shuttlecock trajectory data is essential for advanced sports analytics, player performance evaluation, and broadcast enhancements. However, the shuttlecock's small size, high speed, and frequent occlusions make this a challenging computer vision problem.

---

## Key Innovations and Methodology

### 1. Model Architecture
- **Hybrid CNN-Transformer:**  
  - **Backbone:** EfficientNetB3 (pretrained), providing robust spatial feature extraction.
  - **Temporal Modeling:** Transformer encoder (4 layers, 8 heads) for learning shuttlecock motion patterns across sequences.
  - **Input:** 6-channel (3 RGB + 3 median-subtracted difference images) to enhance motion and background separation.
  - **Output:** For each frame, the model predicts shuttlecock visibility (classification) and (X, Y) position (regression).

### 2. Preprocessing Pipeline
- **Median Background Subtraction:**  
  - Computes a median image for each segment to generate difference images, improving motion saliency and reducing background noise.
- **Label Normalization:**  
  - All (X, Y) labels are normalized to [0, 1] based on original image dimensions, ensuring consistency across varying resolutions.

### 3. Training Strategy
- **Loss Function:**  
  - Weighted sum of Binary Cross-Entropy (visibility), Mean Squared Error (position), and smoothness loss (trajectory continuity).
  - Current weights: BCE=1.0, MSE=10.0, Smooth=0.05 (actively tuning for best balance).
- **Regularization:**  
  - Dropout, gradient clipping, and data augmentation to prevent overfitting.
- **Metrics:**  
  - Tracks F1, precision, recall (for detection), mean distance error, and "within 5px/10px" accuracy (for localization).

### 4. Dataset
- **Source:**  
  - Using the same dataset and annotation format as TrackNet/TrackNetV3 (publicly available Shuttlecock Trajectory Dataset).
  - Each frame is labeled with visibility and (X, Y) position.

---

## Benchmarks and Targets

| Metric         | TrackNetV3 | Target (ShuttleTrack) |
|----------------|------------|----------------------|
| F1 (Detection) | 98.56%     | ≥99%                 |
| Precision      | 97.79%     | ≥99%                 |
| Recall         | 99.33%     | ≥99%                 |
| Within 5px     | ~90%+      | ≥90%                 |
| Within 10px    | ~98%+      | ≥98%                 |
| Distance Error | Very low   | <0.02 (normalized)   |

---

## Current Status

- **Model and pipeline are fully implemented and running on GCP.**
- **Label alignment and preprocessing have been visually and programmatically verified to be accurate.**
- **Training is ongoing with the following results after 4 epochs:**

| Metric         | Value   |
|----------------|---------|
| F1 (Detection) | 93.6%   |
| Precision      | 89.1%   |
| Recall         | 98.6%   |
| Within 5px     | 2.0%    |
| Within 10px    | 9.4%    |
| Distance Error | 0.1455  |

- **Detection metrics are strong and improving.**
- **Position accuracy is improving but remains the main challenge.**
- **Loss weights have been adjusted to prioritize position accuracy, and further tuning is ongoing.**
- **TensorBoard is used for real-time metric tracking and analysis.**

---

## Next Steps

1. **Continue training for additional epochs to allow the model to further optimize position accuracy.**
2. **If "within 5px" accuracy does not improve significantly by epoch 10, implement advanced techniques:**
   - Further loss balancing
   - Advanced data augmentation
   - Test-time augmentation and ensembling
   - Post-processing (e.g., trajectory smoothing)
   - Architectural tweaks if needed
3. **Benchmark against TrackNetV3 using identical metrics and dataset splits.**
4. **Prepare results and visualizations for publication or presentation.**

---

## Conclusion

ShuttleTrack is a modern, robust, and extensible system for shuttlecock detection and tracking.  
With continued tuning and advanced techniques, we are on track to match or surpass the state-of-the-art performance of TrackNetV3, especially in the most challenging metric: precise position localization. 