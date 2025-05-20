# Section 4: Loss Function Engineering

This document provides an overview of the comprehensive loss function engineering implementation in ShuttleTrack, which was part of the improvement roadmap Section 4.

## 1. Focal Loss for Visibility

Focal Loss helps to focus the learning on hard-to-classify examples by down-weighting the contribution of easy examples. We've implemented it for the visibility prediction task:

```python
# Calculate Binary Cross Entropy (raw)
bce_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
    vis_pred, vis_true, reduction='none'
)

# Apply focal loss to focus on hard examples
# Using gamma=2 as in the original Focal Loss paper
gamma = config.get('training', {}).get('focal_gamma', 2.0)
focal_weight = (1 - torch.sigmoid(vis_pred) * vis_true - (1 - vis_true) * (1 - torch.sigmoid(vis_pred))) ** gamma
focal_bce = (bce_loss_raw * focal_weight).mean()
```

This implementation allows the model to focus more on frames where the shuttlecock visibility is difficult to determine (blurry/occluded frames).

## 2. Dynamic Loss Weighting

We implemented dynamic loss weighting that adjusts the contribution of different loss components based on training progress:

```python
# Dynamic loss weighting based on training progress
if epoch is not None and max_epochs is not None:
    progress = epoch / max_epochs
    
    # Gradually increase weight on trajectory smoothness as training progresses
    smooth_weight_factor = min(1.0, 0.5 + progress)  # Ranges from 0.5 to 1.0
    
    # Gradually decrease focal loss weight as training stabilizes
    focal_weight_factor = max(0.3, 1.0 - progress * 0.7)  # Ranges from 1.0 to 0.3
    
    # Gradually increase uncertainty weight as training progresses
    uncertainty_weight_factor = min(1.0, progress * 2)  # Ranges from 0 to 1.0
```

The approach allows us to:
- Start with more focus on basic detection/localization (higher focal weights)
- Gradually introduce more temporal coherence (increasing smoothness weights)
- Ramp up uncertainty modeling as the main model stabilizes

## 3. Uncertainty Modeling

We've added an auxiliary head to predict the uncertainty (σ) of coordinate predictions:

```python
# Uncertainty prediction head
if predict_uncertainty:
    self.uncertainty_head = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2),  # Predict (sigma_x, sigma_y)
        nn.Softplus()  # Ensure positive uncertainty values
    )
```

The uncertainty values are used to weight the regression loss appropriately:

```python
# Uncertainty Loss (negative log likelihood with uncertainty weighting)
nll_loss = (masked_coord_error**2 / (2 * masked_uncertainty + 1e-8) + torch.log(masked_uncertainty + 1e-8))
uncertainty_loss = torch.sum(nll_loss) / (torch.sum(vis_mask_xy) + 1e-8)
```

This approach follows the principles of probabilistic deep learning, where the model learns to predict both the coordinates and its confidence in those predictions.

## 4. Smooth L1 Loss

We've implemented Smooth L1 Loss for position regression, which is more robust to outliers than MSE:

```python
# Smooth L1 Loss for direct coordinate regression
smooth_l1_loss = torch.nn.functional.smooth_l1_loss(
    pred_coords * vis_true.unsqueeze(-1),  # Only consider visible frames
    xy_true * vis_true.unsqueeze(-1),
    beta=0.1,  # Controls the smoothness, smaller values make it more like L1
    reduction='sum'
) / (torch.sum(vis_true) * 2 + 1e-8)  # Normalize by number of visible points
```

Smooth L1 loss behaves like L2 loss for small errors and like L1 loss for large errors, making it robust to outliers while still providing stable gradients near the optimum.

## 5. Combined Loss Function

All these components are combined into a comprehensive loss function:

```python
total_loss = (
    focal_bce_weight * focal_bce +       # Focal loss for visibility
    mse_weight * mse +                   # MSE loss for heatmap regression
    focal_weight * focal_loss +          # Focal loss for heatmaps
    smooth_weight * smooth_loss +        # Trajectory smoothness
    smooth_l1_weight * smooth_l1_loss +  # Smooth L1 for direct coordinate regression
    uncertainty_weight * uncertainty_loss # Uncertainty loss
)
```

## 6. Configuration

All loss weights and hyperparameters are configurable via the YAML configuration file:

```yaml
training:
  # Loss function weights
  loss_weights:
    bce: 1.0
    focal_bce: 1.5
    mse: 1.0
    focal: 0.8
    smooth: 0.15
    smooth_l1: 0.5
    uncertainty: 0.2
  
  # Focal loss parameters
  focal_gamma: 2.0
```

This ensures flexibility for experimentation and hyperparameter tuning.

## Benefits

The comprehensive loss function implementation provides several benefits:

1. **Robust learning** with focal loss focusing on hard examples
2. **Adaptive training** through dynamic loss weighting
3. **Uncertainty awareness** that helps the model handle ambiguous cases
4. **Outlier robustness** through Smooth L1 loss
5. **Temporal consistency** with trajectory smoothness loss

These improvements should significantly enhance the model's ability to track shuttlecocks accurately, even in challenging conditions like occlusions, blur, or fast movement. 