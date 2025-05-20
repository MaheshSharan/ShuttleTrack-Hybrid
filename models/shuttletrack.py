import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.conv(attn_map)
        attn_map = torch.sigmoid(attn_map)
        return x * attn_map

class TrajectoryRectificationModule(nn.Module):
    """Predict missing points in a trajectory based on surrounding frames"""
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)
        
    def forward(self, x, vis_mask=None):
        # x: (B, T, D)
        # vis_mask: (B, T, 1) - 1 for visible, 0 for occluded
        if vis_mask is not None:
            # Keep a copy of the original features
            original_x = x.clone()
            
        # Pass through bidirectional GRU to capture trajectory context
        features, _ = self.gru(x)  # (B, T, 2*hidden_dim)
        rectified = self.fc(features)  # (B, T, D)
        
        if vis_mask is not None:
            # Use original features for visible frames, rectified for occluded
            rectified = vis_mask * original_x + (1 - vis_mask) * rectified
            
        return rectified

class HeatmapDecoder(nn.Module):
    """Decode feature maps to a heatmap indicating shuttlecock position"""
    def __init__(self, in_channels, heatmap_size=64):
        super().__init__()
        self.heatmap_size = heatmap_size
        
        # Upsampling decoder with skip connections
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)  # Final 1x1 conv for heatmap
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.conv4(x))
        
        # Final upsampling to target heatmap size
        x = F.interpolate(x, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=False)
        heatmap = self.conv5(x)  # (B, 1, H, W)
        
        return heatmap

class MultiScaleFeatureExtractor(nn.Module):
    """Extract and fuse features from multiple CNN layers"""
    def __init__(self, feature_dims, out_dim):
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1)
            for dim in feature_dims
        ])
        self.fusion = nn.Conv2d(out_dim * len(feature_dims), out_dim, kernel_size=3, padding=1)
        self.spatial_attention = SpatialAttention(out_dim)
        
    def forward(self, features_list):
        # Adapt each feature map to common dimension
        adapted_features = []
        target_size = features_list[-1].shape[2:]  # Use the size of the last feature map
        
        for i, (feature, adaptor) in enumerate(zip(features_list, self.adaptors)):
            x = adaptor(feature)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            adapted_features.append(x)
        
        # Concatenate along channel dimension
        x = torch.cat(adapted_features, dim=1)
        
        # Fuse multi-scale features
        x = self.fusion(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x

class HybridCNNTransformer(nn.Module):
    """
    Enhanced Hybrid CNN + TransformerEncoder model for shuttlecock detection and tracking.
    - Uses multi-scale CNN features with spatial attention
    - Processes RGB frames, difference images, and optical flow
    - Predicts visibility and position heatmaps
    - Uses a trajectory rectification module for temporal consistency
    """
    def __init__(self, cnn_backbone='efficientnet_b3', input_channels=8, feature_dim=256, 
                 sequence_length=5, heatmap_size=64, nhead=8, num_layers=4, 
                 dropout=0.2, attn_dropout=0.1, predict_uncertainty=True):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.predict_uncertainty = predict_uncertainty
        
        # CNN backbone
        if cnn_backbone == 'resnet18':
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.cnn_out_dim = 512
            # Get intermediate features for multi-scale fusion
            self.feature_extractor = backbone
            feature_dims = [64, 128, 256, 512]  # Output channel dims for ResNet18
        elif cnn_backbone == 'resnet34':
            from torchvision.models import ResNet34_Weights
            backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.cnn_out_dim = 512
            self.feature_extractor = backbone
            feature_dims = [64, 128, 256, 512]  # Output channel dims for ResNet34
        elif cnn_backbone.startswith('efficientnet'):
            # Use timm for EfficientNet models
            backbone = timm.create_model(cnn_backbone, pretrained=True, features_only=True)
            self.cnn = backbone
            self.feature_extractor = backbone
            # Get the channel dimensions from each feature map
            feature_dims = backbone.feature_info.channels()
            self.cnn_out_dim = feature_dims[-1]
        else:
            backbone = getattr(models, cnn_backbone)(weights='DEFAULT')
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_extractor = backbone
            self.cnn_out_dim = list(backbone.children())[-1].in_features if hasattr(list(backbone.children())[-1], 'in_features') else 512
            feature_dims = [64, 128, 256, 512]  # Default ResNet-like dims
        
        # Adjust first layer for input channels (RGB + diff + optical flow)
        if input_channels != 3:
            if cnn_backbone.startswith('efficientnet'):
                # For EfficientNet, replace the first Conv2d in the backbone
                old_conv = self.cnn.conv_stem
                new_conv = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    if input_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:])
                self.cnn.conv_stem = new_conv
                # Also update the feature extractor
                self.feature_extractor.conv_stem = new_conv
            else:
                # For ResNet models
                old_conv = self.feature_extractor.conv1
                new_conv = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    if input_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:])
                self.feature_extractor.conv1 = new_conv
                # Update the cnn sequential as well
                if isinstance(self.cnn, nn.Sequential):
                    for i, module in enumerate(self.cnn.children()):
                        if isinstance(module, nn.Conv2d) and i == 0:
                            self.cnn[i] = new_conv
                            break
        
        # Multi-scale feature extraction and fusion
        self.multi_scale = MultiScaleFeatureExtractor(feature_dims, feature_dim)
        
        # Dropout after CNN features
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding for temporal sequence
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        
        # TransformerEncoder with attention dropout for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        # Set attention dropout manually
        for name, param in encoder_layer.named_parameters():
            if 'self_attn' in name and 'dropout' in name:
                encoder_layer.self_attn.dropout = attn_dropout
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Trajectory rectification module
        self.trajectory_rectifier = TrajectoryRectificationModule(feature_dim)
        
        # Global representation for visibility prediction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Output heads
        self.visibility_head = nn.Linear(feature_dim, 1)  # Predict visibility
        
        # Uncertainty prediction head (if enabled)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # Predict (sigma_x, sigma_y)
                nn.Softplus()  # Ensure positive uncertainty values
            )
        
        # Heatmap decoder: predict position as a heatmap
        self.heatmap_decoder = HeatmapDecoder(feature_dim, heatmap_size=heatmap_size)
        
    def _extract_multi_scale_features(self, x):
        # Process with CNN backbone to get multi-scale features
        if isinstance(self.feature_extractor, timm.models.efficientnet.EfficientNet):
            # For EfficientNet from timm, features_only=True already returns all feature maps
            features_list = self.feature_extractor(x)
        else:
            # For other backbones like ResNet, we need to extract features at different stages
            features_list = []
            
            # ResNet specific extraction
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            
            x = self.feature_extractor.layer1(x)
            features_list.append(x)
            
            x = self.feature_extractor.layer2(x)
            features_list.append(x)
            
            x = self.feature_extractor.layer3(x)
            features_list.append(x)
            
            x = self.feature_extractor.layer4(x)
            features_list.append(x)
        
        # Fuse multi-scale features
        fused_features = self.multi_scale(features_list)
        return fused_features

    def forward(self, frames, diffs, flows=None):
        # frames: (B, T, 3, H, W), diffs: (B, T, 3, H, W), flows: (B, T, 2, H, W)
        B, T, C, H, W = frames.shape
        
        # Combine inputs: RGB frames + difference frames + optical flow
        if flows is not None:
            x = torch.cat([frames, diffs, flows], dim=2)  # (B, T, 3+3+2, H, W)
        else:
            x = torch.cat([frames, diffs], dim=2)  # (B, T, 3+3, H, W)
            
        # Reshape for per-frame processing
        BT = B * T
        x = x.view(BT, x.shape[2], H, W)
        
        # Convert to float and normalize
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
            
        # Extract multi-scale features with spatial attention
        fused_features = self._extract_multi_scale_features(x)  # (BT, feature_dim, h, w)
        
        # Get global representation for visibility and feature vector
        global_feats = self.avgpool(fused_features).view(B, T, self.feature_dim)
        
        # Apply dropout to CNN features
        global_feats = self.dropout(global_feats)
        
        # Apply positional encoding and transformer
        feats = self.pos_encoder(global_feats)  # (B, T, feature_dim)
        seq_feats = self.transformer(feats)  # (B, T, feature_dim)
        
        # Predict visibility logits
        visibility_logits = self.visibility_head(seq_feats).squeeze(-1)  # (B, T)
        
        # Get visibility mask for rectification
        visibility_mask = torch.sigmoid(visibility_logits).unsqueeze(-1)  # (B, T, 1)
        
        # Apply trajectory rectification
        rectified_feats = self.trajectory_rectifier(seq_feats, visibility_mask)  # (B, T, feature_dim)
        
        # Predict uncertainty if enabled
        uncertainty = None
        if self.predict_uncertainty:
            # Predict uncertainty values for x,y coordinates
            uncertainty = self.uncertainty_head(rectified_feats)  # (B, T, 2)
        
        # Reshape rectified features for heatmap decoding
        rectified_feats = rectified_feats.view(BT, self.feature_dim, 1, 1)
        
        # Reshape fused_features to match batch size for feature combination
        fused_BT = fused_features.view(B, T, self.feature_dim, fused_features.shape[2], fused_features.shape[3])
        
        # Process each frame individually for heatmap prediction
        heatmaps = []
        for t in range(T):
            # Use the rectified features to guide the spatial features
            frame_features = fused_BT[:, t]  # (B, feature_dim, h, w)
            frame_rectified = rectified_feats.view(B, T, self.feature_dim)[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, feature_dim, 1, 1)
            
            # Combine spatial features with rectified features using attention mechanism
            attention = torch.sigmoid(self.avgpool(frame_features))  # (B, feature_dim, 1, 1)
            attention = attention * frame_rectified  # Modulate attention with rectified features
            
            # Apply attention to spatial features
            attended_features = frame_features * (1.0 + attention)  # (B, feature_dim, h, w)
            
            # Generate heatmap for this frame
            frame_heatmap = self.heatmap_decoder(attended_features)  # (B, 1, heatmap_size, heatmap_size)
            heatmaps.append(frame_heatmap)
            
        # Stack heatmaps along temporal dimension
        position_heatmaps = torch.stack(heatmaps, dim=1)  # (B, T, 1, heatmap_size, heatmap_size)
        position_heatmaps = position_heatmaps.squeeze(2)  # (B, T, heatmap_size, heatmap_size)
        
        # Create result dictionary
        result = {
            'visibility': visibility_logits,  # (B, T)
            'heatmap': position_heatmaps,     # (B, T, heatmap_size, heatmap_size)
        }
        
        # Add uncertainty if predicted
        if uncertainty is not None:
            result['uncertainty'] = uncertainty  # (B, T, 2)
        
        return result


def extract_coordinates_from_heatmap(heatmap, threshold=0.0):
    """
    Extract the (x, y) coordinates from the predicted heatmap.
    Args:
        heatmap: (B, T, H, W) tensor of predicted heatmaps
        threshold: minimum confidence value to consider a detection
    Returns:
        coords: (B, T, 2) tensor of normalized (x, y) coordinates
        confidences: (B, T) tensor of confidence values
    """
    B, T, H, W = heatmap.shape
    device = heatmap.device
    
    # Reshape heatmap to (B*T, H*W)
    heatmap_flat = heatmap.reshape(B*T, -1)
    
    # Get max values and indices
    max_values, max_indices = torch.max(heatmap_flat, dim=1)  # (B*T,)
    
    # Convert indices to (y, x) coordinates
    y = (max_indices // W).float() / H
    x = (max_indices % W).float() / W
    
    # Reshape to (B, T)
    confidences = max_values.view(B, T)
    
    # Create confidence mask based on threshold
    mask = (confidences > threshold).float()
    
    # Combine (x, y) coordinates
    coords = torch.stack([x.view(B, T), y.view(B, T)], dim=-1)  # (B, T, 2)
    
    # Zero out coordinates with low confidence
    coords = coords * mask.unsqueeze(-1)
    
    return coords, confidences


def build_model_from_config(config):
    model_cfg = config['model']
    cnn_backbone = model_cfg.get('cnn_backbone', 'efficientnet_b3')
    input_size_cfg = model_cfg.get('input_size', 224)
    sequence_length = model_cfg.get('sequence_length', 5)
    feature_dim = model_cfg.get('feature_dim', 256)
    heatmap_size = model_cfg.get('heatmap_size', 64)
    
    # Check if optical flow is enabled
    use_flow = model_cfg.get('use_optical_flow', True)
    input_channels = 8 if use_flow else 6  # RGB(3) + Diff(3) + Flow(2)
    
    # Check if uncertainty prediction is enabled
    predict_uncertainty = model_cfg.get('predict_uncertainty', True)
    
    nhead = model_cfg.get('transformer_nhead', 8)
    num_layers = model_cfg.get('transformer_layers', 4)
    dropout = model_cfg.get('dropout', 0.2)
    attn_dropout = model_cfg.get('attn_dropout', 0.1)
    
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone,
        input_channels=input_channels,
        feature_dim=feature_dim,
        sequence_length=sequence_length,
        heatmap_size=heatmap_size,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        attn_dropout=attn_dropout,
        predict_uncertainty=predict_uncertainty
    ) 