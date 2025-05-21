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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels): # in_channels here is for the input 'x' to SpatialAttention
        super(SpatialAttention, self).__init__()
        # This conv takes 2 channels (avg_out, max_out) and outputs 1 channel for the attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False) # Added bias=False common for attn

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_input = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.conv(attn_input)
        attn_map_sig = torch.sigmoid(attn_map)
        return x * attn_map_sig # Element-wise multiplication

class TrajectoryRectificationModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)
        
    def forward(self, x, vis_mask=None):
        original_x_clone = x.clone() # Keep original features for mixing
            
        features_gru, _ = self.gru(x)
        rectified_gru = self.fc(features_gru)
        
        if vis_mask is not None:
            # Use original features for visible frames, rectified for occluded
            # vis_mask is (B, T, 1), 1 for visible.
            # We want original_x for visible, rectified_gru for occluded.
            rectified_output = vis_mask * original_x_clone + (1 - vis_mask) * rectified_gru
        else:
            rectified_output = rectified_gru # Or just x if no vis_mask and no rectification desired?
            
        return rectified_output

class HeatmapDecoder(nn.Module):
    def __init__(self, in_channels, heatmap_size=64):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256) # Added BatchNorm
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128) # Added BatchNorm
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Added BatchNorm
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)  # Added BatchNorm
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=False)
        heatmap = self.conv5(x)
        return heatmap # Output is logits before sigmoid typically

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, feature_dims, out_dim):
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1) for dim in feature_dims
        ])
        # Input to fusion is out_dim * num_feature_maps
        self.fusion = nn.Conv2d(out_dim * len(feature_dims), out_dim, kernel_size=3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(out_dim) # Added BatchNorm
        self.spatial_attention = SpatialAttention(out_dim) # Takes 'out_dim' channels
        
    def forward(self, features_list):
        adapted_features = []
        if not features_list: return None # Handle empty list
        target_size = features_list[-1].shape[2:]
        
        for feature_map, adaptor_conv in zip(features_list, self.adaptors): # Renamed variables
            x_adapted = adaptor_conv(feature_map)
            if x_adapted.shape[2:] != target_size:
                x_adapted = F.interpolate(x_adapted, size=target_size, mode='bilinear', align_corners=False)
            adapted_features.append(x_adapted)
        
        x_cat = torch.cat(adapted_features, dim=1)
        x_fused = F.relu(self.bn_fusion(self.fusion(x_cat))) # ReLU after BN
        x_attended = self.spatial_attention(x_fused) # Spatial attention applied here
        return x_attended

class HybridCNNTransformer(nn.Module):
    def __init__(self, cnn_backbone='efficientnet_b3', input_channels=8, feature_dim=256, 
                 sequence_length=5, heatmap_size=64, nhead=8, num_layers=4, 
                 dropout=0.2, attn_dropout=0.1, predict_uncertainty=True):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim # This is the 'd_model' for transformer and output of multi-scale
        self.predict_uncertainty = predict_uncertainty
        
        if cnn_backbone == 'resnet18':
            from torchvision.models import ResNet18_Weights
            backbone_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_extractor_cnn = backbone_model # For _extract_multi_scale_features
            # feature_dims from ResNet18: layer1=64, layer2=128, layer3=256, layer4=512
            cnn_feature_dims = [64, 128, 256, 512]
        elif cnn_backbone == 'resnet34':
            from torchvision.models import ResNet34_Weights
            backbone_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.feature_extractor_cnn = backbone_model
            cnn_feature_dims = [64, 128, 256, 512]
        elif cnn_backbone.startswith('efficientnet'):
            backbone_model = timm.create_model(cnn_backbone, pretrained=True, features_only=True)
            self.feature_extractor_cnn = backbone_model
            cnn_feature_dims = backbone_model.feature_info.channels()
        else: # Fallback for other torchvision models, assuming ResNet-like structure
            backbone_model = getattr(models, cnn_backbone)(weights='DEFAULT')
            self.feature_extractor_cnn = backbone_model
            # These are common defaults, might need adjustment for other models
            cnn_feature_dims = [c for i, c in enumerate(getattr(backbone_model, 'feature_info', [])) if i < 4] # Try to get from feature_info
            if not cnn_feature_dims: # Fallback if no feature_info
                 cnn_feature_dims = [64,128,256,512] # Placeholder
                 print(f"[MODEL WARNING] Using default feature_dims for {cnn_backbone}. Please verify.")


        # Adjust first conv layer for input_channels
        if input_channels != 3:
            if cnn_backbone.startswith('efficientnet'):
                old_conv_stem = self.feature_extractor_cnn.conv_stem
                new_conv_stem = nn.Conv2d(input_channels, old_conv_stem.out_channels,
                                          kernel_size=old_conv_stem.kernel_size, stride=old_conv_stem.stride,
                                          padding=old_conv_stem.padding, bias=(old_conv_stem.bias is not None))
                with torch.no_grad():
                    new_conv_stem.weight[:, :3] = old_conv_stem.weight
                    if input_channels > 3: nn.init.kaiming_normal_(new_conv_stem.weight[:, 3:])
                self.feature_extractor_cnn.conv_stem = new_conv_stem
            elif hasattr(self.feature_extractor_cnn, 'conv1'): # ResNet-like
                old_conv1 = self.feature_extractor_cnn.conv1
                new_conv1 = nn.Conv2d(input_channels, old_conv1.out_channels,
                                      kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                      padding=old_conv1.padding, bias=(old_conv1.bias is not None))
                with torch.no_grad():
                    new_conv1.weight[:, :3] = old_conv1.weight
                    if input_channels > 3: nn.init.kaiming_normal_(new_conv1.weight[:, 3:])
                self.feature_extractor_cnn.conv1 = new_conv1
            else:
                print(f"[MODEL WARNING] Could not adapt first conv layer for {cnn_backbone} with {input_channels} channels.")

        self.multi_scale_fusion = MultiScaleFeatureExtractor(cnn_feature_dims, feature_dim) # out_dim is self.feature_dim
        self.cnn_dropout = nn.Dropout(dropout) # Renamed for clarity
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, dropout=dropout, # dropout here is for feed-forward
            activation='gelu', batch_first=True, norm_first=True
        )
        # Manually set attention dropout if TransformerEncoderLayer supports it or if using custom attention
        # PyTorch's TransformerEncoderLayer.self_attn.dropout is for the attention weights
        if hasattr(encoder_layer.self_attn, 'dropout') and isinstance(encoder_layer.self_attn.dropout, float): # Check if it's a float to be set
             encoder_layer.self_attn.dropout = attn_dropout # This might not be the standard way if it expects a Module
        else: # If dropout is a module, we might need to replace it or it's already handled by global dropout
            pass # Assuming dropout in TransformerEncoderLayer constructor handles it or it's part of nn.MultiheadAttention

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Renamed
        self.trajectory_rectifier = TrajectoryRectificationModule(feature_dim)
        self.avgpool_global = nn.AdaptiveAvgPool2d(1) # Renamed for clarity
        self.visibility_head = nn.Linear(feature_dim, 1)
        
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Softplus()
            )
        self.heatmap_decoder_module = HeatmapDecoder(feature_dim, heatmap_size=heatmap_size) # Renamed

    def _extract_multi_scale_features(self, x_input_cnn): # Renamed
        # For timm EfficientNet features_only=True
        if hasattr(self.feature_extractor_cnn, 'feature_info') and callable(getattr(self.feature_extractor_cnn, 'forward_features', None)):
            # Typically, timm models with features_only=True return a list of features
            # Or if they have a .forward_features method that we can call.
            # This part depends heavily on the exact timm model structure.
            # The self.feature_extractor_cnn(x_input_cnn) should return a list if features_only=True was effective.
            raw_features_list = self.feature_extractor_cnn(x_input_cnn)
            # Ensure raw_features_list is indeed a list/tuple of tensors.
            # If it's a single tensor, then features_only might not be working as expected, or it's a different timm model type.
            if not isinstance(raw_features_list, (list, tuple)):
                # This might happen if features_only=False or the model doesn't support it directly in __call__
                # Fallback to ResNet-like extraction if it's a single tensor.
                # This is a common source of error: assuming output structure.
                print(f"[DEBUG MODEL _extract_multi] EfficientNet feature_extractor_cnn did not return a list. Type: {type(raw_features_list)}. Attempting ResNet-like extraction.")
                # Fall through to ResNet-like extraction. This needs to be robust.
                # For now, let's assume the timm model gives a list as per its design.
                if not isinstance(raw_features_list, (list, tuple)): # If still not a list, this path is problematic
                    print(f"[DEBUG MODEL ERROR] Cannot get feature list from {self.feature_extractor_cnn.__class__.__name__}")
                    # Create dummy features to prevent crash, but this indicates a setup issue.
                    dummy_shape = (x_input_cnn.shape[0], self.feature_dim, x_input_cnn.shape[2]//32, x_input_cnn.shape[3]//32)
                    return [torch.zeros(dummy_shape, device=x_input_cnn.device)] * len(self.multi_scale_fusion.adaptors)


        # For torchvision ResNet
        elif hasattr(self.feature_extractor_cnn, 'layer1'):
            raw_features_list = []
            x_f = self.feature_extractor_cnn.conv1(x_input_cnn)
            x_f = self.feature_extractor_cnn.bn1(x_f)
            x_f = self.feature_extractor_cnn.relu(x_f)
            x_f = self.feature_extractor_cnn.maxpool(x_f) # After maxpool is end of "stem"
            
            x_f = self.feature_extractor_cnn.layer1(x_f); raw_features_list.append(x_f)
            x_f = self.feature_extractor_cnn.layer2(x_f); raw_features_list.append(x_f)
            x_f = self.feature_extractor_cnn.layer3(x_f); raw_features_list.append(x_f)
            x_f = self.feature_extractor_cnn.layer4(x_f); raw_features_list.append(x_f)
        else:
            print(f"[DEBUG MODEL ERROR] CNN backbone {self.feature_extractor_cnn.__class__.__name__} not supported for multi-scale extraction in this manner.")
            # Fallback: use only the final output of the backbone if it's a sequential-like model
            # This is a guess and might not be multi-scale.
            final_feat = self.feature_extractor_cnn(x_input_cnn) # This might be a single tensor.
            # We need a list for multi_scale_fusion. This part is tricky without knowing the backbone's output.
            # To avoid crashing, create a list of one, but this bypasses multi-scale.
            raw_features_list = [final_feat] * len(self.multi_scale_fusion.adaptors) # This is not ideal for multi-scale.

        # Ensure the number of features matches the number of adaptors
        if len(raw_features_list) != len(self.multi_scale_fusion.adaptors):
            print(f"[DEBUG MODEL WARNING] Mismatch in feature maps ({len(raw_features_list)}) and adaptors ({len(self.multi_scale_fusion.adaptors)}). Using last {len(self.multi_scale_fusion.adaptors)} features if available.")
            raw_features_list = raw_features_list[-len(self.multi_scale_fusion.adaptors):] # Take the last N features

        if not raw_features_list: # If still empty
            print("[DEBUG MODEL ERROR] No features extracted from CNN backbone!")
            dummy_shape = (x_input_cnn.shape[0], self.feature_dim, x_input_cnn.shape[2]//32, x_input_cnn.shape[3]//32)
            return torch.zeros(dummy_shape, device=x_input_cnn.device) # Return a single dummy tensor for fusion to process


        fused_spatial_features = self.multi_scale_fusion(raw_features_list)
        # print(f"[DEBUG MODEL _extract] fused_spatial_features stats: min={fused_spatial_features.min().item():.3f}, max={fused_spatial_features.max().item():.3f}, mean={fused_spatial_features.mean().item():.3f}")
        return fused_spatial_features

    def forward(self, frames, diffs, flows=None):
        B, T, _, H, W = frames.shape # Use _ for C_frames as it's known
        
        # print(f"\n[DEBUG MODEL FORWARD] --- Start ---")
        # print(f"[DEBUG MODEL FORWARD] Input frames shape: {frames.shape}, min: {frames.min():.2f}, max: {frames.max():.2f}")
        # print(f"[DEBUG MODEL FORWARD] Input diffs shape: {diffs.shape}, min: {diffs.min():.2f}, max: {diffs.max():.2f}")
        # if flows is not None: print(f"[DEBUG MODEL FORWARD] Input flows shape: {flows.shape}, min: {flows.min():.2f}, max: {flows.max():.2f}")

        if flows is not None:
            x_combined = torch.cat([frames, diffs, flows], dim=2)
        else:
            x_combined = torch.cat([frames, diffs], dim=2)
        
        # Check expected input channels vs actual
        # cnn_first_layer_in_channels = self.feature_extractor_cnn.conv_stem.in_channels if hasattr(self.feature_extractor_cnn, 'conv_stem') else self.feature_extractor_cnn.conv1.in_channels
        # if x_combined.shape[2] != cnn_first_layer_in_channels:
        #    print(f"[DEBUG MODEL ERROR] Combined input x channels: {x_combined.shape[2]}, but CNN expects: {cnn_first_layer_in_channels}")


        x_reshaped_BT = x_combined.view(B * T, x_combined.shape[2], H, W)

        if x_reshaped_BT.dtype == torch.uint8:
            x_normalized_BT = x_reshaped_BT.float() / 255.0
        elif x_reshaped_BT.dtype != torch.float32: # Handle other types like double
            x_normalized_BT = x_reshaped_BT.float()
        else:
            x_normalized_BT = x_reshaped_BT # Already float32
        
        # print(f"[DEBUG MODEL FORWARD] x_normalized_BT for CNN: shape={x_normalized_BT.shape}, min={x_normalized_BT.min().item():.2f}, max={x_normalized_BT.max().item():.2f}, dtype={x_normalized_BT.dtype}")
        if torch.isnan(x_normalized_BT).any() or torch.isinf(x_normalized_BT).any(): print("[DEBUG MODEL ERROR] NaN/Inf in x_normalized_BT!")
        
        fused_spatial_features_BT = self._extract_multi_scale_features(x_normalized_BT) # (B*T, feature_dim, h, w)
        # print(f"[DEBUG MODEL FORWARD] fused_spatial_features_BT (CNN out): shape={fused_spatial_features_BT.shape}, min={fused_spatial_features_BT.min().item():.3f}, max={fused_spatial_features_BT.max().item():.3f}")
        if torch.isnan(fused_spatial_features_BT).any() or torch.isinf(fused_spatial_features_BT).any(): print("[DEBUG MODEL ERROR] NaN/Inf in fused_spatial_features_BT!")

        # Global features for Transformer path
        global_feats_BT = self.avgpool_global(fused_spatial_features_BT) # (B*T, feature_dim, 1, 1)
        global_feats_B_T_D = global_feats_BT.view(B, T, self.feature_dim) # (B, T, feature_dim)
        global_feats_dropped = self.cnn_dropout(global_feats_B_T_D)
        # print(f"[DEBUG MODEL FORWARD] global_feats_dropped for Transformer: shape={global_feats_dropped.shape}, min={global_feats_dropped.min().item():.3f}, max={global_feats_dropped.max().item():.3f}")

        temp_feats_pos_enc = self.pos_encoder(global_feats_dropped)
        temp_seq_features = self.transformer_encoder(temp_feats_pos_enc) # (B, T, feature_dim)
        # print(f"[DEBUG MODEL FORWARD] temp_seq_features (Transformer out): shape={temp_seq_features.shape}, min={temp_seq_features.min().item():.3f}, max={temp_seq_features.max().item():.3f}")
        if torch.isnan(temp_seq_features).any() or torch.isinf(temp_seq_features).any(): print("[DEBUG MODEL ERROR] NaN/Inf in temp_seq_features (Transformer output)!")

        # Visibility Prediction
        visibility_logits_B_T = self.visibility_head(temp_seq_features).squeeze(-1) # (B, T)
        # print(f"[DEBUG MODEL FORWARD] visibility_logits_B_T: shape={visibility_logits_B_T.shape}, example={visibility_logits_B_T[0, :min(5,T)] if B>0 and T>0 else 'N/A'}")
        # print(f"[DEBUG MODEL FORWARD] visibility_logits_B_T stats: min={visibility_logits_B_T.min().item():.3f}, max={visibility_logits_B_T.max().item():.3f}, mean={visibility_logits_B_T.mean().item():.3f}")

        # Trajectory Rectification
        vis_mask_for_rectifier_B_T_1 = torch.sigmoid(visibility_logits_B_T.detach()).unsqueeze(-1) # Detach logits for mask
        rectified_temp_features = self.trajectory_rectifier(temp_seq_features, vis_mask_for_rectifier_B_T_1) # (B, T, feature_dim)
        # print(f"[DEBUG MODEL FORWARD] rectified_temp_features: shape={rectified_temp_features.shape}, min={rectified_temp_features.min().item():.3f}, max={rectified_temp_features.max().item():.3f}")
        if torch.isnan(rectified_temp_features).any() or torch.isinf(rectified_temp_features).any(): print("[DEBUG MODEL ERROR] NaN/Inf in rectified_temp_features!")

        # Uncertainty Prediction (Optional)
        uncertainty_output = None
        if self.predict_uncertainty:
            uncertainty_output = self.uncertainty_head(rectified_temp_features) # (B, T, 2)
            # print(f"[DEBUG MODEL FORWARD] uncertainty_output: shape={uncertainty_output.shape}, example={uncertainty_output[0,0] if B>0 and T>0 else 'N/A'}")
            # print(f"[DEBUG MODEL FORWARD] uncertainty_output stats: min={uncertainty_output.min().item():.3f}, max={uncertainty_output.max().item():.3f}, mean={uncertainty_output.mean().item():.3f}")
            if torch.isnan(uncertainty_output).any() or torch.isinf(uncertainty_output).any(): print("[DEBUG MODEL ERROR] NaN/Inf in uncertainty_output!")
        
        # Heatmap Decoding
        # Reshape fused_spatial_features_BT to (B, T, D, fh, fw) for per-frame processing in loop
        fh, fw = fused_spatial_features_BT.shape[2], fused_spatial_features_BT.shape[3]
        fused_spatial_features_B_T_D_fh_fw = fused_spatial_features_BT.view(B, T, self.feature_dim, fh, fw)
        
        output_heatmaps_list = []
        for t_loop_idx in range(T):
            current_spatial_features = fused_spatial_features_B_T_D_fh_fw[:, t_loop_idx] # (B, D, fh, fw)
            current_rectified_temp = rectified_temp_features[:, t_loop_idx].unsqueeze(-1).unsqueeze(-1) # (B, D, 1, 1)
            
            # Modulate spatial features with temporal context
            attention_gate = torch.sigmoid(self.avgpool_global(current_spatial_features)) # (B, D, 1, 1)
            attention_gate_modulated = attention_gate * current_rectified_temp
            
            attended_spatial_for_decoder = current_spatial_features * (1.0 + attention_gate_modulated) # Additive/Multiplicative interaction
            
            frame_heatmap_logits = self.heatmap_decoder_module(attended_spatial_for_decoder) # (B, 1, H_map, W_map)
            output_heatmaps_list.append(frame_heatmap_logits)
            
        position_heatmaps_final = torch.stack(output_heatmaps_list, dim=1).squeeze(2) # (B, T, H_map, W_map)
        # print(f"[DEBUG MODEL FORWARD] position_heatmaps_final (logits): shape={position_heatmaps_final.shape}, min={position_heatmaps_final.min().item():.3f}, max={position_heatmaps_final.max().item():.3f}, mean={position_heatmaps_final.mean().item():.3f}")
        if torch.isnan(position_heatmaps_final).any() or torch.isinf(position_heatmaps_final).any(): print("[DEBUG MODEL ERROR] NaN/Inf in position_heatmaps_final!")
        
        # print(f"[DEBUG MODEL FORWARD] --- End ---")
        
        result_dict = {
            'visibility': visibility_logits_B_T,
            'heatmap': position_heatmaps_final,
        }
        if uncertainty_output is not None:
            result_dict['uncertainty'] = uncertainty_output
        
        return result_dict


def extract_coordinates_from_heatmap(heatmap, threshold=0.0): # Keep threshold low if heatmap is logits
    B, T, H, W = heatmap.shape
    
    heatmap_flat = heatmap.reshape(B*T, -1)
    max_values, max_indices = torch.max(heatmap_flat, dim=1)
    
    # Normalize coordinates to [0, 1]
    # y_coords = (max_indices // W).float() / (H -1) if H > 1 else torch.zeros_like(max_indices).float()
    # x_coords = (max_indices % W).float() / (W -1) if W > 1 else torch.zeros_like(max_indices).float()
    # Correct normalization: pixel center / Total_pixels. For H, index is 0 to H-1. Center is index + 0.5.
    # So, (index + 0.5) / H.  Approximate with index / H if H is large.
    # Your original was y = (max_indices // W).float() / H and x = (max_indices % W).float() / W
    # This maps indices [0, H-1] to [0, (H-1)/H]. This is generally fine.
    y_coords = (max_indices // W).float() / H 
    x_coords = (max_indices % W).float() / W
    
    coords_BT_2 = torch.stack([x_coords, y_coords], dim=-1) # (B*T, 2)
    coords_B_T_2 = coords_BT_2.view(B, T, 2) # (B, T, 2)
    
    confidences_B_T = max_values.view(B, T) # (B, T)
    
    # Masking by threshold (on max_values which are heatmap logits/probs)
    # If heatmap is logits, thresholding here might be tricky without sigmoid first.
    # Assuming heatmap values are already scaled (e.g. 0-1 after sigmoid if applied, or we are interested in relative max)
    # Your heatmap_decoder outputs logits, so max_values are logits.
    # If threshold is for confidence *after* sigmoid:
    # confidence_probs = torch.sigmoid(confidences_B_T) 
    # mask_B_T = (confidence_probs > threshold).float()

    # Using threshold directly on max_values (logits or unnormalized scores)
    mask_B_T = (confidences_B_T > threshold).float()

    coords_B_T_2_masked = coords_B_T_2 * mask_B_T.unsqueeze(-1)
    
    return coords_B_T_2_masked, confidences_B_T # Return unmasked coords, and separate confidences for eval


def build_model_from_config(config):
    model_cfg = config['model']
    cnn_backbone_cfg = model_cfg.get('cnn_backbone', 'efficientnet_b3') # Renamed
    # input_size_cfg = model_cfg.get('input_size', 224) # Not used in model constructor directly
    sequence_length_cfg = model_cfg.get('sequence_length', 5) # Renamed
    feature_dim_cfg = model_cfg.get('feature_dim', 256) # Renamed
    heatmap_size_cfg = model_cfg.get('heatmap_size', 64) # Renamed
    
    use_flow_cfg = model_cfg.get('use_optical_flow', True) # Renamed
    input_channels_val = 8 if use_flow_cfg else 6 # Renamed
    
    predict_uncertainty_cfg = model_cfg.get('predict_uncertainty', True) # Renamed
    
    nhead_cfg = model_cfg.get('transformer_nhead', 8) # Renamed
    num_layers_cfg = model_cfg.get('transformer_layers', 4) # Renamed
    dropout_cfg = model_cfg.get('dropout', 0.2) # Renamed
    attn_dropout_cfg = model_cfg.get('attn_dropout', 0.1) # Renamed
    
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone_cfg, input_channels=input_channels_val,
        feature_dim=feature_dim_cfg, sequence_length=sequence_length_cfg,
        heatmap_size=heatmap_size_cfg, nhead=nhead_cfg, num_layers=num_layers_cfg,
        dropout=dropout_cfg, attn_dropout=attn_dropout_cfg,
        predict_uncertainty=predict_uncertainty_cfg
    )