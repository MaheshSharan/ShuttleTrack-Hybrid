# models/shuttletrack.py
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
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False) 

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_input = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.conv(attn_input)
        attn_map_sig = torch.sigmoid(attn_map)
        return x * attn_map_sig

class TrajectoryRectificationModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)
        
    def forward(self, x, vis_mask=None):
        original_x_clone = x.clone()
        features_gru, _ = self.gru(x)
        rectified_gru = self.fc(features_gru)
        if vis_mask is not None:
            rectified_output = vis_mask * original_x_clone + (1 - vis_mask) * rectified_gru
        else:
            rectified_output = rectified_gru
        return rectified_output

class HeatmapDecoder(nn.Module):
    def __init__(self, in_channels, heatmap_size=64):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
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
        return heatmap

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, feature_dims, out_dim):
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1) for dim in feature_dims
        ])
        self.fusion = nn.Conv2d(out_dim * len(feature_dims), out_dim, kernel_size=3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(out_dim)
        self.spatial_attention = SpatialAttention(out_dim)
        
    def forward(self, features_list):
        adapted_features = []
        if not features_list or not isinstance(features_list, (list,tuple)) or not all(isinstance(f, torch.Tensor) for f in features_list):
            print(f"[DEBUG MultiScale ERROR] features_list is not a list of tensors or is empty. Got: {type(features_list)}")
            if features_list and isinstance(features_list, list) and features_list[0] is not None and isinstance(features_list[0], torch.Tensor): # Try to make a dummy if possible
                 dummy_b = features_list[0].shape[0]
                 dummy_h, dummy_w = features_list[0].shape[2]//4, features_list[0].shape[3]//4 
                 out_channels_constructor = self.fusion.out_channels
                 print(f"[DEBUG MultiScale WARNING] Creating dummy output as features_list was problematic.")
                 return torch.zeros((dummy_b, out_channels_constructor, dummy_h, dummy_w), device=features_list[0].device)
            else: # Cannot even make a sensible dummy, this will likely lead to subsequent errors or require specific handling
                print(f"[DEBUG MultiScale CRITICAL] Cannot create dummy, input totally unusable.")
                # Returning a single pixel zero tensor, this will almost certainly break things but avoids None.
                # This part indicates a severe upstream problem.
                return torch.zeros((1, self.fusion.out_channels, 1, 1), device='cpu') # A very basic fallback

        target_size = features_list[-1].shape[2:]
        
        for feature_map, adaptor_conv in zip(features_list, self.adaptors):
            x_adapted = adaptor_conv(feature_map)
            if x_adapted.shape[2:] != target_size:
                x_adapted = F.interpolate(x_adapted, size=target_size, mode='bilinear', align_corners=False)
            adapted_features.append(x_adapted)
        
        x_cat = torch.cat(adapted_features, dim=1)
        x_fused = F.relu(self.bn_fusion(self.fusion(x_cat)))
        x_attended = self.spatial_attention(x_fused)
        return x_attended

class HybridCNNTransformer(nn.Module):
    def __init__(self, cnn_backbone='efficientnet_b3', input_channels=8, feature_dim=256, 
                 sequence_length=5, heatmap_size=64, nhead=8, num_layers=4, 
                 dropout=0.2, attn_dropout=0.1, predict_uncertainty=True):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.predict_uncertainty = predict_uncertainty
        
        if cnn_backbone.startswith('efficientnet'):
            self.feature_extractor_cnn = timm.create_model(
                cnn_backbone, 
                pretrained=True, 
                features_only=True 
            )
            # When features_only=True, feature_info.channels() gives the channels for the outputted feature maps
            cnn_feature_dims = self.feature_extractor_cnn.feature_info.channels()
            print(f"[DEBUG __INIT__] EfficientNet: cnn_feature_dims from feature_info: {cnn_feature_dims}")
        elif cnn_backbone.startswith('resnet'):
            weights_map = {'resnet18': models.ResNet18_Weights.DEFAULT, 'resnet34': models.ResNet34_Weights.DEFAULT}
            weights = weights_map.get(cnn_backbone, 'DEFAULT')
            self.feature_extractor_cnn = getattr(models, cnn_backbone)(weights=weights)
            if cnn_backbone == 'resnet18': cnn_feature_dims = [64, 128, 256, 512]
            elif cnn_backbone == 'resnet34': cnn_feature_dims = [64, 128, 256, 512]
            else: cnn_feature_dims = [256, 512, 1024, 2048] # ResNet50+ default block outputs
            print(f"[DEBUG __INIT__] ResNet: cnn_feature_dims set to: {cnn_feature_dims}")
        else: 
            self.feature_extractor_cnn = getattr(models, cnn_backbone)(weights='DEFAULT')
            cnn_feature_dims = [128, 256, 512, 512] 
            print(f"[DEBUG __INIT__ WARNING] Using generic placeholder feature_dims for {cnn_backbone}: {cnn_feature_dims}")

        if input_channels != 3:
            if hasattr(self.feature_extractor_cnn, 'conv_stem'): 
                old_conv_stem = self.feature_extractor_cnn.conv_stem
                new_conv_stem = nn.Conv2d(input_channels, old_conv_stem.out_channels,
                                          kernel_size=old_conv_stem.kernel_size, stride=old_conv_stem.stride,
                                          padding=old_conv_stem.padding, bias=(old_conv_stem.bias is not None))
                with torch.no_grad():
                    new_conv_stem.weight.data[:, :3] = old_conv_stem.weight.data.clone()
                    if input_channels > 3 and new_conv_stem.weight.shape[1] > 3: 
                        nn.init.kaiming_normal_(new_conv_stem.weight.data[:, 3:])
                self.feature_extractor_cnn.conv_stem = new_conv_stem
            elif hasattr(self.feature_extractor_cnn, 'conv1'): 
                old_conv1 = self.feature_extractor_cnn.conv1
                new_conv1 = nn.Conv2d(input_channels, old_conv1.out_channels,
                                      kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                      padding=old_conv1.padding, bias=(old_conv1.bias is not None))
                with torch.no_grad():
                    new_conv1.weight.data[:, :3] = old_conv1.weight.data.clone()
                    if input_channels > 3 and new_conv1.weight.shape[1] > 3:
                         nn.init.kaiming_normal_(new_conv1.weight.data[:, 3:])
                self.feature_extractor_cnn.conv1 = new_conv1
            else: print(f"[MODEL WARNING] Could not adapt first conv layer for {cnn_backbone}")

        self.multi_scale_fusion = MultiScaleFeatureExtractor(cnn_feature_dims, feature_dim)
        self.cnn_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trajectory_rectifier = TrajectoryRectificationModule(feature_dim)
        self.avgpool_global = nn.AdaptiveAvgPool2d(1)
        self.visibility_head = nn.Linear(feature_dim, 1)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softplus())
        self.heatmap_decoder_module = HeatmapDecoder(feature_dim, heatmap_size=heatmap_size)

    def _extract_multi_scale_features(self, x_input_cnn):
        raw_features_list = None

        # --- Debugging Info about expected adaptors ---
        num_adaptors_expected = len(self.multi_scale_fusion.adaptors)
        print(f"[DEBUG EXTRACT PRE-CHECK] Number of adaptors in multi_scale_fusion: {num_adaptors_expected}")
        # for i, adaptor in enumerate(self.multi_scale_fusion.adaptors): # Less verbose for now
        #     print(f"[DEBUG EXTRACT PRE-CHECK] Adaptor {i} expects in_channels: {adaptor.in_channels}")

        # Simplified condition: If it has 'feature_info', it's likely a timm model that supports features_only behavior.
        # The 'features_only' attribute itself might not be present on the model object in all timm versions,
        # but the behavior (returning a list) is tied to how it was created with features_only=True.
        is_likely_timm_features_model = hasattr(self.feature_extractor_cnn, 'feature_info')
        
        print(f"[DEBUG EXTRACT CONDITION CHECK] Backbone class: {self.feature_extractor_cnn.__class__.__name__}")
        print(f"[DEBUG EXTRACT CONDITION CHECK] is_likely_timm_features_model (based on hasattr feature_info): {is_likely_timm_features_model}")


        if is_likely_timm_features_model:
            print(f"[DEBUG EXTRACT] Path taken: Attempting feature extraction with timm model: {self.feature_extractor_cnn.__class__.__name__}")
            potential_features_from_backbone = self.feature_extractor_cnn(x_input_cnn) 

            print(f"[DEBUG EXTRACT TIMM OUTPUT] Type of 'potential_features_from_backbone': {type(potential_features_from_backbone)}")
            if isinstance(potential_features_from_backbone, (list, tuple)):
                print(f"[DEBUG EXTRACT TIMM OUTPUT] Length of list/tuple: {len(potential_features_from_backbone)}")
                all_items_are_tensors = True
                if not potential_features_from_backbone: 
                    print(f"[DEBUG EXTRACT TIMM OUTPUT]   It's an EMPTY list/tuple.")
                    all_items_are_tensors = False 
                for i_feat, feat_item in enumerate(potential_features_from_backbone):
                    if isinstance(feat_item, torch.Tensor):
                        print(f"[DEBUG EXTRACT TIMM OUTPUT]   Item {i_feat} is a Tensor with shape: {feat_item.shape}")
                    else:
                        print(f"[DEBUG EXTRACT TIMM OUTPUT]   Item {i_feat} is NOT a Tensor. Type: {type(feat_item)}")
                        all_items_are_tensors = False
                
                if all_items_are_tensors and potential_features_from_backbone: 
                    raw_features_list = list(potential_features_from_backbone)
                    print(f"[DEBUG EXTRACT TIMM] Successfully got a list of {len(raw_features_list)} feature Tensors from timm backbone.")
                else:
                    print(f"[DEBUG EXTRACT TIMM WARNING] Timm model returned a list/tuple, but it was empty or contained non-Tensor items.")
                    raw_features_list = None 
            elif isinstance(potential_features_from_backbone, torch.Tensor):
                print(f"[DEBUG EXTRACT TIMM OUTPUT] It's a single Tensor with shape: {potential_features_from_backbone.shape}")
                print(f"[DEBUG EXTRACT TIMM WARNING] Timm model returned a single Tensor. If multi_scale_fusion expects multiple, this will be an issue.")
                if num_adaptors_expected == 1: # If only one feature expected by fusion
                    raw_features_list = [potential_features_from_backbone]
                else:
                    raw_features_list = None 
            else:
                print(f"[DEBUG EXTRACT TIMM OUTPUT] It's something else unexpected. Cannot process.")
                raw_features_list = None 
        
        elif hasattr(self.feature_extractor_cnn, 'layer1') and hasattr(self.feature_extractor_cnn, 'layer4'): # ResNet-like
            print(f"[DEBUG EXTRACT] Path taken: ResNet-like feature extraction for {self.feature_extractor_cnn.__class__.__name__}.")
            resnet_features = []
            x_f = self.feature_extractor_cnn.conv1(x_input_cnn)
            x_f = self.feature_extractor_cnn.bn1(x_f)
            x_f = self.feature_extractor_cnn.relu(x_f)
            x_f = self.feature_extractor_cnn.maxpool(x_f)
            x_f = self.feature_extractor_cnn.layer1(x_f); resnet_features.append(x_f)
            x_f = self.feature_extractor_cnn.layer2(x_f); resnet_features.append(x_f)
            x_f = self.feature_extractor_cnn.layer3(x_f); resnet_features.append(x_f)
            x_f = self.feature_extractor_cnn.layer4(x_f); resnet_features.append(x_f)
            raw_features_list = resnet_features
            print(f"[DEBUG EXTRACT RESNET] Backbone returned a list of {len(raw_features_list)} features.")
            # for i, feat in enumerate(raw_features_list): # Less verbose for now
            #     print(f"[DEBUG EXTRACT RESNET]   Backbone feature {i} shape: {feat.shape}")
        else:
             print(f"[DEBUG EXTRACT] Path taken: Fallback - No specific timm (based on feature_info) or ResNet path matched for {self.feature_extractor_cnn.__class__.__name__}.")


        if raw_features_list is None or not raw_features_list: 
            print(f"[DEBUG EXTRACT ERROR] Feature extraction failed OR resulted in an empty list. `raw_features_list` is problematic. Creating DUMMY features.")
            dummy_list = []
            for i in range(num_adaptors_expected):
                adaptor_in_channels = self.multi_scale_fusion.adaptors[i].in_channels
                dummy_h = max(1, x_input_cnn.shape[2] // (2**(i+2))) 
                dummy_w = max(1, x_input_cnn.shape[3] // (2**(i+2)))
                dummy_list.append(torch.zeros((x_input_cnn.shape[0], adaptor_in_channels, dummy_h, dummy_w), device=x_input_cnn.device))
            raw_features_list = dummy_list
            print(f"[DEBUG EXTRACT ERROR] Created {len(raw_features_list)} DUMMY feature tensors with shapes matching adaptor inputs.")

        if len(raw_features_list) != num_adaptors_expected:
            print(f"[DEBUG EXTRACT MISMATCH!] Number of features in `raw_features_list` ({len(raw_features_list)}) != Number of adaptors expected ({num_adaptors_expected}). THIS IS A PROBLEM.")
            # Attempt to reconcile, but this indicates a config/init issue.
            if len(raw_features_list) > num_adaptors_expected:
                raw_features_list = raw_features_list[:num_adaptors_expected] # Take first N
            else: # Fewer features than adaptors
                if raw_features_list: last_valid_feat = raw_features_list[-1]
                else: last_valid_feat = torch.zeros((x_input_cnn.shape[0], self.multi_scale_fusion.adaptors[0].in_channels, 1,1), device=x_input_cnn.device) # Absolute fallback
                
                for i_pad in range(num_adaptors_expected - len(raw_features_list)):
                     missing_adaptor_idx = len(raw_features_list) + i_pad
                     expected_c = self.multi_scale_fusion.adaptors[missing_adaptor_idx].in_channels
                     raw_features_list.append(torch.zeros((last_valid_feat.shape[0], expected_c, last_valid_feat.shape[2], last_valid_feat.shape[3]), device=x_input_cnn.device))
            print(f"[DEBUG EXTRACT MISMATCH!] Adjusted `raw_features_list` to have {len(raw_features_list)} items.")


        if not all(isinstance(f, torch.Tensor) for f in raw_features_list) or not raw_features_list:
            print("[DEBUG EXTRACT CRITICAL ERROR] `raw_features_list` (len {len(raw_features_list) if raw_features_list else 0}) is invalid before fusion call!")
            return torch.zeros((x_input_cnn.shape[0], self.feature_dim, max(1, x_input_cnn.shape[2]//32), max(1, x_input_cnn.shape[3]//32)), device=x_input_cnn.device)

        # print(f"[DEBUG EXTRACT] Calling multi_scale_fusion with a list of {len(raw_features_list)} features. First feature shape (if any): {raw_features_list[0].shape if raw_features_list else 'N/A'}")
        fused_spatial_features = self.multi_scale_fusion(raw_features_list)
        return fused_spatial_features

    def forward(self, frames, diffs, flows=None):
        B, T, _, H, W = frames.shape 
        if flows is not None: x_combined = torch.cat([frames, diffs, flows], dim=2)
        else: x_combined = torch.cat([frames, diffs], dim=2)
        x_reshaped_BT = x_combined.view(B * T, x_combined.shape[2], H, W)
        if x_reshaped_BT.dtype == torch.uint8: x_normalized_BT = x_reshaped_BT.float() / 255.0
        elif x_reshaped_BT.dtype != torch.float32: x_normalized_BT = x_reshaped_BT.float()
        else: x_normalized_BT = x_reshaped_BT
        if torch.isnan(x_normalized_BT).any(): print(f"[DEBUG MODEL ERROR] NaN in x_normalized_BT! Sum: {torch.sum(torch.isnan(x_normalized_BT))}")
        if torch.isinf(x_normalized_BT).any(): print(f"[DEBUG MODEL ERROR] Inf in x_normalized_BT! Sum: {torch.sum(torch.isinf(x_normalized_BT))}")
        
        fused_spatial_features_BT = self._extract_multi_scale_features(x_normalized_BT)
        if torch.isnan(fused_spatial_features_BT).any(): print(f"[DEBUG MODEL ERROR] NaN in fused_spatial_features_BT! Sum: {torch.sum(torch.isnan(fused_spatial_features_BT))}")
        if torch.isinf(fused_spatial_features_BT).any(): print(f"[DEBUG MODEL ERROR] Inf in fused_spatial_features_BT! Sum: {torch.sum(torch.isinf(fused_spatial_features_BT))}")

        global_feats_BT = self.avgpool_global(fused_spatial_features_BT)
        global_feats_B_T_D = global_feats_BT.view(B, T, self.feature_dim)
        global_feats_dropped = self.cnn_dropout(global_feats_B_T_D)
        temp_feats_pos_enc = self.pos_encoder(global_feats_dropped)
        temp_seq_features = self.transformer_encoder(temp_feats_pos_enc)
        if torch.isnan(temp_seq_features).any(): print(f"[DEBUG MODEL ERROR] NaN in temp_seq_features! Sum: {torch.sum(torch.isnan(temp_seq_features))}")
        
        visibility_logits_B_T = self.visibility_head(temp_seq_features).squeeze(-1)
        if torch.isnan(visibility_logits_B_T).any(): print(f"[DEBUG MODEL ERROR] NaN in visibility_logits_B_T! Sum: {torch.sum(torch.isnan(visibility_logits_B_T))}")
        
        vis_mask_for_rectifier_B_T_1 = torch.sigmoid(visibility_logits_B_T.detach()).unsqueeze(-1)
        rectified_temp_features = self.trajectory_rectifier(temp_seq_features, vis_mask_for_rectifier_B_T_1)
        if torch.isnan(rectified_temp_features).any(): print(f"[DEBUG MODEL ERROR] NaN in rectified_temp_features! Sum: {torch.sum(torch.isnan(rectified_temp_features))}")

        uncertainty_output = None
        if self.predict_uncertainty: 
            if hasattr(self, 'uncertainty_head'): 
                uncertainty_output = self.uncertainty_head(rectified_temp_features)
                if torch.isnan(uncertainty_output).any(): print(f"[DEBUG MODEL ERROR] NaN in uncertainty_output! Sum: {torch.sum(torch.isnan(uncertainty_output))}")
            else: print("[DEBUG MODEL WARNING] predict_uncertainty is True, but uncertainty_head is missing.")
        
        fh, fw = fused_spatial_features_BT.shape[2], fused_spatial_features_BT.shape[3]
        fused_spatial_features_B_T_D_fh_fw = fused_spatial_features_BT.view(B, T, self.feature_dim, fh, fw)
        output_heatmaps_list = []
        for t_loop_idx in range(T):
            current_spatial_features = fused_spatial_features_B_T_D_fh_fw[:, t_loop_idx]
            current_rectified_temp = rectified_temp_features[:, t_loop_idx].unsqueeze(-1).unsqueeze(-1)
            attention_gate = torch.sigmoid(self.avgpool_global(current_spatial_features))
            attention_gate_modulated = attention_gate * current_rectified_temp
            attended_spatial_for_decoder = current_spatial_features * (1.0 + attention_gate_modulated)
            frame_heatmap_logits = self.heatmap_decoder_module(attended_spatial_for_decoder)
            output_heatmaps_list.append(frame_heatmap_logits)
        position_heatmaps_final = torch.stack(output_heatmaps_list, dim=1).squeeze(2)
        if torch.isnan(position_heatmaps_final).any(): print(f"[DEBUG MODEL ERROR] NaN in position_heatmaps_final! Sum: {torch.sum(torch.isnan(position_heatmaps_final))}")
        
        result_dict = {'visibility': visibility_logits_B_T, 'heatmap': position_heatmaps_final}
        if self.predict_uncertainty and uncertainty_output is not None: 
            result_dict['uncertainty'] = uncertainty_output
        return result_dict

def extract_coordinates_from_heatmap(heatmap, threshold=0.0):
    B, T, H, W = heatmap.shape
    heatmap_flat = heatmap.reshape(B*T, -1)
    max_values, max_indices = torch.max(heatmap_flat, dim=1)
    y_coords = (max_indices // W).float() / H 
    x_coords = (max_indices % W).float() / W
    coords_BT_2 = torch.stack([x_coords, y_coords], dim=-1)
    coords_B_T_2 = coords_BT_2.view(B, T, 2)
    confidences_B_T = max_values.view(B, T)
    mask_B_T = (confidences_B_T > threshold).float()
    coords_B_T_2_masked = coords_B_T_2 * mask_B_T.unsqueeze(-1)
    return coords_B_T_2_masked, confidences_B_T

def build_model_from_config(config):
    model_cfg = config['model']
    cnn_backbone_cfg = model_cfg.get('cnn_backbone', 'efficientnet_b3')
    sequence_length_cfg = model_cfg.get('sequence_length', 5)
    feature_dim_cfg = model_cfg.get('feature_dim', 256)
    heatmap_size_cfg = model_cfg.get('heatmap_size', 64)
    use_flow_cfg = model_cfg.get('use_optical_flow', True)
    input_channels_val = 8 if use_flow_cfg else 6
    predict_uncertainty_cfg = model_cfg.get('predict_uncertainty', True)
    nhead_cfg = model_cfg.get('transformer_nhead', 8)
    num_layers_cfg = model_cfg.get('transformer_layers', 4)
    dropout_cfg = model_cfg.get('dropout', 0.2)
    attn_dropout_cfg = model_cfg.get('attn_dropout', 0.1)
    
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone_cfg, input_channels=input_channels_val,
        feature_dim=feature_dim_cfg, sequence_length=sequence_length_cfg,
        heatmap_size=heatmap_size_cfg, nhead=nhead_cfg, num_layers=num_layers_cfg,
        dropout=dropout_cfg, attn_dropout=attn_dropout_cfg,
        predict_uncertainty=predict_uncertainty_cfg
    )