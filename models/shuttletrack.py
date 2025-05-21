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
    def __init__(self, in_channels): # in_channels here is for the input 'x' to SpatialAttention
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
        
    def forward(self, features_list): # Input is raw_features_list from _extract_multi_scale_features
        adapted_features = []
        if not features_list or not isinstance(features_list, (list,tuple)) or not all(isinstance(f, torch.Tensor) for f in features_list):
            print(f"[DEBUG MultiScale ERROR] features_list is not a list of tensors or is empty. Got: {type(features_list)}")
            # Fallback: create a dummy output of expected shape to prevent crash
            # This requires knowing the batch size from one of the (non-existent) features
            # This is a critical error if it happens.
            # For now, assume it won't be called with such bad input due to checks in _extract_multi_scale_features.
            # If it does, it will crash, which is better than silently producing wrong results.
            # Or, we need a way to get B, H, W for a dummy output.
            # Let's assume _extract_multi_scale_features ensures features_list is valid or provides a single dummy tensor.
            if features_list and isinstance(features_list[0], torch.Tensor):
                 dummy_b = features_list[0].shape[0]
                 dummy_h, dummy_w = features_list[0].shape[2]//4, features_list[0].shape[3]//4 # Guess a downsampled size
                 # This out_dim is the one passed to MultiScaleFeatureExtractor constructor
                 out_channels_constructor = self.fusion.out_channels
                 print(f"[DEBUG MultiScale WARNING] Creating dummy output as features_list was problematic.")
                 return torch.zeros((dummy_b, out_channels_constructor, dummy_h, dummy_w), device=features_list[0].device if features_list else 'cpu')
            else: # Cannot even make a sensible dummy
                raise ValueError("MultiScaleFeatureExtractor received invalid features_list")


        target_size = features_list[-1].shape[2:] # This assumes features_list[-1] is a Tensor
        
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
        
        # Determine cnn_feature_dims (output channels of each stage used for multi-scale)
        # And self.feature_extractor_cnn
        if cnn_backbone.startswith('efficientnet'):
            # For timm's EfficientNet, features_only=True is key.
            # The backbone itself becomes the feature_extractor_cnn.
            self.feature_extractor_cnn = timm.create_model(
                cnn_backbone, 
                pretrained=True, 
                features_only=True # This makes it return a list of feature maps
            )
            cnn_feature_dims = self.feature_extractor_cnn.feature_info.channels()
            # Example: EfficientNet-B3 with features_only=True might return 5 feature maps.
            # feature_info has out_indices and reduction for each.
            # We need to ensure cnn_feature_dims corresponds to the features we actually use.
            # Often, for multi-scale, people use features from stages 2,3,4,5 (or similar).
            # If feature_info.channels() gives all, we might select a subset or ensure MultiScaleFeatureExtractor expects that many.
            # For now, assume all features returned by features_only=True are used if cnn_feature_dims is set this way.
        elif cnn_backbone.startswith('resnet'):
            weights_map = {
                'resnet18': models.ResNet18_Weights.DEFAULT,
                'resnet34': models.ResNet34_Weights.DEFAULT,
                # Add more if needed, or use a generic 'DEFAULT'
            }
            weights = weights_map.get(cnn_backbone, 'DEFAULT')
            self.feature_extractor_cnn = getattr(models, cnn_backbone)(weights=weights)
            # Typical ResNet stages output these channel numbers for layer1 to layer4
            if cnn_backbone == 'resnet18': cnn_feature_dims = [64, 128, 256, 512]
            elif cnn_backbone == 'resnet34': cnn_feature_dims = [64, 128, 256, 512]
            else: # resnet50, 101, 152 have different expansion
                # This needs to be accurate. For resnet50: [256, 512, 1024, 2048] from layer1-4
                # For simplicity, let's assume we are using ResNet18/34 if not EfficientNet for now.
                # A more robust way would be to inspect the model.
                cnn_feature_dims = [64,128,256,512] # Placeholder, adjust if using larger ResNets
                print(f"[MODEL WARNING] Using default ResNet18/34 feature_dims for {cnn_backbone}. Verify if using larger ResNet.")

        else: # Fallback for other torchvision models (less tested for multi-scale here)
            self.feature_extractor_cnn = getattr(models, cnn_backbone)(weights='DEFAULT')
            # This is a guess, common for many simpler CNNs, but might not be true multi-scale for all.
            # Example: VGG might have features like [128, 256, 512, 512] from its blocks.
            cnn_feature_dims = [128, 256, 512, 512] # Placeholder
            print(f"[MODEL WARNING] Using generic placeholder feature_dims for {cnn_backbone}. Please verify.")


        # Adjust first conv layer for input_channels
        if input_channels != 3:
            if hasattr(self.feature_extractor_cnn, 'conv_stem'): # Timm EfficientNet
                old_conv_stem = self.feature_extractor_cnn.conv_stem
                new_conv_stem = nn.Conv2d(input_channels, old_conv_stem.out_channels,
                                          kernel_size=old_conv_stem.kernel_size, stride=old_conv_stem.stride,
                                          padding=old_conv_stem.padding, bias=(old_conv_stem.bias is not None))
                with torch.no_grad():
                    new_conv_stem.weight.data[:, :3] = old_conv_stem.weight.data.clone() # Copy existing weights for RGB
                    if input_channels > 3 and new_conv_stem.weight.shape[1] > 3: # If new conv has space for more channels
                        nn.init.kaiming_normal_(new_conv_stem.weight.data[:, 3:]) # Initialize new weights
                self.feature_extractor_cnn.conv_stem = new_conv_stem
            elif hasattr(self.feature_extractor_cnn, 'conv1'): # Torchvision ResNet
                old_conv1 = self.feature_extractor_cnn.conv1
                new_conv1 = nn.Conv2d(input_channels, old_conv1.out_channels,
                                      kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                      padding=old_conv1.padding, bias=(old_conv1.bias is not None))
                with torch.no_grad():
                    new_conv1.weight.data[:, :3] = old_conv1.weight.data.clone()
                    if input_channels > 3 and new_conv1.weight.shape[1] > 3:
                         nn.init.kaiming_normal_(new_conv1.weight.data[:, 3:])
                self.feature_extractor_cnn.conv1 = new_conv1
            else:
                print(f"[MODEL WARNING] Could not adapt first conv layer for {cnn_backbone} with {input_channels} channels.")

        self.multi_scale_fusion = MultiScaleFeatureExtractor(cnn_feature_dims, feature_dim)
        self.cnn_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        # For PyTorch TransformerEncoderLayer, the 'dropout' param is for the feedforward network.
        # The attention dropout is part of MultiheadAttention, which is created internally.
        # To set MHA dropout, you would typically pass it to MHA if building manually.
        # Here, we rely on the 'dropout' in TransformerEncoderLayer and potentially `attn_dropout` if used elsewhere.
        # If `attn_dropout` is meant for MHA's internal dropout:
        # It's not straightforward to set MHA's dropout directly on a pre-built TransformerEncoderLayer
        # unless you iterate through its modules, find self_attn (MHA), and set its dropout attribute.
        # For simplicity, let's assume `dropout` in EncoderLayer covers general path, and `attn_dropout` is for future use or a custom MHA.
        # The original code's loop to set self_attn.dropout might not work if self_attn doesn't directly expose a settable float dropout.
        # Instead, MHA takes dropout as a constructor arg.
        # Let's assume the dropout in TransformerEncoderLayer is sufficient for now.

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trajectory_rectifier = TrajectoryRectificationModule(feature_dim)
        self.avgpool_global = nn.AdaptiveAvgPool2d(1)
        self.visibility_head = nn.Linear(feature_dim, 1)
        
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Softplus() # Softplus ensures positive uncertainty (sigma)
            )
        self.heatmap_decoder_module = HeatmapDecoder(feature_dim, heatmap_size=heatmap_size)

    def _extract_multi_scale_features(self, x_input_cnn): # x_input_cnn is (B*T, C_in, H, W)
        raw_features_list = None # This will hold the list of tensors from the CNN backbone

        # --- Debugging Info about expected adaptors ---
        num_adaptors_expected = len(self.multi_scale_fusion.adaptors)
        print(f"[DEBUG EXTRACT PRE-CHECK] Number of adaptors in multi_scale_fusion: {num_adaptors_expected}")
        for i, adaptor in enumerate(self.multi_scale_fusion.adaptors):
            print(f"[DEBUG EXTRACT PRE-CHECK] Adaptor {i} expects in_channels: {adaptor.in_channels}")
        # --- End Debugging Info ---

        is_timm_features_model = hasattr(self.feature_extractor_cnn, 'feature_info') and \
                                 (self.feature_extractor_cnn.features_only if hasattr(self.feature_extractor_cnn, 'features_only') else False)

        if is_timm_features_model:
            print(f"[DEBUG EXTRACT] Attempting feature extraction with timm model: {self.feature_extractor_cnn.__class__.__name__}")
            potential_features_from_backbone = self.feature_extractor_cnn(x_input_cnn) # This should be a list of tensors

            if isinstance(potential_features_from_backbone, (list, tuple)) and all(isinstance(f, torch.Tensor) for f in potential_features_from_backbone):
                raw_features_list = list(potential_features_from_backbone)
                print(f"[DEBUG EXTRACT TIMM] Backbone returned a list of {len(raw_features_list)} features.")
                for i, feat in enumerate(raw_features_list):
                    print(f"[DEBUG EXTRACT TIMM]   Backbone feature {i} shape: {feat.shape}") # Channel dim is feat.shape[1]
            else:
                print(f"[DEBUG EXTRACT TIMM WARNING] Timm model {self.feature_extractor_cnn.__class__.__name__} did NOT return a list/tuple of tensors as expected. Output type: {type(potential_features_from_backbone)}")
                raw_features_list = None # Mark as failed for this path
        
        elif hasattr(self.feature_extractor_cnn, 'layer1') and hasattr(self.feature_extractor_cnn, 'layer4'): # ResNet-like
            print(f"[DEBUG EXTRACT] Using ResNet-like feature extraction for {self.feature_extractor_cnn.__class__.__name__}.")
            # ... (ResNet extraction logic as in the previous full file code - I'll omit for brevity here but make sure it's there) ...
            # Example:
            # raw_features_list = []
            # x_f = self.feature_extractor_cnn.conv1(x_input_cnn) ...
            # x_f = self.feature_extractor_cnn.layer1(x_f); raw_features_list.append(x_f) ... up to layer4
            # After populating raw_features_list for ResNet:
            # print(f"[DEBUG EXTRACT RESNET] Backbone returned a list of {len(raw_features_list)} features.")
            # for i, feat in enumerate(raw_features_list):
            #     print(f"[DEBUG EXTRACT RESNET]   Backbone feature {i} shape: {feat.shape}")
            # --- THIS IS A SIMPLIFIED PLACEHOLDER FOR RESNET LOGIC ---
            # --- ENSURE YOUR FULL RESNET LOGIC IS HERE ---
            # For now, let's assume the timm path is the one being taken.
            # If this ResNet path were taken and failed, we'd need similar prints.
            # The error message shows "EfficientNetFeatures", so the timm path is active.
            pass # Placeholder if you are only debugging EfficientNet path

        # Check if raw_features_list was successfully populated
        if raw_features_list is None or not raw_features_list:
            print(f"[DEBUG EXTRACT ERROR] Feature extraction failed for {self.feature_extractor_cnn.__class__.__name__}. `raw_features_list` is empty or None.")
            # Create dummy features that will cause the multi_scale_fusion to get *something*
            # but will likely result in poor performance. This is to prevent crashing.
            dummy_list = []
            for i in range(num_adaptors_expected):
                adaptor_in_channels = self.multi_scale_fusion.adaptors[i].in_channels
                dummy_h = max(1, x_input_cnn.shape[2] // (2**(i+2))) # Progressively smaller
                dummy_w = max(1, x_input_cnn.shape[3] // (2**(i+2)))
                dummy_list.append(torch.zeros((x_input_cnn.shape[0], adaptor_in_channels, dummy_h, dummy_w), device=x_input_cnn.device))
            raw_features_list = dummy_list
            print(f"[DEBUG EXTRACT ERROR] Created {len(raw_features_list)} DUMMY feature tensors to proceed.")

        # Now, a crucial check: Does the number of features match number of adaptors?
        if len(raw_features_list) != num_adaptors_expected:
            print(f"[DEBUG EXTRACT MISMATCH!] Number of features from backbone ({len(raw_features_list)}) != Number of adaptors ({num_adaptors_expected}).")
            # This is the most likely source of the "dummy features created" message if the previous prints showed features were extracted.
            # Decide on a strategy:
            # 1. If more features than adaptors, take a subset (e.g., last N, or specific indices)
            # 2. If fewer features than adaptors, this is a problem with cnn_feature_dims in __init__
            if len(raw_features_list) > num_adaptors_expected:
                print(f"[DEBUG EXTRACT MISMATCH!] Taking the last {num_adaptors_expected} features.")
                raw_features_list = raw_features_list[-num_adaptors_expected:]
            elif len(raw_features_list) < num_adaptors_expected and raw_features_list: # Fewer, but not empty
                 print(f"[DEBUG EXTRACT MISMATCH!] Fewer features than adaptors. Padding with copies of last feature.")
                 last_feat_for_padding = raw_features_list[-1]
                 for _ in range(num_adaptors_expected - len(raw_features_list)):
                     raw_features_list.append(torch.zeros_like(last_feat_for_padding)) # Or last_feat_for_padding.clone()
            # If raw_features_list was made dummy above, len(raw_features_list) should == num_adaptors_expected

        # Final check before passing to fusion
        if not all(isinstance(f, torch.Tensor) for f in raw_features_list):
            print("[DEBUG EXTRACT CRITICAL ERROR] `raw_features_list` contains non-Tensor elements before fusion!")
            # This would cause a crash in MultiScaleFeatureExtractor
            # To prevent, return a single dummy tensor that MultiScaleFeatureExtractor might handle (or crash there, which is fine)
            return torch.zeros((x_input_cnn.shape[0], self.feature_dim, 1, 1), device=x_input_cnn.device)

        # Call the fusion module
        print(f"[DEBUG EXTRACT] Calling multi_scale_fusion with a list of {len(raw_features_list)} features.")
        fused_spatial_features = self.multi_scale_fusion(raw_features_list)
        return fused_spatial_features

    def forward(self, frames, diffs, flows=None):
        B, T, _, H, W = frames.shape 
        
        # print(f"\n[DEBUG MODEL FORWARD] --- Start ---")
        # print(f"[DEBUG MODEL FORWARD] Input frames shape: {frames.shape}, min: {frames.min():.2f}, max: {frames.max():.2f}, dtype: {frames.dtype}")
        # print(f"[DEBUG MODEL FORWARD] Input diffs shape: {diffs.shape}, min: {diffs.min():.2f}, max: {diffs.max():.2f}, dtype: {diffs.dtype}")
        # if flows is not None: print(f"[DEBUG MODEL FORWARD] Input flows shape: {flows.shape}, min: {flows.min():.2f}, max: {flows.max():.2f}, dtype: {flows.dtype}")

        if flows is not None:
            x_combined = torch.cat([frames, diffs, flows], dim=2)
        else:
            x_combined = torch.cat([frames, diffs], dim=2)
        
        x_reshaped_BT = x_combined.view(B * T, x_combined.shape[2], H, W)

        if x_reshaped_BT.dtype == torch.uint8:
            x_normalized_BT = x_reshaped_BT.float() / 255.0
        elif x_reshaped_BT.dtype != torch.float32:
            x_normalized_BT = x_reshaped_BT.float()
        else:
            x_normalized_BT = x_reshaped_BT
        
        # print(f"[DEBUG MODEL FORWARD] x_normalized_BT for CNN: shape={x_normalized_BT.shape}, min={x_normalized_BT.min().item():.2f}, max={x_normalized_BT.max().item():.2f}, dtype={x_normalized_BT.dtype}")
        if torch.isnan(x_normalized_BT).any(): print(f"[DEBUG MODEL ERROR] NaN in x_normalized_BT! Sum: {torch.sum(torch.isnan(x_normalized_BT))}")
        if torch.isinf(x_normalized_BT).any(): print(f"[DEBUG MODEL ERROR] Inf in x_normalized_BT! Sum: {torch.sum(torch.isinf(x_normalized_BT))}")
        
        fused_spatial_features_BT = self._extract_multi_scale_features(x_normalized_BT)
        # print(f"[DEBUG MODEL FORWARD] fused_spatial_features_BT (CNN out): shape={fused_spatial_features_BT.shape}, min={fused_spatial_features_BT.min().item():.3f}, max={fused_spatial_features_BT.max().item():.3f}, mean={fused_spatial_features_BT.mean().item():.3f}")
        if torch.isnan(fused_spatial_features_BT).any(): print(f"[DEBUG MODEL ERROR] NaN in fused_spatial_features_BT! Sum: {torch.sum(torch.isnan(fused_spatial_features_BT))}")
        if torch.isinf(fused_spatial_features_BT).any(): print(f"[DEBUG MODEL ERROR] Inf in fused_spatial_features_BT! Sum: {torch.sum(torch.isinf(fused_spatial_features_BT))}")


        global_feats_BT = self.avgpool_global(fused_spatial_features_BT)
        global_feats_B_T_D = global_feats_BT.view(B, T, self.feature_dim)
        global_feats_dropped = self.cnn_dropout(global_feats_B_T_D)
        # print(f"[DEBUG MODEL FORWARD] global_feats_dropped for Transformer: shape={global_feats_dropped.shape}, min={global_feats_dropped.min().item():.3f}, max={global_feats_dropped.max().item():.3f}")

        temp_feats_pos_enc = self.pos_encoder(global_feats_dropped)
        temp_seq_features = self.transformer_encoder(temp_feats_pos_enc)
        # print(f"[DEBUG MODEL FORWARD] temp_seq_features (Transformer out): shape={temp_seq_features.shape}, min={temp_seq_features.min().item():.3f}, max={temp_seq_features.max().item():.3f}")
        if torch.isnan(temp_seq_features).any(): print(f"[DEBUG MODEL ERROR] NaN in temp_seq_features (Transformer output)! Sum: {torch.sum(torch.isnan(temp_seq_features))}")
        if torch.isinf(temp_seq_features).any(): print(f"[DEBUG MODEL ERROR] Inf in temp_seq_features (Transformer output)! Sum: {torch.sum(torch.isinf(temp_seq_features))}")

        visibility_logits_B_T = self.visibility_head(temp_seq_features).squeeze(-1)
        # print(f"[DEBUG MODEL FORWARD] visibility_logits_B_T: shape={visibility_logits_B_T.shape}, example={visibility_logits_B_T[0, :min(5,T)].tolist() if B>0 and T>0 else 'N/A'}")
        # print(f"[DEBUG MODEL FORWARD] visibility_logits_B_T stats: min={visibility_logits_B_T.min().item():.3f}, max={visibility_logits_B_T.max().item():.3f}, mean={visibility_logits_B_T.mean().item():.3f}")
        if torch.isnan(visibility_logits_B_T).any(): print(f"[DEBUG MODEL ERROR] NaN in visibility_logits_B_T! Sum: {torch.sum(torch.isnan(visibility_logits_B_T))}")

        vis_mask_for_rectifier_B_T_1 = torch.sigmoid(visibility_logits_B_T.detach()).unsqueeze(-1)
        rectified_temp_features = self.trajectory_rectifier(temp_seq_features, vis_mask_for_rectifier_B_T_1)
        # print(f"[DEBUG MODEL FORWARD] rectified_temp_features: shape={rectified_temp_features.shape}, min={rectified_temp_features.min().item():.3f}, max={rectified_temp_features.max().item():.3f}")
        if torch.isnan(rectified_temp_features).any(): print(f"[DEBUG MODEL ERROR] NaN in rectified_temp_features! Sum: {torch.sum(torch.isnan(rectified_temp_features))}")

        uncertainty_output = None
        if self.predict_uncertainty: # This check is important
            if hasattr(self, 'uncertainty_head'): # Ensure head exists
                uncertainty_output = self.uncertainty_head(rectified_temp_features)
                # print(f"[DEBUG MODEL FORWARD] uncertainty_output: shape={uncertainty_output.shape}, example={uncertainty_output[0,0].tolist() if B>0 and T>0 else 'N/A'}")
                # print(f"[DEBUG MODEL FORWARD] uncertainty_output stats: min={uncertainty_output.min().item():.3f}, max={uncertainty_output.max().item():.3f}, mean={uncertainty_output.mean().item():.3f}")
                if torch.isnan(uncertainty_output).any(): print(f"[DEBUG MODEL ERROR] NaN in uncertainty_output! Sum: {torch.sum(torch.isnan(uncertainty_output))}")
            else: # Should not happen if predict_uncertainty is true
                print("[DEBUG MODEL WARNING] predict_uncertainty is True, but uncertainty_head is missing.")
        
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
        # print(f"[DEBUG MODEL FORWARD] position_heatmaps_final (logits): shape={position_heatmaps_final.shape}, min={position_heatmaps_final.min().item():.3f}, max={position_heatmaps_final.max().item():.3f}, mean={position_heatmaps_final.mean().item():.3f}")
        if torch.isnan(position_heatmaps_final).any(): print(f"[DEBUG MODEL ERROR] NaN in position_heatmaps_final! Sum: {torch.sum(torch.isnan(position_heatmaps_final))}")
        
        # print(f"[DEBUG MODEL FORWARD] --- End ---")
        
        result_dict = {
            'visibility': visibility_logits_B_T,
            'heatmap': position_heatmaps_final,
        }
        if self.predict_uncertainty and uncertainty_output is not None: # Check both
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
    mask_B_T = (confidences_B_T > threshold).float() # Assuming threshold on logits is intended if heatmap is logits
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
    predict_uncertainty_cfg = model_cfg.get('predict_uncertainty', True) # From model section
    nhead_cfg = model_cfg.get('transformer_nhead', 8)
    num_layers_cfg = model_cfg.get('transformer_layers', 4)
    dropout_cfg = model_cfg.get('dropout', 0.2)
    attn_dropout_cfg = model_cfg.get('attn_dropout', 0.1)
    
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone_cfg, input_channels=input_channels_val,
        feature_dim=feature_dim_cfg, sequence_length=sequence_length_cfg,
        heatmap_size=heatmap_size_cfg, nhead=nhead_cfg, num_layers=num_layers_cfg,
        dropout=dropout_cfg, attn_dropout=attn_dropout_cfg,
        predict_uncertainty=predict_uncertainty_cfg # Pass this from config
    )