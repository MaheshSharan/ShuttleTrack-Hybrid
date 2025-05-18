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

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN + TransformerEncoder model for shuttlecock detection and trajectory tracking.
    Now outputs a heatmap (1xHxW) for position and a separate visibility head.
    """
    def __init__(self, cnn_backbone='efficientnet_b3', input_channels=6, feature_dim=256, sequence_length=5, heatmap_size=56, nhead=8, num_layers=4, dropout=0.2, attn_dropout=0.1):
        super().__init__()
        # CNN backbone
        if cnn_backbone == 'resnet18':
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.cnn_out_dim = 512
        elif cnn_backbone == 'resnet34':
            from torchvision.models import ResNet34_Weights
            backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.cnn_out_dim = 512
        elif cnn_backbone.startswith('efficientnet'):
            # Use timm for EfficientNet models
            backbone = timm.create_model(cnn_backbone, pretrained=True, features_only=True)
            self.cnn = backbone
            self.cnn_out_dim = backbone.feature_info.channels()[-1]  # Get the last layer's channels
        else:
            backbone = getattr(models, cnn_backbone)(weights='DEFAULT')
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            self.cnn_out_dim = list(backbone.children())[-1].in_features if hasattr(list(backbone.children())[-1], 'in_features') else 512
        
        # Adjust first layer for input channels
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
            else:
                # For ResNet models
                old_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                with torch.no_grad():
                    backbone.conv1.weight[:, :3] = old_conv.weight
                    if input_channels > 3:
                        nn.init.kaiming_normal_(backbone.conv1.weight[:, 3:])
        
        self.cnn_proj = nn.Conv2d(self.cnn_out_dim, feature_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.heatmap_size = heatmap_size
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        for name, param in encoder_layer.named_parameters():
            if 'self_attn' in name and 'dropout' in name:
                encoder_layer.self_attn.dropout = attn_dropout
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output heads
        self.heatmap_head = nn.Linear(feature_dim, heatmap_size * heatmap_size)
        self.visibility_head = nn.Linear(feature_dim, 1)

    def forward(self, frames, diffs):
        B, T, C, H, W = frames.shape
        x = torch.cat([frames, diffs], dim=2)
        x = x.view(B * T, x.shape[2], H, W)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
        if isinstance(self.cnn, nn.Sequential):
            feats = self.cnn(x)
        else:
            feats = self.cnn(x)[-1]
        feats = self.cnn_proj(feats)
        feats = self.avgpool(feats).view(B, T, self.feature_dim)
        feats = self.dropout(feats)
        feats = self.pos_encoder(feats)
        seq_feats = self.transformer(feats)
        # Output heads
        heatmaps = self.heatmap_head(seq_feats)  # (B, T, H*W)
        heatmaps = heatmaps.view(B, T, 1, self.heatmap_size, self.heatmap_size)  # (B, T, 1, H, W)
        visibility = self.visibility_head(seq_feats).squeeze(-1)  # (B, T)
        return {'heatmap': heatmaps, 'visibility': visibility}


def build_model_from_config(config):
    model_cfg = config['model']
    cnn_backbone = model_cfg.get('cnn_backbone', 'efficientnet_b3')
    input_size = model_cfg.get('input_size', 224)
    sequence_length = model_cfg.get('sequence_length', 5)
    feature_dim = 256
    heatmap_size = model_cfg.get('heatmap_size', 56)
    nhead = model_cfg.get('transformer_nhead', 8)
    num_layers = model_cfg.get('transformer_layers', 4)
    dropout = model_cfg.get('dropout', 0.2)
    attn_dropout = model_cfg.get('attn_dropout', 0.1)
    
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone,
        input_channels=6,
        feature_dim=feature_dim,
        sequence_length=sequence_length,
        heatmap_size=heatmap_size,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        attn_dropout=attn_dropout
    ) 