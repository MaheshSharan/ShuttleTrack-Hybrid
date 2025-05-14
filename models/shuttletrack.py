import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
    - For each frame (RGB + diff), extract features with a CNN backbone.
    - Stack features across the sequence and feed into a TransformerEncoder for temporal modeling.
    - Output a sequence of predictions (visibility, x, y) for each frame.
    """
    def __init__(self, cnn_backbone='resnet18', input_channels=6, feature_dim=256, sequence_length=5, out_dim=3, nhead=4, num_layers=2):
        super().__init__()
        # CNN backbone (remove avgpool & fc)
        # Use new weights argument to avoid deprecation warning
        if cnn_backbone == 'resnet18':
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif cnn_backbone == 'resnet34':
            from torchvision.models import ResNet34_Weights
            backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            backbone = getattr(models, cnn_backbone)(weights='DEFAULT')
        if input_channels != 3:
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
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.cnn_out_dim = list(backbone.children())[-1].in_features if hasattr(list(backbone.children())[-1], 'in_features') else 512
        self.cnn_proj = nn.Conv2d(self.cnn_out_dim, feature_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        # Positional encoding for temporal sequence
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=sequence_length)
        # TransformerEncoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output head: predict (visibility, x, y) for each frame
        self.head = nn.Linear(feature_dim, out_dim)

    def forward(self, frames, diffs):
        # frames, diffs: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        x = torch.cat([frames, diffs], dim=2)  # (B, T, 6, H, W)
        x = x.view(B * T, 6, H, W)
        feats = self.cnn(x)  # (B*T, F, h, w)
        feats = self.cnn_proj(feats)  # (B*T, feature_dim, h, w)
        feats = self.avgpool(feats).view(B, T, self.feature_dim)  # (B, T, feature_dim)
        feats = self.pos_encoder(feats)  # (B, T, feature_dim)
        seq_feats = self.transformer(feats)  # (B, T, feature_dim)
        preds = self.head(seq_feats)  # (B, T, 3)
        return preds


def build_model_from_config(config):
    model_cfg = config['model']
    cnn_backbone = model_cfg.get('cnn_backbone', 'resnet18')
    input_size = model_cfg.get('input_size', 224)
    sequence_length = model_cfg.get('sequence_length', 5)
    feature_dim = 256
    out_dim = 3
    nhead = model_cfg.get('transformer_nhead', 4)
    num_layers = model_cfg.get('transformer_layers', 2)
    return HybridCNNTransformer(
        cnn_backbone=cnn_backbone,
        input_channels=6,
        feature_dim=feature_dim,
        sequence_length=sequence_length,
        out_dim=out_dim,
        nhead=nhead,
        num_layers=num_layers
    ) 