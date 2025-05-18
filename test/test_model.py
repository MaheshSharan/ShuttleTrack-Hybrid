import torch
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.shuttletrack import build_model_from_config

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config = load_config('config/shuttletrack.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    model = build_model_from_config(config).to(device)
    print(f'Model architecture: {config["model"]["cnn_backbone"]} with {config["model"]["transformer_layers"]} transformer layers')
    print(f'Input channels: 6 (3 RGB + 3 diff)')
    
    model.eval()
    
    # Create dummy input (batch of 2, sequence_length, 3, 224, 224)
    seq_len = config['model']['sequence_length']
    input_size = config['model']['input_size']
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
        
    # Creates dummy frames and diffs - both are (batch, seq_len, channels, height, width)
    frames = torch.randn(2, seq_len, 3, input_size[0], input_size[1]).to(device)
    diffs = torch.randn(2, seq_len, 3, input_size[0], input_size[1]).to(device)
    
    print(f"Input shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Diffs: {diffs.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(frames, diffs)
        
    print('Output shape:', outputs.shape)  # Should be (2, seq_len, 3)
    print('Output format: [visibility_logit, x_normalized, y_normalized]')
    
    # Test for numerical issues (NaN or Inf values)
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        print("WARNING: Model produced NaN or Inf values!")
    else:
        print("✓ No numerical issues detected")
        
    print('Model test completed successfully.') 