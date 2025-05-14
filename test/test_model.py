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
    model.eval()
    # Create dummy input (batch of 2, sequence_length, 3, 224, 224)
    seq_len = config['model']['sequence_length']
    input_size = config['model']['input_size']
    frames = torch.randn(2, seq_len, 3, input_size, input_size).to(device)
    diffs = torch.randn(2, seq_len, 3, input_size, input_size).to(device)
    with torch.no_grad():
        out = model(frames, diffs)
    print('Output shape:', out.shape)  # Should be (2, seq_len, 3)
    print('Model test completed successfully.') 