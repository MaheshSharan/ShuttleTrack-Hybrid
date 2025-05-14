import torch
import yaml
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.shuttletrack import build_model_from_config
from utils.dataset import ShuttleTrackDataset

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == '__main__':
    config = load_config('config/shuttletrack.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    model = build_model_from_config(config).to(device)
    model = load_checkpoint(model, 'checkpoint_best.pth', device)
    model.eval()
    dataset = ShuttleTrackDataset('processed_data', split='valid', sequence_length=config['model']['sequence_length'], augment=False)
    # Visualize the first batch
    sample = dataset[0]
    frames = sample['frames'].unsqueeze(0).to(device)  # (1, T, C, H, W)
    diffs = sample['diffs'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(frames, diffs).cpu().numpy()[0]  # (T, 3)
    labels = sample['labels'].numpy()  # (T, 3)
    # Plot each frame
    for t in range(frames.shape[1]):
        frame = frames[0, t].permute(1, 2, 0).cpu().numpy()
        plt.imshow(frame)
        # True position
        if labels[t, 0] > 0.5:
            plt.scatter([labels[t, 1] * frame.shape[1]], [labels[t, 2] * frame.shape[0]], c='lime', label='True', s=60, marker='o')
        # Predicted position
        if 1 / (1 + np.exp(-pred[t, 0])) > 0.5:
            plt.scatter([pred[t, 1] * frame.shape[1]], [pred[t, 2] * frame.shape[0]], c='red', label='Pred', s=60, marker='x')
        plt.title(f'Frame {t}')
        plt.legend()
        plt.axis('off')
        plt.show() 