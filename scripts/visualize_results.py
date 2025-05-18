import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import ShuttleTrackDataset
from models.shuttletrack import build_model_from_config
import yaml
from glob import glob
from torch.utils.data import DataLoader
from utils.eval import heatmap_to_xy
from tensorboard.backend.event_processing import event_accumulator
import cv2


RESULTS_DIR = 'results/visualizations'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load config and model ---
config_path = 'config/shuttletrack.yaml'
ckpt_path = 'checkpoints/checkpoint_best.pth'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
model = build_model_from_config(config)
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --- Load validation set ---
val_set = ShuttleTrackDataset(
    config['data']['processed_dataset_path'],
    split='valid',
    sequence_length=config['model']['sequence_length'],
    augment=False,
    input_size=config['model']['input_size'],
    augmentation_config=config['training'].get('augmentations', {})
)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# --- Visualize predictions ---
def plot_trajectory(frames, pred_xy, true_xy, save_path):
    fig, ax = plt.subplots()
    img = frames[0].permute(1,2,0).cpu().numpy()
    ax.imshow(img.astype(np.uint8))
    ax.plot(true_xy[:,0]*img.shape[1], true_xy[:,1]*img.shape[0], 'go-', label='GT')
    ax.plot(pred_xy[:,0]*img.shape[1], pred_xy[:,1]*img.shape[0], 'ro-', label='Pred')
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)

def plot_heatmap_overlay(frame, heatmap, save_path):
    img = frame.permute(1,2,0).cpu().numpy().astype(np.uint8)
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = np.stack([heatmap]*3, axis=-1)
    heatmap = np.clip(heatmap, 0, 255)
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    plt.imsave(save_path, overlay)

for i, batch in enumerate(val_loader):
    frames = batch['frames']  # (T, C, H, W)
    heatmaps = batch['heatmap']  # (T, H, W)
    with torch.no_grad():
        pred = model(frames.unsqueeze(0), batch['diffs'].unsqueeze(0))
        pred_xy = heatmap_to_xy(pred['heatmap']).squeeze(0).cpu().numpy()  # (T, 2)
    true_xy = heatmap_to_xy(batch['heatmap'].unsqueeze(0)).squeeze(0).cpu().numpy()
    plot_trajectory(frames, pred_xy, true_xy, os.path.join(RESULTS_DIR, f'traj_{i}.png'))
    # Overlay heatmap for first frame
    plot_heatmap_overlay(frames[0], pred['heatmap'][0,0,0], os.path.join(RESULTS_DIR, f'heatmap_{i}.png'))
    if i >= 20:
        break

# --- Plot training curves from TensorBoard logs ---
def plot_tensorboard_curves(logdir, save_dir):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.figure()
        plt.plot(steps, vals)
        plt.title(tag)
        plt.xlabel('Epoch')
        plt.ylabel(tag)
        plt.savefig(os.path.join(save_dir, f'{tag.replace("/", "_")}.png'))
        plt.close()

plot_tensorboard_curves('runs/shuttletrack', RESULTS_DIR)

print(f"Visualizations saved to {RESULTS_DIR}") 