import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.eval import compute_distance_error, compute_visibility_f1

# Example usage: python utils/benchmark.py --ours predictions.csv --baseline baseline.csv --output results/benchmark_report.md
import argparse
parser = argparse.ArgumentParser(description='Benchmark ShuttleTrack vs. Baseline')
parser.add_argument('--ours', type=str, required=True, help='Path to our model predictions CSV')
parser.add_argument('--baseline', type=str, required=True, help='Path to baseline predictions CSV')
parser.add_argument('--gt', type=str, required=True, help='Path to ground truth CSV')
parser.add_argument('--output', type=str, default='results/benchmark_report.md', help='Path to output report')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

def load_preds(path):
    df = pd.read_csv(path)
    vis = df['visibility'].values.reshape(-1, 1)
    xy = df[['x', 'y']].values.reshape(-1, 2)
    return vis, xy

def main():
    ours_vis, ours_xy = load_preds(args.ours)
    base_vis, base_xy = load_preds(args.baseline)
    gt_df = pd.read_csv(args.gt)
    gt_vis = gt_df['visibility'].values.reshape(-1, 1)
    gt_xy = gt_df[['x', 'y']].values.reshape(-1, 2)
    # Metrics
    ours_dist = compute_distance_error(ours_xy[None, ...], gt_xy[None, ...], visibility=gt_vis[None, ...])
    base_dist = compute_distance_error(base_xy[None, ...], gt_xy[None, ...], visibility=gt_vis[None, ...])
    ours_f1 = compute_visibility_f1(ours_vis[None, ...], gt_vis[None, ...])
    base_f1 = compute_visibility_f1(base_vis[None, ...], gt_vis[None, ...])
    # Report
    with open(args.output, 'w') as f:
        f.write('# Benchmark Report\n')
        f.write('| Model | Distance Error | Visibility F1 |\n')
        f.write('|-------|----------------|---------------|\n')
        f.write(f'| ShuttleTrack | {ours_dist:.4f} | {ours_f1:.4f} |\n')
        f.write(f'| Baseline     | {base_dist:.4f} | {base_f1:.4f} |\n')
    # Plot
    plt.figure(figsize=(6,4))
    plt.bar(['ShuttleTrack', 'Baseline'], [ours_dist, base_dist], color=['blue', 'gray'])
    plt.ylabel('Distance Error')
    plt.title('Trajectory Error Comparison')
    plt.savefig(os.path.join(os.path.dirname(args.output), 'distance_error.png'))
    print(f'Benchmark report saved to {args.output}')

if __name__ == '__main__':
    main() 