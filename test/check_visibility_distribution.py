import os
import numpy as np
from tabulate import tabulate

PROCESSED_ROOT = 'processed_data'
SPLITS = ['Train', 'valid']  # You can change this as needed
LOG_FILE = 'test.log.txt'

results = []

for split in SPLITS:
    split_dir = os.path.join(PROCESSED_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    matches = [d for d in os.listdir(split_dir) if d.startswith('match')]
    for match in matches:
        match_dir = os.path.join(split_dir, match)
        segments = os.listdir(match_dir)
        for segment in segments:
            seg_dir = os.path.join(match_dir, segment)
            flows_dir = os.path.join(seg_dir, 'flows')
            heatmaps_dir = os.path.join(seg_dir, 'heatmaps')
            if not (os.path.exists(flows_dir) and os.path.exists(heatmaps_dir)):
                continue
            flow_files = sorted([f for f in os.listdir(flows_dir) if f.endswith('.npz')])
            heatmap_files = sorted([f for f in os.listdir(heatmaps_dir) if f.endswith('.npz')])
            flow_valid = 0
            flow_corrupt = 0
            flow_missing = 0
            for flow_file in flow_files:
                flow_path = os.path.join(flows_dir, flow_file)
                try:
                    arr = np.load(flow_path)
                    if 'flow' in arr and arr['flow'].ndim == 3:
                        flow_valid += 1
                    else:
                        flow_corrupt += 1
                except Exception as e:
                    flow_corrupt += 1
            # Check for missing flows (should match number of frames/heatmaps)
            expected_flows = len(heatmap_files)
            if len(flow_files) < expected_flows:
                flow_missing = expected_flows - len(flow_files)
            # Heatmap visibility check
            visible = 0
            invisible = 0
            for heatmap_file in heatmap_files:
                heatmap_path = os.path.join(heatmaps_dir, heatmap_file)
                try:
                    arr = np.load(heatmap_path)
                    hm = arr['heatmap']
                    if np.max(hm) == 0:
                        invisible += 1
                    else:
                        visible += 1
                except Exception as e:
                    # Treat as invisible if cannot load
                    invisible += 1
            results.append([
                split, match, segment, expected_flows, flow_valid, flow_corrupt, flow_missing, visible, invisible
            ])

# Write results to log file
headers = [
    'Split', 'Match', 'Segment', 'Frames', 'Flow Valid', 'Flow Corrupt', 'Flow Missing', 'Visible', 'Invisible'
]
with open(LOG_FILE, 'w') as f:
    table = tabulate(results, headers=headers, tablefmt='grid')
    f.write(table + '\n')
    # Print summary
    total_frames = sum(r[3] for r in results)
    total_flow_valid = sum(r[4] for r in results)
    total_flow_corrupt = sum(r[5] for r in results)
    total_flow_missing = sum(r[6] for r in results)
    total_visible = sum(r[7] for r in results)
    total_invisible = sum(r[8] for r in results)
    f.write('\nSUMMARY:\n')
    f.write(f'Total frames: {total_frames}\n')
    f.write(f'Total flow valid: {total_flow_valid}\n')
    f.write(f'Total flow corrupt: {total_flow_corrupt}\n')
    f.write(f'Total flow missing: {total_flow_missing}\n')
    f.write(f'Total visible: {total_visible} ({total_visible/total_frames:.1%})\n')
    f.write(f'Total invisible: {total_invisible} ({total_invisible/total_frames:.1%})\n')
print(f"Results saved to {LOG_FILE}")
