import os
import shutil
import subprocess
import yaml
from datetime import datetime

EXPERIMENTS = [
    {'bce': 1.0, 'mse': 5.0, 'smooth': 0.05},
    {'bce': 1.0, 'mse': 7.0, 'smooth': 0.05},
    {'bce': 1.0, 'mse': 10.0, 'smooth': 0.05},
    {'bce': 1.0, 'mse': 7.0, 'smooth': 0.1},
    {'bce': 1.0, 'mse': 10.0, 'smooth': 0.1},
]

BASE_CONFIG = 'config/shuttletrack.yaml'
EXPERIMENTS_DIR = 'experiments'
TRAIN_SCRIPT = 'scripts/train.py'

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

with open(BASE_CONFIG, 'r') as f:
    base_cfg = yaml.safe_load(f)

results = []

for i, exp in enumerate(EXPERIMENTS):
    exp_name = f"exp_bce{exp['bce']}_mse{exp['mse']}_smooth{exp['smooth']}"
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # Copy base config and update loss weights
    cfg = base_cfg.copy()
    cfg['training']['loss_weights'] = exp
    exp_cfg_path = os.path.join(exp_dir, 'shuttletrack.yaml')
    with open(exp_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    # Run training as subprocess
    log_path = os.path.join(exp_dir, 'train.log')
    print(f"[RUNNING] {exp_name}")
    with open(log_path, 'w') as logf:
        proc = subprocess.run([
            'python', TRAIN_SCRIPT,
            '--config', exp_cfg_path
        ], stdout=logf, stderr=subprocess.STDOUT)
    # Optionally, parse results from log or checkpoint
    # For now, just record status
    results.append({'exp': exp_name, 'returncode': proc.returncode, 'log': log_path})

# Summarize
print("\n=== Experiment Results ===")
for r in results:
    print(f"{r['exp']}: {'OK' if r['returncode']==0 else 'FAIL'} | Log: {r['log']}") 