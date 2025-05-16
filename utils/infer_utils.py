import yaml
import torch
import cv2

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

def preprocess_frame(frame, size=224):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size, size))
    frame = frame.astype(float) / 255.0
    return frame 