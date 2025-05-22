import numpy as np
f = np.load('processed_data/Train/match17/2_01_01/flows/000006.npz')
print(f.files)
print(f['flow'].shape)