import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import torch.nn.functional as F
from MI_train import EEGTrainerFineTune
from Models import SCCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SCCNet(4, 1249, 3).to(device)  # channel, samples, class
checkpoint = torch.load("checkpoints/ft-epoch99.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
x_batch = torch.from_numpy(np.empty((1, 1, 4, 1249))).float().to(device)
# print(x_batch)
outputs = model(x_batch)
print(np.argmax(outputs.cpu().detach().numpy()))