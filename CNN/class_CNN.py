import numpy as np
import torch
from torch.utils.data import Dataset

class DatasetClass(Dataset):  # 🔁 inherits from PyTorch Dataset
    def __init__(self, data, max_len=127):
        self.data = data  # list of (spec, label)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spec, label = self.data[index]

        # Pad spectrogram on time axis to max_len
        padded_spec = np.pad(
            spec,
            pad_width=((0, 0), (0, self.max_len - spec.shape[1])),  # (freq_pad, time_pad)
            mode='constant',
            constant_values=-80  # silence in dB
        )

        # Convert to PyTorch tensors
        padded_spec = torch.tensor(padded_spec, dtype=torch.float32).unsqueeze(0)  # shape: (1, 128, max_len)
        label = torch.tensor(label, dtype=torch.long)

        return padded_spec, label
