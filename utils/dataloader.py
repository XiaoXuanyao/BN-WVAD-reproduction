from torch.utils.data import Dataset, DataLoader, Sampler
import os
import numpy as np
import torch
import torch.nn as nn


# Custom Dataset for loading .npy files and their labels
class MDataset(Dataset):
    def __init__(self, folder):
        self.file_list = []
        self.label_list = []
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                self.file_list.append(os.path.join(folder, file))
                if "Normal" in file:
                    self.label_list.append(0)
                else:
                    self.label_list.append(1)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        data = torch.tensor(data, dtype=torch.float16)
        data = torch.mean(data, dim=-1).float()
        label = self.label_list[idx]
        label = torch.tensor(label, dtype=torch.int)
        return data, label


# Custom Sampler to create balanced batches of normal and abnormal samples
class MSampler(Sampler):
    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0, "Batch size must be even."
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_indices = np.where(self.labels == 0)[0]
        self.neg_indices = np.where(self.labels == 1)[0]
        self.num_batches = max(len(self.pos_indices), len(self.neg_indices)) * 2 // batch_size * 2

    def __iter__(self):
        for i in range(self.num_batches):
            pos = np.random.permutation(self.pos_indices)
            neg = np.random.permutation(self.neg_indices)
            half = self.batch_size // 2
            batch = np.concatenate([pos[:half], neg[:half]])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches