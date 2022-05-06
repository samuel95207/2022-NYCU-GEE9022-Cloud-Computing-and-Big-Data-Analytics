import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd


class NormalDataset(Dataset):
    def __init__(self, filename, x_name, time_steps=100, device=torch.device("cpu")):
        self.df = pd.read_csv(filename)
        self.x_name = x_name

        x_np = self.df[x_name].to_numpy().astype('float32')
        x_np_len = x_np.shape[0]
        x_padding_len = time_steps - x_np_len % time_steps
        x_np = np.concatenate((x_np, x_np[:x_padding_len]))
        x_np = x_np.reshape((int(x_np.shape[0] / time_steps), time_steps))
        self.x = torch.tensor(x_np).to(device)

        self.real_len = x_np_len

    def __getitem__(self, index):
        return self.x[index], 0

    def __len__(self):
        return self.x.shape[0]

    def get_1D_seq(self):
        return self.df[self.x_name].to_numpy().astype('float32')

    def get_labels(self):
        return np.zeros((self.real_len))

    def get_real_len(self):
        return self.real_len


class LabeledDataset(Dataset):
    def __init__(self, filename, x_name, label_name, time_steps=100, device=torch.device("cpu")):
        self.df = pd.read_csv(filename)
        self.x_name = x_name
        self.label_name = label_name

        x_np = self.df[x_name].to_numpy().astype('float32')
        x_np_len = x_np.shape[0]
        x_padding_len = time_steps - x_np_len % time_steps
        x_np = np.concatenate((x_np, x_np[:x_padding_len]))
        x_np = x_np.reshape((int(x_np.shape[0] / time_steps), time_steps))
        self.x = torch.tensor(x_np).to(device)

        labels_np = self.df[label_name].to_numpy().astype('int32')
        labels_np = np.concatenate((labels_np, labels_np[:x_padding_len]))
        labels_np = labels_np.reshape((int(labels_np.shape[0] / time_steps), time_steps))
        self.labels = labels_np

        self.real_len = x_np_len

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return self.x.shape[0]

    def get_1D_seq(self):
        return self.df[self.x_name].to_numpy().astype('float32')

    def get_labels(self):
        return self.df[self.label_name].to_numpy().astype('int32')

    def get_real_len(self):
        return self.real_len
