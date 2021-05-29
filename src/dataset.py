import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset to load the data.


class MovingMnistDataset(Dataset):
    def __init__(self, path, n_frames=4, shuffle=False):
        super(MovingMnistDataset, self).__init__()

        self.path = path
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.filenames = sorted(os.listdir(path))

        # Aux variables for indexing correctly
        self.seq_num = 0
        self.rel_idx = 0
        self.seq_length = np.load(
            self.path + self.filenames[self.seq_num]).shape[0]

    def __getitem__(self, idx):

        # some initialization for each epoch
        if idx == 0 and self.shuffle:
            random.shuffle(self.filenames)
        if idx == 0:
            self.seq_num = 0

        # sequence number management
        if (idx + (self.n_frames - 1) * (self.seq_num + 1)) % self.seq_length == 0:
            self.seq_num += 1

        # index within the sequence
        self.rel_idx = (idx + (self.n_frames - 1) *
                        self.seq_num) % self.seq_length
        if idx <= self.seq_length - self.n_frames:
            self.rel_idx = idx

        # images loading
        if self.rel_idx == 0:
            self.images = torch.FloatTensor(
                [np.load(self.path + self.filenames[self.seq_num])]
            ).squeeze()

        frames_in = self.images[self.rel_idx: self.rel_idx+self.n_frames-1]
        frames_out = self.images[self.n_frames-1].unsqueeze(dim=0)

        # return indexes
        curr_idxs = np.arange(self.rel_idx, self.rel_idx + self.n_frames)

        return self.seq_num, curr_idxs, frames_in, frames_out

    def __len__(self):
        return ((self.seq_length - self.n_frames) * len(self.filenames))
