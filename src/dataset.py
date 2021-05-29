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

# class MovingMnistDataset(Dataset):
#     """Dataset for loading sequences of the moving mnist dataset.

#     Args:
#         path (str): Path to the .npy files
#         csv ([type]): .csv containing the files names
#         shuffle (bool, optional): Whether to shuffle the sequences. Defaults to False.
#     """

#     def __init__(self, path, csv, shuffle=False):
#         super(MovingMnistDataset, self).__init__()

#         self.path = path
#         self.csv = csv
#         self.df = pd.read_csv(csv, header=None)
#         self.shuffle = shuffle

#         # Holds the names for one sequence at a time.
#         self.sequence_names = self.df.iloc[0]

#         self.curr_seq = 0
#         # Relative index to the current sequence
#         self.rel_idx = 0
#         # Fixes the gap between the __getitem__ idx and the index in the
#         # sequences. (The first idx is a linear idx wo jumps, and the second
#         # jumps from one sequence to the next when the sliding window in one
#         # sequence reaches the end).
#         self.gap = 0

#     def __getitem__(self, idx):
#         if idx == 0:
#             self.curr_seq = 0
#             self.rel_idx = 0
#             self.gap = 0
#             if self.shuffle:
#                 self.df = self.df.sample(frac=1)

#             self.sequence_names = self.df.iloc[self.curr_seq]

#             self.images = torch.FloatTensor([np.load(self.path + img_name)
#                                              for img_name in self.sequence_names])

#         if (idx + 3 * (self.curr_seq + 1)) % 20 == 0:
#             self.curr_seq += 1
#             self.gap = self.curr_seq * 3
#             self.sequence_names = self.df.iloc[self.curr_seq]

#             self.images = torch.FloatTensor([np.load(self.path + img_name)
#                                              for img_name in self.sequence_names])

#         idx += self.gap

#         self.rel_idx = idx % 20
#         idxs = np.arange(self.rel_idx, self.rel_idx + 4)

#         inputs = self.images[self.rel_idx:self.rel_idx + 3, :, :]
#         target = self.images[self.rel_idx + 3, :, :].unsqueeze(dim=0)

#         return self.curr_seq, idxs, inputs, target

#     def __len__(self):
#         return (len(self.sequence_names) - 3) * (self.df.shape[0])
