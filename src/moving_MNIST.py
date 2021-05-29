import gzip
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from numpy.lib.type_check import imag

# The code in this file is from https://github.com/vincent-leguen/PhyDNet
# with some modifications (make digits grow in size, image size, movement, etc).

# This file is for generating the dataset.


def load_mnist(root, image_size):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    if image_size != 64:
        mnist = torch.from_numpy(mnist).unsqueeze(1).float()
        ratio = 28 / 64
        out_size = int(image_size * ratio)
        mnist = F.interpolate(mnist, (out_size, out_size), mode='nearest')
        mnist = mnist.numpy()
    return mnist


def crop_2_size(arr, size):
    # Takes numpy array of shape (C, 1, H, W) and crops in the center
    C, _, H, W = arr.shape
    Hpos, Wpos = int(H / 2), int(W / 2)
    S = int(size / 2)
    arr = arr[:, :, Hpos - S:Hpos + S, Wpos - S:Wpos + S]
    return arr


class MovingMNIST(data.Dataset):
    """[summary]

    Args:
        data ([type]): [description]
    """

    def __init__(self,
                 root,
                 n_frames=10,
                 n_sequences=int(1e4),
                 num_objects=[2],
                 image_size=64,
                 bounce_off_edges=True,
                 speed=1):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.mnist = load_mnist(root, image_size)
        self.length = n_sequences
        self.num_objects = num_objects
        self.n_frames = n_frames
        self.bounce_off_edges = bounce_off_edges
        self.ratio = 28 / 64  # ratio between digits and img size

        # For generating data
        if self.bounce_off_edges:
            self.image_size_ = image_size
        else:
            self.true_image_size_ = image_size
            # int(image_size * (1 + self.ratio))
            self.image_size_ = int(image_size * 1.7)

        self.digit_size_ = 28 if image_size == 64 else int(
            image_size * self.ratio)
        self.step_length_ = speed * 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames, self.image_size_,
                        self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames
        # Sample number of objects
        num_digits = random.choice(self.num_objects)
        # Generate data on the fly
        images = self.generate_moving_mnist(num_digits)

        r = 1  # patch size (a 4 dans les PredRNN)
        w = int(self.image_size_ / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        if not self.bounce_off_edges:
            images = crop_2_size(images, self.true_image_size_)

        images = torch.from_numpy(images / 255.0).contiguous().float()

        out = [int(idx), images]
        return out

    def __len__(self):
        return self.length
