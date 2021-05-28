import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hf import str2bool
from src.moving_MNIST import MovingMNIST

# Parsing arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", default='train',
                help="string: train, test, val. Defaults to train.")
ap.add_argument("-nf", "--n-frames", default=20, type=int,
                help="Num of frames per sequence. Defaults to 20.")
ap.add_argument("-ns", "--n-sequences", default=1000, type=int,
                help="Num of sequences. Defaults to 1000.")
ap.add_argument("-no", "--num-objects", default=[2], nargs="+", type=int,
                help="List. Posible digits per sequence. Defaults to [2].")
ap.add_argument("-sz", "--image-size", default=64, type=int,
                help="Size of images in sequence. Defaults to 64 per dim.")
ap.add_argument("-sp", "--speed", default=1, type=int,
                help="Speed of digits. Int in [0, 1]. Defaults to 1.")
ap.add_argument("-boe", "--bounce-off-edges", default=False, type=str2bool,
                help="Allow digits to leave the image. Defaults to False.")

params = vars(ap.parse_args())
params['dir'] = 'data/' + params['dataset']

# for key in params:
#     print(key, params[key], type(params[key]))


try:
    os.mkdir(params['dir'])
except OSError:
    print(f'Failed to create {params["dir"]}. It may already exist.')
else:
    print(f'Successfully created the directory {params["dir"]}.')

# Generate data
mm = MovingMNIST(
    root='data',
    n_frames=params['n_frames'],
    n_sequences=params['n_sequences'],
    num_objects=params['num_objects'],
    image_size=params['image_size'],
    speed=params['speed'],
    bounce_off_edges=params['bounce_off_edges']
)

loader = DataLoader(
    dataset=mm,
    batch_size=1,
)

with tqdm(loader, desc=f'Sequence ', unit='sequences') as loader_pbar:
    for idx, sequence in loader_pbar:

        seq = sequence.squeeze().numpy()

        filename = 'movMNIST_' + \
            params['dataset'] + '_' + format(int(idx), '04d')
        path = params['dir'] + '/' + filename

        np.save(path, seq)
