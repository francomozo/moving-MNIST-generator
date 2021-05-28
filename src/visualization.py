import time

import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_sequence(seq):
    seq = seq.squeeze()
    C, _, _ = seq.shape

    columns = C
    rows = 1
    _, axes = plt.subplots(rows, columns, figsize=(2*C, 2*C))

    for idx, ax in enumerate(axes):
        img = seq[idx]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(idx)
    plt.show()


def plot_on_spot(seq):
    seq = seq.squeeze()
    C, _, _ = seq.shape

    plt.figure()
    for idx in range(C):
        img = seq[idx]
        plt.imshow(img)
        plt.axis('off')
        plt.title(idx + 1)
        plt.show()
        time.sleep(.5)
        clear_output(wait=True)
