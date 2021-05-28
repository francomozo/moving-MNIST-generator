# Moving MNIST Generated Dataset
* The moving MNIST dataset was generated using the classic MNIST dataset using code from https://github.com/vincent-leguen/PhyDNet.
* Moving MNIST is a sequence-like dataset consisting of **N** sequences of length **L**. It is stored as N *.npy* files, each of shape *(L, S, S)* where **S** is the shape of the images. Each one is named *"movingMNIST_dataset_seqnum.py"* where *dataset* is *train/test/val* and *seqnum* the sequence number. 
* The files are stored in *"data/dataset"* where *dataset* is *train/test/val*. 


## Obs
* If the classic MNIST dataset is not in data/, download with:
```console
foo@bar:~$ cd scripts
foo@bar:~$ bash download_dataset.sh
```

## Usage
```python
foo@bar:~$ python generator -h
python generate.py -h
usage: generate.py [-h] [-d DATASET] [-nf N_FRAMES] [-ns N_SEQUENCES] [-no NUM_OBJECTS [NUM_OBJECTS ...]] [-sz IMAGE_SIZE] [-sp SPEED]
                   [-boe BOUNCE_OFF_EDGES]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        string: train, test, val. Defaults to train.
  -nf N_FRAMES, --n-frames N_FRAMES
                        Num of frames per sequence. Defaults to 20.
  -ns N_SEQUENCES, --n-sequences N_SEQUENCES
                        Num of sequences. Defaults to 1000.
  -no NUM_OBJECTS [NUM_OBJECTS ...], --num-objects NUM_OBJECTS [NUM_OBJECTS ...]
                        List. Posible digits per sequence. Defaults to [2].
  -sz IMAGE_SIZE, --image-size IMAGE_SIZE
                        Size of images in sequence. Defaults to 64 per dim.
  -sp SPEED, --speed SPEED
                        Speed of digits. Int in [0, 1]. Defaults to 1.
  -boe BOUNCE_OFF_EDGES, --bounce-off-edges BOUNCE_OFF_EDGES
                        Allow digits to leave the image. Defaults to False.
```
