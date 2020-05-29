import time
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
import h5py
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def calculate_nll(paths):
    f1 = h5py.File(paths[0], 'r')
    t1 = f1.get('data')
    t1 = np.array(t1)

    f2 = h5py.File(paths[1], 'r')
    t2 = f1.get('data')
    t2 = np.array(t2)

    return np.square(np.subtract(t1, t2)).mean()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, nargs=2,
                        help=('Path to hdf5 files to calculate reconstruction loss on'))

    args = parser.parse_args()
    loss = calculate_nll(args.path)
    print('NLL / Reconstruction loss (mse): {}'.format(loss))


