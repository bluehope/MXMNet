import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch_geometric.data import DataLoader

from qm9_dataset import QM9
from model import MXMNet, Config

import ase
import ase.io


class MyTransform(object):

    def __init__(self,
                target: int) -> None:

        if target in [7, 8, 9, 10]:
            
            target = target + 5

        self.target = target

    def __call__(self,
                data):

        data.y = data.y[:, self.target]

        return data


def main(args: Dict[str, Any]) -> None:

    config = Config(dim = args['dim'], n_layer = args['n_layer'], cutoff = args['cutoff'])
    model = MXMNet(config)
    model.load_state_dict(torch.load(args['model_checkpoint'], map_location = 'cpu'))
    model = model.to('cuda')

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', 'QM9')
    dataset = QM9(path, transform = MyTransform(args['target'])).shuffle()
    print('# of graphs:', len(dataset))

    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    data = next(iter(dataloader)).to('cuda')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type = str,
                        help = 'model checkpoint path')
    parser.add_argument('--n_layer', type = int,
                        default = 6,
                        help = '# of hidden layers')
    parser.add_argument('--dim', type = int,
                        default = 128,
                        help = 'size of input hidden units')
    parser.add_argument('--target', type = int,
                        default = 7,
                        help = 'index of target (0 ~ 11) for prediction')
    parser.add_argument('--cutoff', type = float,
                        default = 5.0,
                        help = 'distance cutoff used in the global layer')
    parser.add_argument('--output', type = str,
                        help = 'figure file path')
    args = vars(parser.parse_args())

    main(args)
