import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch_geometric.data import DataLoader

from utils import EMA
from model import MXMNet, Config
from qm9_dataset import QM9

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


def measure_force(model: nn.Module,
                dataloader: DataLoader) -> None:

    count_atom = 0
    force_sum = 0

    for data in dataloader:

        data = data.to('cuda')
        data.pos = data.pos.requires_grad_(True)
        data.pos.retain_grad()

        output = model(data)
        force = torch.autograd.grad(output,
                                    data.pos,
                                    grad_outputs = torch.ones_like(output).to('cuda'),
                                    create_graph = True)[0]

        num_atom = data.batch.size(0)
        count_atom += num_atom
        force_sum += torch.sum(torch.sqrt(torch.sum(force ** 2, dim = -1))).item()

    return force_sum / count_atom


def main(args: Dict[str, Any]) -> None:

    config = Config(dim = args['dim'], n_layer = args['n_layer'], cutoff = args['cutoff'])
    model = MXMNet(config)
    model.load_state_dict(torch.load(args['model_checkpoint'], map_location = 'cpu'))
    model = model.to('cuda')

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', 'QM9')
    dataset = QM9(path, transform = MyTransform(args['target'])).shuffle()
    print('# of graphs:', len(dataset))

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    train_loader = DataLoader(train_dataset, batch_size = args['batch_size'], shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = args['batch_size'], shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size = args['batch_size'], shuffle = False)

    train_force = measure_force(model, train_loader)
    test_force = measure_force(model, test_loader)
    val_force = measure_force(model, val_loader)

    print('[info] Model checkpoint: {}'.format(args['model_checkpoint']))
    print('[info] Train force: {:.7f}, Test force: {:.7f}, Valid force: {:.7f}'.format(train_force, test_force, val_force))


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
    parser.add_argument('--dataset', type = str,
                        default = 'QM9',
                        help = 'dataset name')
    parser.add_argument('--batch_size', type = int,
                        default = 128,
                        help = 'batch size')
    parser.add_argument('--target', type = int,
                        default = 7,
                        help = 'index of target (0 ~ 11) for prediction')
    parser.add_argument('--cutoff', type = float,
                        default = 5.0,
                        help = 'distance cutoff used in the global layer')
    args = vars(parser.parse_args())

    main(args)
