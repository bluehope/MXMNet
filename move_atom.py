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

from qm9_dataset import QM9
from model import MXMNet, Config


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

    dataloader = DataLoader(dataset, batch_size = args['batch_size'], shuffle = False)










# CH4 = QM9 (dsgdb9nsd_000001.xyz)

# AtomZ = [6, 1, 1, 1, 1] 

# Position = [[-1.26981359e-02,  1.08580416e+00,  8.00099580e-03],
#        [ 2.15041600e-03, -6.03131760e-03,  1.97612040e-03],
#        [ 1.01173084e+00,  1.46375116e+00,  2.76574800e-04],
#        [-5.40815069e-01,  1.44752661e+00, -8.76643715e-01],
#        [-5.23813634e-01,  1.43793264e+00,  9.06397294e-01]]

# displacement_direction = torch.linspace(-7,7,70*2)

# for (d_i, delta) in enumerate(z_direction):
#     Position2 = copy.copy(Position)
#     Position2[-1,1] += delta # y shift
#     Position2[-1,2] += delta # z  shift

# Energy, forces = model(Position2, AtomZ)

# mean_force = torch.mean(torch.sqrt(torch.sum(forces**2, dim=1))).item()






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
