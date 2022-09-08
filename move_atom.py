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

    new_pos = torch.Tensor([[-1.26981359e-02,  1.08580416e+00,  8.00099580e-03],
                           [ 2.15041600e-03, -6.03131760e-03,  1.97612040e-03],
                           [ 1.01173084e+00,  1.46375116e+00,  2.76574800e-04],
                           [-5.40815069e-01,  1.44752661e+00, -8.76643715e-01],
                           [-5.23813634e-01,  1.43793264e+00,  9.06397294e-01]]).to('cuda')
    

    energy_list = list()
    force_list = list()
    displacement_direction = torch.linspace(-7, 7, 70 * 2)

    for (d_i, delta) in enumerate(displacement_direction):

        data.pos = new_pos.clone()
        data.pos[-1, 1] += delta.to('cuda') # y shift
        data.pos[-1, 2] += delta.to('cuda') # z  shift

        data.pos = data.pos.requires_grad_(True)
        data.pos.retain_grad()

        energy = model(data)
        force = torch.autograd.grad(energy,
                            data.pos,
                            grad_outputs = torch.ones_like(energy).to('cuda'),
                            create_graph = True)[0]
        energy_list.append(energy.detach().cpu().numpy())
        force_list.append(torch.sum(torch.sqrt(torch.sum(force ** 2, dim = -1))).item())

    energy_list = np.array(energy_list)
    force_list = np.array(force_list)

    fig, ax = plt.subplots(2, 1, figsize = (8, 8), sharex = True)

    ax[0].plot(displacement_direction.numpy().squeeze(), energy_list.squeeze(), 'b-')
    ax[1].plot(displacement_direction.numpy().squeeze(), force_list.squeeze(), 'r-')
    ax[0].set_title('MXMNet - Energy')
    ax[1].set_title('MXMNet - Force')

    plt.tight_layout()
    plt.savefig(args['output'], dpi = 300)
    plt.close()


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
