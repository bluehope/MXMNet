import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EMA
from model import MXMNet, Config
from qm9_dataset import QM9


def main(args: Dict[str, Any]) -> None:

    pass


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
