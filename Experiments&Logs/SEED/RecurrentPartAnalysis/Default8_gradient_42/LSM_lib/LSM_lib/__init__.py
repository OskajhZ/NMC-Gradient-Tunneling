'''
By Xiangnan Zhang, 2025
    School of Future Technologies, Beijing Institute of Technology
'''
import torch
torch.set_float32_matmul_precision('high')

from ._base import *

from . import _base
from . import reservoir
from . import EEG
from . import training_utils
from . import models
