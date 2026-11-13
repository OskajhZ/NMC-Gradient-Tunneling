'''
By Xiangnan Zhang, 2025
    School of Future Technologies, Beijing Institute of Technology

Package initialisation. Sets float32 matmul precision, cuDNN benchmark mode,
and torch.dynamo error suppression. Exports all public symbols from _base.
'''
import torch
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Let torch.compile fall back to eager on unexpected errors (e.g. triton version mismatch)
    torch._dynamo.config.suppress_errors = True

from ._base import *

from . import _base
from . import reservoir
from . import EEG
from . import training_utils
from . import models
