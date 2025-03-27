# Pre-import these in the correct order to avoid circular imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Signal that imports are ready
IMPORTS_READY = True