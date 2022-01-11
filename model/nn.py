"""
Various utilities for neural networks.
"""

from enum import Enum
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.utils.checkpoint

import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @th.jit.script
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class CheckpointGNShiftScaleSiLU(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.gn = GroupNorm32(num_groups, num_channels)

    def forward(
        self,
        x,
        scale1: Optional[th.Tensor] = None,
        shift1: Optional[th.Tensor] = None,
        scale2: Optional[th.Tensor] = None,
        shift2: Optional[th.Tensor] = None,
    ):
        return torch_checkpoint(self._forward,
                                (x, scale1, shift1, scale2, shift2), True)

    def _forward(
        self,
        x,
        scale1: Optional[th.Tensor] = None,
        shift1: Optional[th.Tensor] = None,
        scale2: Optional[th.Tensor] = None,
        shift2: Optional[th.Tensor] = None,
    ):
        x = self.gn(x)
        if scale1 is not None:
            x = x * scale1
        if shift1 is not None:
            x = x + shift1
        if scale2 is not None:
            x = x * scale2
        if shift2 is not None:
            x = x + shift2
        x = x * th.sigmoid(x)
        return x


@th.jit.script
def fused_gn_shift_scale_silu(
    x,
    num_groups: int,
    weight,
    bias,
    scale1: th.Tensor,
    shift1: th.Tensor,
):
    x = F.group_norm(x, num_groups, weight, bias)
    x = x * scale1 + shift1
    x = x * th.sigmoid(x)
    return x


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GateType(Enum):
    # always use the b path
    always_b = 'alwaysb'
    # a + b
    add = 'add'
    # a + alpha * b
    alpha_add = 'alphaadd'
    # (1-gate) a + (gate) b
    sigmoid_gate = 'sigmoidgate'
    # (1-alpha) a + (alpha) b
    alpha_gate = 'alphagate'

    def requires_alpha(self):
        return self in [
            GateType.alpha_add, GateType.sigmoid_gate, GateType.alpha_gate
        ]


class GatedConv(nn.Module):
    def __init__(self,
                 a_channels,
                 b_channels,
                 out_channels,
                 kernel_size,
                 padding: int,
                 gate_type: GateType,
                 gate_init: float = 0,
                 has_conv_a: bool = True):
        super().__init__()
        if has_conv_a:
            self.conv_a = nn.Conv2d(a_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)
        else:
            self.conv_a = None
        self.conv_b = nn.Conv2d(a_channels + b_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                padding=padding)

        self.gate_type = gate_type

        if gate_type.requires_alpha():
            self.alpha = nn.Parameter(torch.tensor(gate_init).float())

    def forward(self, a, b=None):
        if self.gate_type != GateType.always_b:
            if self.conv_a is not None:
                h = self.conv_a(a)
            else:
                h = a
        else:
            assert b is not None, 'always b requires b'

        if b is not None:
            b = torch.cat([a, b], dim=1)
            b = self.conv_b(b)

            if self.gate_type == GateType.always_b:
                h = b
            elif self.gate_type == GateType.add:
                h = (h + b) / math.sqrt(2)
            elif self.gate_type == GateType.alpha_add:
                h = (h + self.alpha * b) / torch.sqrt(1 + self.alpha.pow(2))
            elif self.gate_type == GateType.sigmoid_gate:
                gate = torch.sigmoid(self.alpha)
                h = ((1 - gate) * h +
                     gate * b) / torch.sqrt((1 - gate).pow(2) + gate.pow(2))
            elif self.gate_type == GateType.alpha_gate:
                gate = self.alpha
                h = ((1 - gate) * h +
                     gate * b) / torch.sqrt((1 - gate).pow(2) + gate.pow(2))
            else:
                raise NotImplementedError()

        return h


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state)
    else:
        return func(*args)


class CheckpointFunction(th.autograd.Function):
    # doesn't work with automatic mixed precision
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class CloneGrad(torch.autograd.Function):
    """
    a, b => a
    both a, b recieve the same gradient
    """
    @staticmethod
    def forward(ctx, a, b):
        return a

    @staticmethod
    def backward(ctx, grad):
        return grad, grad