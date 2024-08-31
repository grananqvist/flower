"""Define custom layers for model."""
# Code in this file is adapted from:
#
# https://github.com/apple/pfl-research/blob/main/benchmarks/model/pytorch/layer.py
# Copyright © 2023-2024 Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

from abc import ABC

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase


class _FrozenBatchNorm(_NormBase, ABC):
    """Freeze the statistics during training.

    A special batch normalization module that will freeze the statistics during training
    and only update the affine parameters.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """PyTorch forward propagation."""
        self._check_input_dim(input)

        # turn of training so no batchnorm statistics is collected
        # and use pretrained statistics in training as well
        self.training = False

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            # If buffers are not tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class FrozenBatchNorm1D(_FrozenBatchNorm):
    """Batch norm with frozed statistics for 1D inputs."""

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class FrozenBatchNorm2D(_FrozenBatchNorm):
    """Batch norm with frozed statistics for 2D inputs."""

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class FrozenBatchNorm3D(_FrozenBatchNorm):
    """Batch norm with frozed statistics for 3D inputs."""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
