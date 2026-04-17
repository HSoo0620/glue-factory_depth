"""Vendored from gluefactory/utils/tensor.py — batch_to_device + map_tensor."""
from __future__ import annotations

import collections.abc as _abc

import numpy as np  # noqa: F401  (retained for compatibility with upstream call sites)
import torch

_string_classes = (str, bytes)


def map_tensor(input_, func):
    if isinstance(input_, _string_classes):
        return input_
    if isinstance(input_, _abc.Mapping):
        return {k: map_tensor(v, func) for k, v in input_.items()}
    if isinstance(input_, _abc.Sequence):
        return [map_tensor(v, func) for v in input_]
    if input_ is None:
        return None
    return func(input_)


def batch_to_device(batch, device, non_blocking: bool = True):
    def _f(t):
        return t.to(device=device, non_blocking=non_blocking)
    return map_tensor(batch, _f)
