"""Weighted contrastive objectives used by CA-EasyRec."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_numpy_inputs(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    if logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional matrix")
    if weights.shape != logits.shape:
        raise ValueError("weights and logits must have the same shape")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must contain one label per logit row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must use an integer dtype")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("each label must index a valid logit column")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("weights must contain finite positive values")
    return logits, labels.astype(np.int64, copy=False), weights


def weighted_info_nce_numpy(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate confidence-weighted InfoNCE with stable log-sum-exp.

    A weight multiplies the corresponding exponential term in the denominator.
    Positive positions are forcibly reset to unit weight.
    """

    logits, labels, weights = _validate_numpy_inputs(logits, labels, weights)
    effective_weights = weights.astype(
        np.result_type(weights.dtype, np.float64), copy=True
    )
    row_indices = np.arange(logits.shape[0])
    effective_weights[row_indices, labels] = 1.0
    adjusted_logits = logits + np.log(effective_weights)
    row_max = adjusted_logits.max(axis=1, keepdims=True)
    log_denominator = row_max[:, 0] + np.log(
        np.exp(adjusted_logits - row_max).sum(axis=1)
    )
    positive_logits = adjusted_logits[row_indices, labels]
    return float(np.mean(log_denominator - positive_logits))


def weighted_info_nce_torch(
    logits: Any,
    labels: Any,
    weights: Any,
) -> Any:
    """PyTorch weighted InfoNCE that preserves gradients only for logits."""

    import torch
    from torch.nn import functional

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional matrix")
    if weights.shape != logits.shape:
        raise ValueError("weights and logits must have the same shape")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must contain one label per logit row")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must use an integer dtype")
    if torch.any(labels < 0).item() or torch.any(labels >= logits.shape[1]).item():
        raise ValueError("each label must index a valid logit column")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values")
    if not torch.isfinite(weights).all().item() or torch.any(weights <= 0.0).item():
        raise ValueError("weights must contain finite positive values")

    labels = labels.to(device=logits.device, dtype=torch.long)
    effective_weights = (
        weights.detach()
        .to(
            device=logits.device,
            dtype=logits.dtype,
        )
        .clone()
    )
    row_indices = torch.arange(logits.shape[0], device=logits.device)
    effective_weights[row_indices, labels] = 1.0
    adjusted_logits = logits + torch.log(effective_weights)
    return functional.cross_entropy(adjusted_logits, labels)
