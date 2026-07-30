"""Confidence calibration from the CA-EasyRec manuscript.

The NumPy implementation is an auditable reference.  The PyTorch implementation
has the same contract and explicitly detaches all teacher-derived values.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_parameters(epsilon: float, gamma: float, delta: float) -> None:
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in the interval (0, 1]")
    if gamma <= 0.0:
        raise ValueError("gamma must be greater than zero")
    if delta <= 0.0:
        raise ValueError("delta must be greater than zero")


def confidence_weights_numpy(
    teacher_scores: np.ndarray,
    eligible_mask: np.ndarray,
    epsilon: float = 0.3,
    gamma: float = 1.0,
    delta: float = 1e-6,
) -> np.ndarray:
    """Convert teacher affinities into per-negative confidence weights.

    Args:
        teacher_scores: Two-dimensional affinity matrix shaped ``[B, C]``.
        eligible_mask: Boolean matrix with ``True`` only for same-domain
            negatives. Positive and cross-domain positions must be ``False``.
        epsilon: Minimum allowed negative weight.
        gamma: Sharpness of the confidence-to-weight mapping.
        delta: Numerical stabilizer added to each row standard deviation.

    Returns:
        A floating matrix shaped ``[B, C]``. Ineligible entries and rows with
        fewer than two eligible negatives contain unit weights.
    """

    _validate_parameters(epsilon, gamma, delta)
    scores = np.asarray(teacher_scores)
    mask = np.asarray(eligible_mask, dtype=bool)
    if scores.ndim != 2:
        raise ValueError("teacher_scores must be a two-dimensional matrix")
    if scores.shape != mask.shape:
        raise ValueError("teacher_scores and eligible_mask must have the same shape")
    if not np.issubdtype(scores.dtype, np.number):
        raise TypeError("teacher_scores must contain numeric values")
    if not np.isfinite(scores).all():
        raise ValueError("teacher_scores must contain only finite values")

    output_dtype = np.result_type(scores.dtype, np.float32)
    weights = np.ones(scores.shape, dtype=output_dtype)
    scores = scores.astype(output_dtype, copy=False)

    for row_index in range(scores.shape[0]):
        eligible_indices = np.flatnonzero(mask[row_index])
        if eligible_indices.size < 2:
            continue
        row_scores = scores[row_index, eligible_indices]
        standardized = (row_scores - row_scores.mean()) / (
            row_scores.std(ddof=0) + delta
        )
        confidence = 1.0 / (1.0 + np.exp(-np.clip(standardized, -60.0, 60.0)))
        weights[row_index, eligible_indices] = epsilon + (1.0 - epsilon) * np.power(
            1.0 - confidence, gamma
        )

    return weights


def confidence_weights_torch(
    teacher_scores: Any,
    eligible_mask: Any,
    epsilon: float = 0.3,
    gamma: float = 1.0,
    delta: float = 1e-6,
) -> Any:
    """Differentiable-training counterpart with stop-gradient teacher weights.

    PyTorch is imported lazily so the reference equations remain usable in
    lightweight environments.
    """

    import torch

    _validate_parameters(epsilon, gamma, delta)
    if not isinstance(teacher_scores, torch.Tensor):
        raise TypeError("teacher_scores must be a torch.Tensor")
    if not isinstance(eligible_mask, torch.Tensor):
        raise TypeError("eligible_mask must be a torch.Tensor")
    if teacher_scores.ndim != 2:
        raise ValueError("teacher_scores must be a two-dimensional matrix")
    if teacher_scores.shape != eligible_mask.shape:
        raise ValueError("teacher_scores and eligible_mask must have the same shape")
    if not torch.is_floating_point(teacher_scores):
        raise TypeError("teacher_scores must use a floating-point dtype")
    if not torch.isfinite(teacher_scores).all().item():
        raise ValueError("teacher_scores must contain only finite values")

    scores = teacher_scores.detach()
    mask = eligible_mask.to(device=scores.device, dtype=torch.bool)
    weights = torch.ones_like(scores)

    for row_index in range(scores.shape[0]):
        row_mask = mask[row_index]
        if int(row_mask.sum().item()) < 2:
            continue
        row_scores = scores[row_index, row_mask]
        standardized = (row_scores - row_scores.mean()) / (
            row_scores.std(unbiased=False) + delta
        )
        confidence = torch.sigmoid(standardized)
        weights[row_index, row_mask] = epsilon + (1.0 - epsilon) * torch.pow(
            1.0 - confidence, gamma
        )

    return weights.detach()
