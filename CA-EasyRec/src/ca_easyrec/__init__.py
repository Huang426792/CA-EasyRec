"""Confidence-Aware EasyRec research components."""

from .losses import weighted_info_nce_numpy, weighted_info_nce_torch
from .weighting import confidence_weights_numpy, confidence_weights_torch

__all__ = [
    "confidence_weights_numpy",
    "confidence_weights_torch",
    "weighted_info_nce_numpy",
    "weighted_info_nce_torch",
]

__version__ = "0.1.0"
