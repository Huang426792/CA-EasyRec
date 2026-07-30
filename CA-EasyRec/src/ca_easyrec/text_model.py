"""Lightweight text encoder used by the reproducible toy experiment."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def hash_profile_tokens(text: str, vocabulary_size: int) -> list[int]:
    """Map profile tokens to stable IDs without Python's randomized hash."""

    if vocabulary_size < 3:
        raise ValueError("vocabulary_size must be at least 3")
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        tokens = ["<empty>"]
    token_ids: list[int] = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        token_ids.append(
            int.from_bytes(digest[:8], byteorder="big") % (vocabulary_size - 1) + 1
        )
    return token_ids


class HashingProfileEncoder(nn.Module):
    """A small shared profile encoder with deterministic tokenization.

    This model keeps the toy experiment quick and offline. Full EasyRec
    experiments use the upstream RoBERTa encoder through the integration guide.
    """

    def __init__(self, vocabulary_size: int = 4096, embedding_dim: int = 64) -> None:
        super().__init__()
        if vocabulary_size < 3:
            raise ValueError("vocabulary_size must be at least 3")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, profiles: Sequence[str]) -> torch.Tensor:
        if not profiles:
            raise ValueError("profiles must contain at least one string")
        token_lists = [
            hash_profile_tokens(profile, self.vocabulary_size) for profile in profiles
        ]
        max_length = max(len(tokens) for tokens in token_lists)
        device = self.embedding.weight.device
        token_ids = torch.zeros(
            (len(token_lists), max_length),
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for row_index, tokens in enumerate(token_lists):
            length = len(tokens)
            token_ids[row_index, :length] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )
            mask[row_index, :length] = True
        token_embeddings = self.embedding(token_ids)
        pooled = (token_embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return functional.normalize(pooled, p=2, dim=-1)
