"""All-rank recommendation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Set as AbstractSet

import numpy as np


def _validate_item_ids(
    user_index: int,
    item_ids: AbstractSet[int],
    num_items: int,
) -> None:
    for item_id in item_ids:
        if not isinstance(item_id, (int, np.integer)):
            raise TypeError(f"user {user_index} contains a non-integer item ID")
        if item_id < 0 or item_id >= num_items:
            raise ValueError(
                f"user {user_index} contains out-of-range item ID {item_id}"
            )


def recall_ndcg_at_k(
    scores: np.ndarray,
    truth: Sequence[AbstractSet[int]],
    seen: Sequence[AbstractSet[int]],
    k: int,
) -> dict[str, float | int]:
    """Calculate macro Recall@K and NDCG@K after masking seen items."""

    score_matrix = np.asarray(scores)
    if score_matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    if not np.isfinite(score_matrix).all():
        raise ValueError("scores must contain only finite values")
    if len(truth) != score_matrix.shape[0] or len(seen) != score_matrix.shape[0]:
        raise ValueError("truth and seen must contain one set per score row")
    if k <= 0 or k > score_matrix.shape[1]:
        raise ValueError("k must be between 1 and the number of items")

    working_scores = score_matrix.astype(np.float64, copy=True)
    num_items = score_matrix.shape[1]
    for user_index, (truth_items, seen_items) in enumerate(zip(truth, seen)):
        _validate_item_ids(user_index, truth_items, num_items)
        _validate_item_ids(user_index, seen_items, num_items)
        if seen_items:
            working_scores[user_index, list(seen_items)] = -np.inf

    recalls: list[float] = []
    ndcgs: list[float] = []
    for user_index, truth_items in enumerate(truth):
        if not truth_items:
            continue
        ranked_items = np.argsort(
            -working_scores[user_index],
            kind="stable",
        )[:k]
        hits = [1.0 if int(item_id) in truth_items else 0.0 for item_id in ranked_items]
        recalls.append(sum(hits) / len(truth_items))
        discounted_gain = sum(
            hit / np.log2(rank + 2.0) for rank, hit in enumerate(hits)
        )
        ideal_hit_count = min(len(truth_items), k)
        ideal_gain = sum(1.0 / np.log2(rank + 2.0) for rank in range(ideal_hit_count))
        ndcgs.append(discounted_gain / ideal_gain)

    evaluated_users = len(recalls)
    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users": evaluated_users,
    }
