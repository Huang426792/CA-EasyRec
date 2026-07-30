"""Training utilities shared by the demo and EasyRec adapter."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .data import ResearchDomain
from .losses import weighted_info_nce_torch
from .teacher_bank import TeacherEmbeddingBank
from .text_model import HashingProfileEncoder
from .weighting import confidence_weights_torch


@dataclass(frozen=True)
class CABatchOutput:
    """Inspectable outputs of one confidence-aware contrastive batch."""

    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    teacher_scores: torch.Tensor
    eligible_mask: torch.Tensor


@dataclass(frozen=True)
class TextTrainingConfig:
    """Training parameters for the standalone profile encoder."""

    embedding_dim: int = 64
    vocabulary_size: int = 4096
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 3e-2
    temperature: float = 0.05
    epsilon: float = 0.3
    gamma: float = 1.0
    seed: int = 2026
    device: str = "cpu"


def _validate_embedding_batch(
    name: str,
    embeddings: torch.Tensor,
    batch_size: int,
    embedding_dim: int | None,
) -> int:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if embeddings.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if embeddings.shape[0] != batch_size:
        raise ValueError(f"{name} must contain {batch_size} rows")
    if embedding_dim is not None and embeddings.shape[1] != embedding_dim:
        raise ValueError("all text embedding dimensions must match")
    if not torch.is_floating_point(embeddings):
        raise TypeError(f"{name} must use a floating-point dtype")
    return embeddings.shape[1]


def ca_easyrec_batch_loss(
    *,
    user_embeddings: torch.Tensor,
    positive_item_embeddings: torch.Tensor,
    negative_item_embeddings: torch.Tensor,
    user_domains: Sequence[str],
    user_ids: torch.Tensor,
    positive_item_domains: Sequence[str],
    positive_item_ids: torch.Tensor,
    negative_item_domains: Sequence[str],
    negative_item_ids: torch.Tensor,
    teacher_bank: TeacherEmbeddingBank,
    temperature: float = 0.05,
    epsilon: float = 0.3,
    gamma: float = 1.0,
) -> CABatchOutput:
    """Calculate the CA-EasyRec loss for an EasyRec-style profile batch.

    Candidate ordering is ``[positive batch, explicitly sampled negative
    batch]``. The positive label for row ``r`` is therefore column ``r``.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if user_ids.ndim != 1:
        raise ValueError("user_ids must be one-dimensional")
    batch_size = user_ids.shape[0]
    if batch_size == 0:
        raise ValueError("a contrastive batch cannot be empty")
    embedding_dim = _validate_embedding_batch(
        "user_embeddings",
        user_embeddings,
        batch_size,
        None,
    )
    _validate_embedding_batch(
        "positive_item_embeddings",
        positive_item_embeddings,
        batch_size,
        embedding_dim,
    )
    _validate_embedding_batch(
        "negative_item_embeddings",
        negative_item_embeddings,
        batch_size,
        embedding_dim,
    )
    metadata_lengths = (
        len(user_domains),
        positive_item_ids.shape[0],
        len(positive_item_domains),
        negative_item_ids.shape[0],
        len(negative_item_domains),
    )
    if any(length != batch_size for length in metadata_lengths):
        raise ValueError("all IDs and domain lists must align with the batch")

    device = user_embeddings.device
    normalized_users = functional.normalize(user_embeddings, p=2, dim=-1)
    candidates = torch.cat(
        (positive_item_embeddings, negative_item_embeddings),
        dim=0,
    )
    normalized_candidates = functional.normalize(candidates, p=2, dim=-1)
    logits = normalized_users @ normalized_candidates.T / temperature
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    candidate_domains = list(positive_item_domains) + list(negative_item_domains)
    candidate_ids = torch.cat(
        (
            positive_item_ids.to(device=device, dtype=torch.long),
            negative_item_ids.to(device=device, dtype=torch.long),
        )
    )
    teacher_scores, valid_teacher_scores = teacher_bank.score_matrix(
        user_domains=list(user_domains),
        user_ids=user_ids.to(device=device, dtype=torch.long),
        item_domains=candidate_domains,
        item_ids=candidate_ids,
    )
    eligible_mask = valid_teacher_scores.clone()
    eligible_mask[labels, labels] = False
    weights = confidence_weights_torch(
        teacher_scores,
        eligible_mask,
        epsilon=epsilon,
        gamma=gamma,
    )
    loss = weighted_info_nce_torch(logits, labels, weights)
    return CABatchOutput(
        loss=loss,
        logits=logits,
        labels=labels,
        weights=weights,
        teacher_scores=teacher_scores,
        eligible_mask=eligible_mask,
    )


def _positive_sets(domain: ResearchDomain) -> list[set[int]]:
    positives: list[set[int]] = [set() for _ in range(domain.num_users)]
    for user_id, item_id in domain.train_edges.T.tolist():
        positives[user_id].add(item_id)
    return positives


def _negative_item(
    domain: ResearchDomain,
    user_id: int,
    positives: list[set[int]],
    random_generator: random.Random,
) -> int:
    if len(positives[user_id]) >= domain.num_items:
        raise ValueError(
            f"{domain.name} user {user_id} has no unobserved negative item"
        )
    while True:
        candidate = random_generator.randrange(domain.num_items)
        if candidate not in positives[user_id]:
            return candidate


def train_profile_encoder(
    domains: dict[str, ResearchDomain],
    teacher_bank: TeacherEmbeddingBank,
    config: TextTrainingConfig | None = None,
) -> tuple[HashingProfileEncoder, list[float]]:
    """Train the standalone shared profile encoder with CA-EasyRec loss."""

    config = config or TextTrainingConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not domains:
        raise ValueError("at least one training domain is required")

    random_generator = random.Random(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    encoder = HashingProfileEncoder(
        vocabulary_size=config.vocabulary_size,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=config.learning_rate,
    )
    examples: list[tuple[str, int, int]] = []
    positives_by_domain: dict[str, list[set[int]]] = {}
    for domain_name in sorted(domains):
        domain = domains[domain_name]
        if domain.train_edges.shape[1] == 0:
            raise ValueError(f"domain {domain_name!r} has no training interactions")
        positives_by_domain[domain_name] = _positive_sets(domain)
        examples.extend(
            (domain_name, int(user_id), int(item_id))
            for user_id, item_id in domain.train_edges.T.tolist()
        )

    history: list[float] = []
    encoder.train()
    for _ in range(config.epochs):
        random_generator.shuffle(examples)
        batch_losses: list[float] = []
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            user_domains = [domain_name for domain_name, _, _ in batch]
            user_ids_list = [user_id for _, user_id, _ in batch]
            positive_ids_list = [item_id for _, _, item_id in batch]
            negative_ids_list = [
                _negative_item(
                    domains[domain_name],
                    user_id,
                    positives_by_domain[domain_name],
                    random_generator,
                )
                for domain_name, user_id, _ in batch
            ]
            user_profiles = [
                domains[domain_name].user_profiles[user_id]
                for domain_name, user_id, _ in batch
            ]
            positive_profiles = [
                domains[domain_name].item_profiles[item_id]
                for domain_name, _, item_id in batch
            ]
            negative_profiles = [
                domains[domain_name].item_profiles[item_id]
                for (domain_name, _, _), item_id in zip(batch, negative_ids_list)
            ]
            optimizer.zero_grad()
            output = ca_easyrec_batch_loss(
                user_embeddings=encoder(user_profiles),
                positive_item_embeddings=encoder(positive_profiles),
                negative_item_embeddings=encoder(negative_profiles),
                user_domains=user_domains,
                user_ids=torch.tensor(user_ids_list, device=device),
                positive_item_domains=user_domains,
                positive_item_ids=torch.tensor(positive_ids_list, device=device),
                negative_item_domains=user_domains,
                negative_item_ids=torch.tensor(negative_ids_list, device=device),
                teacher_bank=teacher_bank,
                temperature=config.temperature,
                epsilon=config.epsilon,
                gamma=config.gamma,
            )
            output.loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(output.loss.detach().cpu().item()))
        history.append(sum(batch_losses) / len(batch_losses))
    encoder.eval()
    return encoder, history
