"""A compact LightGCN teacher for source-domain collaborative confidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class LightGCNTrainingConfig:
    """Training parameters for one source-domain teacher."""

    embedding_dim: int = 32
    num_layers: int = 2
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-2
    l2_weight: float = 1e-4
    seed: int = 2026
    device: str = "cpu"


class LightGCN(nn.Module):
    """LightGCN with normalized user-item propagation and BPR training."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _validate_edge_index(self, edge_index: torch.Tensor) -> None:
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, number_of_edges]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must use an integer dtype")
        if edge_index.shape[1] == 0:
            raise ValueError("edge_index must contain at least one interaction")
        users, items = edge_index[0], edge_index[1]
        if torch.any(users < 0).item() or torch.any(users >= self.num_users).item():
            raise ValueError("edge_index contains an out-of-range user ID")
        if torch.any(items < 0).item() or torch.any(items >= self.num_items).item():
            raise ValueError("edge_index contains an out-of-range item ID")

    def normalized_adjacency(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Build ``D^-1/2 A D^-1/2`` for the bipartite interaction graph."""

        self._validate_edge_index(edge_index)
        device = self.user_embedding.weight.device
        edge_index = edge_index.to(device=device, dtype=torch.long)
        user_nodes = edge_index[0]
        item_nodes = edge_index[1] + self.num_users
        rows = torch.cat((user_nodes, item_nodes))
        columns = torch.cat((item_nodes, user_nodes))
        node_count = self.num_users + self.num_items
        degrees = torch.zeros(node_count, device=device)
        degrees.scatter_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        inverse_sqrt_degree = degrees.clamp_min(1.0).pow(-0.5)
        values = inverse_sqrt_degree[rows] * inverse_sqrt_degree[columns]
        adjacency = torch.sparse_coo_tensor(
            torch.stack((rows, columns)),
            values,
            size=(node_count, node_count),
            device=device,
        )
        return adjacency.coalesce()

    def propagate(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return averaged initial and propagated user/item embeddings."""

        adjacency = self.normalized_adjacency(edge_index)
        embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight),
            dim=0,
        )
        layer_embeddings = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return (
            final_embeddings[: self.num_users],
            final_embeddings[self.num_users :],
        )

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        l2_weight: float = 1e-4,
    ) -> torch.Tensor:
        """Calculate BPR loss using propagated scores and raw-embedding L2."""

        if l2_weight < 0.0:
            raise ValueError("l2_weight must be non-negative")
        if not (user_ids.shape == positive_item_ids.shape == negative_item_ids.shape):
            raise ValueError("user and positive/negative item IDs must share a shape")
        device = self.user_embedding.weight.device
        user_ids = user_ids.to(device=device, dtype=torch.long)
        positive_item_ids = positive_item_ids.to(device=device, dtype=torch.long)
        negative_item_ids = negative_item_ids.to(device=device, dtype=torch.long)
        if (
            torch.any(user_ids < 0).item()
            or torch.any(user_ids >= self.num_users).item()
        ):
            raise ValueError("batch contains an out-of-range user ID")
        for item_ids in (positive_item_ids, negative_item_ids):
            if (
                torch.any(item_ids < 0).item()
                or torch.any(item_ids >= self.num_items).item()
            ):
                raise ValueError("batch contains an out-of-range item ID")

        all_users, all_items = self.propagate(edge_index)
        users = all_users[user_ids]
        positives = all_items[positive_item_ids]
        negatives = all_items[negative_item_ids]
        positive_scores = (users * positives).sum(dim=-1)
        negative_scores = (users * negatives).sum(dim=-1)
        ranking_loss = -functional.logsigmoid(positive_scores - negative_scores).mean()
        raw_regularization = (
            self.user_embedding(user_ids).pow(2).sum(dim=-1)
            + self.item_embedding(positive_item_ids).pow(2).sum(dim=-1)
            + self.item_embedding(negative_item_ids).pow(2).sum(dim=-1)
        ).mean()
        return ranking_loss + l2_weight * raw_regularization


def _sample_negative_items(
    user_ids: torch.Tensor,
    positive_sets: Sequence[set[int]],
    num_items: int,
    generator: torch.Generator,
) -> torch.Tensor:
    negatives: list[int] = []
    for user_id in user_ids.tolist():
        positives = positive_sets[user_id]
        if len(positives) >= num_items:
            raise ValueError(
                f"user {user_id} interacted with every item; no BPR negative exists"
            )
        while True:
            candidate = int(
                torch.randint(
                    low=0,
                    high=num_items,
                    size=(1,),
                    generator=generator,
                ).item()
            )
            if candidate not in positives:
                negatives.append(candidate)
                break
    return torch.tensor(negatives, dtype=torch.long)


def train_lightgcn(
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
    config: LightGCNTrainingConfig | None = None,
) -> tuple[LightGCN, list[float]]:
    """Train one deterministic source-domain LightGCN teacher."""

    config = config or LightGCNTrainingConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)
    model._validate_edge_index(edge_index)
    graph_edges = edge_index.to(device=device, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    random_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    positive_sets: list[set[int]] = [set() for _ in range(num_users)]
    for user_id, item_id in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        positive_sets[user_id].add(item_id)

    history: list[float] = []
    edge_count = edge_index.shape[1]
    for _ in range(config.epochs):
        permutation = torch.randperm(edge_count, generator=random_generator)
        epoch_losses: list[float] = []
        for start in range(0, edge_count, config.batch_size):
            positions = permutation[start : start + config.batch_size]
            users = edge_index[0, positions]
            positives = edge_index[1, positions]
            negatives = _sample_negative_items(
                users,
                positive_sets,
                num_items,
                random_generator,
            )
            optimizer.zero_grad()
            loss = model.bpr_loss(
                users,
                positives,
                negatives,
                graph_edges,
                l2_weight=config.l2_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        history.append(sum(epoch_losses) / len(epoch_losses))

    model.eval()
    return model, history
