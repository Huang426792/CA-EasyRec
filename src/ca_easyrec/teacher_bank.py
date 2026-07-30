"""Domain-separated frozen LightGCN embedding storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch


class TeacherEmbeddingBank:
    """Store and score frozen user/item embeddings without cross-domain leakage."""

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._domains: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def domain_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._domains))

    def add_domain(
        self,
        name: str,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("domain name must be a non-empty string")
        if not isinstance(user_embeddings, torch.Tensor) or not isinstance(
            item_embeddings,
            torch.Tensor,
        ):
            raise TypeError("user and item embeddings must be torch.Tensor values")
        if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ValueError("user and item embeddings must be two-dimensional")
        if user_embeddings.shape[1] != item_embeddings.shape[1]:
            raise ValueError("user and item embedding dimension must match")
        if user_embeddings.shape[0] == 0 or item_embeddings.shape[0] == 0:
            raise ValueError("a teacher domain must contain users and items")
        if (
            not torch.isfinite(user_embeddings).all().item()
            or not torch.isfinite(item_embeddings).all().item()
        ):
            raise ValueError("teacher embeddings must contain only finite values")

        users = user_embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
        items = item_embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
        users.requires_grad_(False)
        items.requires_grad_(False)
        self._domains[name] = (users, items)

    def score_matrix(
        self,
        user_domains: Sequence[str],
        user_ids: torch.Tensor,
        item_domains: Sequence[str],
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a user-by-item grid, returning scores and a validity mask."""

        if user_ids.ndim != 1 or item_ids.ndim != 1:
            raise ValueError("user_ids and item_ids must be one-dimensional")
        if len(user_domains) != user_ids.shape[0]:
            raise ValueError("user_domains must align with user_ids")
        if len(item_domains) != item_ids.shape[0]:
            raise ValueError("item_domains must align with item_ids")
        if user_ids.dtype not in (torch.int32, torch.int64) or item_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("user_ids and item_ids must use integer dtypes")

        output_device = user_ids.device
        output_dtype = torch.float32
        scores = torch.zeros(
            (user_ids.shape[0], item_ids.shape[0]),
            dtype=output_dtype,
            device=output_device,
        )
        valid = torch.zeros_like(scores, dtype=torch.bool)
        user_ids_cpu = user_ids.detach().to(device="cpu", dtype=torch.long)
        item_ids_cpu = item_ids.detach().to(device="cpu", dtype=torch.long)

        for domain_name, (domain_users, domain_items) in self._domains.items():
            user_positions = [
                position
                for position, name in enumerate(user_domains)
                if name == domain_name
            ]
            item_positions = [
                position
                for position, name in enumerate(item_domains)
                if name == domain_name
            ]
            if not user_positions or not item_positions:
                continue
            selected_user_ids = user_ids_cpu[user_positions]
            selected_item_ids = item_ids_cpu[item_positions]
            if (
                torch.any(selected_user_ids < 0).item()
                or torch.any(selected_user_ids >= domain_users.shape[0]).item()
            ):
                raise ValueError(f"domain {domain_name!r} contains an invalid user ID")
            if (
                torch.any(selected_item_ids < 0).item()
                or torch.any(selected_item_ids >= domain_items.shape[0]).item()
            ):
                raise ValueError(f"domain {domain_name!r} contains an invalid item ID")
            block = domain_users[selected_user_ids] @ domain_items[selected_item_ids].T
            user_position_tensor = torch.tensor(
                user_positions,
                dtype=torch.long,
                device=output_device,
            )
            item_position_tensor = torch.tensor(
                item_positions,
                dtype=torch.long,
                device=output_device,
            )
            scores[
                user_position_tensor[:, None],
                item_position_tensor[None, :],
            ] = block.to(output_device)
            valid[
                user_position_tensor[:, None],
                item_position_tensor[None, :],
            ] = True

        return scores.detach(), valid

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "domains": {
                name: {
                    "user_embeddings": users,
                    "item_embeddings": items,
                }
                for name, (users, items) in self._domains.items()
            },
        }
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> TeacherEmbeddingBank:
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(f"teacher bank does not exist: {input_path}")
        try:
            payload = torch.load(
                input_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(input_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("teacher bank payload must be a dictionary")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported teacher bank format version")
        domains = payload.get("domains")
        if not isinstance(domains, dict):
            raise TypeError("teacher bank payload is missing domains")

        bank = cls()
        for name, embeddings in domains.items():
            if not isinstance(embeddings, dict):
                raise TypeError(f"invalid teacher payload for domain {name!r}")
            try:
                users = embeddings["user_embeddings"]
                items = embeddings["item_embeddings"]
            except KeyError as error:
                raise ValueError(
                    f"teacher payload for domain {name!r} is incomplete"
                ) from error
            bank.add_domain(name, users, items)
        return bank
