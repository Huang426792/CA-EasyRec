"""Commands for preparing CA-EasyRec teachers from official EasyRec data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .data import load_easyrec_domain
from .lightgcn import LightGCNTrainingConfig, train_lightgcn
from .teacher_bank import TeacherEmbeddingBank


def train_official_teachers(
    *,
    data_root: str | Path,
    domains: Sequence[str],
    output_path: str | Path,
    embedding_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-2,
    l2_weight: float = 1e-4,
    seed: int = 2026,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train and export one frozen LightGCN teacher per EasyRec source domain."""

    normalized_domains = tuple(dict.fromkeys(domain.strip() for domain in domains))
    if not normalized_domains or any(not domain for domain in normalized_domains):
        raise ValueError("domains must contain at least one non-empty name")

    teacher_bank = TeacherEmbeddingBank()
    histories: dict[str, list[float]] = {}
    for offset, domain_name in enumerate(normalized_domains):
        domain = load_easyrec_domain(data_root, domain_name)
        edge_index = torch.tensor(domain.train_edges, dtype=torch.long)
        model, history = train_lightgcn(
            edge_index=edge_index,
            num_users=domain.num_users,
            num_items=domain.num_items,
            config=LightGCNTrainingConfig(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l2_weight=l2_weight,
                seed=seed + offset,
                device=device,
            ),
        )
        with torch.inference_mode():
            user_embeddings, item_embeddings = model.propagate(edge_index.to(device))
        teacher_bank.add_domain(
            domain_name,
            user_embeddings,
            item_embeddings,
        )
        histories[domain_name] = history

    output_path = Path(output_path)
    teacher_bank.save(output_path)
    history_path = output_path.with_suffix(output_path.suffix + ".history.json")
    history_path.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return histories


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CA-EasyRec LightGCN teachers from EasyRec data.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/easyrec_teachers.pt")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    histories = train_official_teachers(
        data_root=arguments.data_root,
        domains=arguments.domains,
        output_path=arguments.output,
        embedding_dim=arguments.embedding_dim,
        num_layers=arguments.num_layers,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        l2_weight=arguments.l2_weight,
        seed=arguments.seed,
        device=arguments.device,
    )
    summary = {
        domain: {
            "epochs": len(history),
            "final_loss": history[-1],
        }
        for domain, history in histories.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
