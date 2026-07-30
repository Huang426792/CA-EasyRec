"""Deterministic end-to-end CA-EasyRec smoke experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .data import ResearchDomain
from .lightgcn import (
    LightGCNTrainingConfig,
    train_lightgcn,
)
from .metrics import recall_ndcg_at_k
from .teacher_bank import TeacherEmbeddingBank
from .training import TextTrainingConfig, train_profile_encoder


def _theme_domain(
    name: str,
    themes: Sequence[str],
    user_template: str,
    item_template: str,
) -> ResearchDomain:
    user_profiles = tuple(user_template.format(theme=theme) for theme in themes)
    item_profiles: list[str] = []
    train_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []
    for user_id, theme in enumerate(themes):
        base_item = len(item_profiles)
        item_profiles.extend(
            item_template.format(theme=theme, variant=variant)
            for variant in ("beginner", "classic", "new")
        )
        train_pairs.extend(((user_id, base_item), (user_id, base_item + 1)))
        test_pairs.append((user_id, base_item + 2))
    return ResearchDomain(
        name=name,
        raw_user_ids=tuple(f"{name}-u{index}" for index in range(len(themes))),
        raw_item_ids=tuple(f"{name}-i{index}" for index in range(len(item_profiles))),
        user_profiles=user_profiles,
        item_profiles=tuple(item_profiles),
        train_edges=np.asarray(train_pairs, dtype=np.int64).T,
        validation_edges=np.empty((2, 0), dtype=np.int64),
        test_edges=np.asarray(test_pairs, dtype=np.int64).T,
    )


def make_toy_domains() -> dict[str, ResearchDomain]:
    """Create two small domains with transparent text-interest structure."""

    books = _theme_domain(
        name="books",
        themes=("space", "history", "cooking", "mystery"),
        user_template="This reader enjoys {theme} books and related stories.",
        item_template="A {variant} {theme} book for readers.",
    )
    games = _theme_domain(
        name="games",
        themes=("strategy", "sports", "puzzle", "adventure"),
        user_template="This player enjoys {theme} games and challenges.",
        item_template="A {variant} {theme} game for players.",
    )
    return {"books": books, "games": games}


def _edge_sets(
    edges: np.ndarray,
    num_users: int,
) -> list[set[int]]:
    result: list[set[int]] = [set() for _ in range(num_users)]
    for user_id, item_id in edges.T.tolist():
        result[user_id].add(item_id)
    return result


def _evaluate_domains(
    encoder: torch.nn.Module,
    domains: dict[str, ResearchDomain],
    k: int,
) -> dict[str, object]:
    domain_metrics: dict[str, dict[str, float | int]] = {}
    weighted_recall = 0.0
    weighted_ndcg = 0.0
    total_users = 0
    encoder.eval()
    with torch.inference_mode():
        for domain_name in sorted(domains):
            domain = domains[domain_name]
            user_embeddings = encoder(domain.user_profiles)
            item_embeddings = encoder(domain.item_profiles)
            score_matrix = (
                (user_embeddings @ item_embeddings.T).detach().to(device="cpu").numpy()
            )
            truth = _edge_sets(domain.test_edges, domain.num_users)
            seen = _edge_sets(domain.train_edges, domain.num_users)
            metrics = recall_ndcg_at_k(score_matrix, truth, seen, k=k)
            domain_metrics[domain_name] = metrics
            users = int(metrics["users"])
            total_users += users
            weighted_recall += float(metrics[f"recall@{k}"]) * users
            weighted_ndcg += float(metrics[f"ndcg@{k}"]) * users
    if total_users == 0:
        raise ValueError("the toy evaluation contains no test users")
    return {
        f"recall@{k}": weighted_recall / total_users,
        f"ndcg@{k}": weighted_ndcg / total_users,
        "users": total_users,
        "domains": domain_metrics,
    }


def run_demo(
    *,
    output_directory: str | Path,
    seed: int = 2026,
    teacher_epochs: int = 20,
    text_epochs: int = 20,
    embedding_dim: int = 16,
    k: int = 3,
    device: str = "cpu",
) -> dict[str, object]:
    """Train teachers and profile encoder, evaluate, and save artifacts."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    domains = make_toy_domains()

    teacher_bank = TeacherEmbeddingBank()
    teacher_histories: dict[str, list[float]] = {}
    for domain_name in sorted(domains):
        domain = domains[domain_name]
        teacher, history = train_lightgcn(
            edge_index=torch.tensor(domain.train_edges, dtype=torch.long),
            num_users=domain.num_users,
            num_items=domain.num_items,
            config=LightGCNTrainingConfig(
                embedding_dim=embedding_dim,
                num_layers=1,
                epochs=teacher_epochs,
                batch_size=8,
                learning_rate=5e-2,
                l2_weight=1e-4,
                seed=seed,
                device=device,
            ),
        )
        with torch.inference_mode():
            users, items = teacher.propagate(
                torch.tensor(
                    domain.train_edges,
                    dtype=torch.long,
                    device=device,
                )
            )
        teacher_bank.add_domain(domain_name, users, items)
        teacher_histories[domain_name] = history
    teacher_bank.save(output_path / "teacher.pt")

    encoder, text_history = train_profile_encoder(
        domains,
        teacher_bank,
        TextTrainingConfig(
            embedding_dim=embedding_dim,
            vocabulary_size=1024,
            epochs=text_epochs,
            batch_size=8,
            learning_rate=3e-2,
            temperature=0.1,
            epsilon=0.3,
            gamma=1.0,
            seed=seed,
            device=device,
        ),
    )
    torch.save(
        {
            "format_version": 1,
            "model": "HashingProfileEncoder",
            "vocabulary_size": encoder.vocabulary_size,
            "embedding_dim": encoder.embedding_dim,
            "state_dict": {
                key: value.detach().cpu() for key, value in encoder.state_dict().items()
            },
        },
        output_path / "text_model.pt",
    )
    metrics = _evaluate_domains(encoder, domains, k=k)
    metrics["metadata"] = {
        "result_type": "toy_smoke_test",
        "seed": seed,
        "teacher_epochs": teacher_epochs,
        "text_epochs": text_epochs,
        "embedding_dim": embedding_dim,
        "teacher_final_loss": {
            domain: history[-1] for domain, history in teacher_histories.items()
        },
        "text_final_loss": text_history[-1],
        "note": (
            "These toy metrics verify the code path and are not Sports, "
            "Steam, or Yelp paper results."
        ),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic CA-EasyRec toy experiment.",
    )
    parser.add_argument("--output", default="artifacts/demo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--text-epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    metrics = run_demo(
        output_directory=arguments.output,
        seed=arguments.seed,
        teacher_epochs=arguments.teacher_epochs,
        text_epochs=arguments.text_epochs,
        embedding_dim=arguments.embedding_dim,
        k=arguments.k,
        device=arguments.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
