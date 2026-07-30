"""Data contracts for standalone and official EasyRec experiments."""

from __future__ import annotations

import csv
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _empty_edges() -> np.ndarray:
    return np.empty((2, 0), dtype=np.int64)


@dataclass(frozen=True)
class ResearchDomain:
    """One domain with contiguous IDs, profiles, and interaction splits."""

    name: str
    raw_user_ids: tuple[str, ...]
    raw_item_ids: tuple[str, ...]
    user_profiles: tuple[str, ...]
    item_profiles: tuple[str, ...]
    train_edges: np.ndarray
    validation_edges: np.ndarray
    test_edges: np.ndarray

    @property
    def num_users(self) -> int:
        return len(self.user_profiles)

    @property
    def num_items(self) -> int:
        return len(self.item_profiles)


@dataclass(frozen=True)
class ResearchDataset:
    """A named collection of source or target recommendation domains."""

    domains: dict[str, ResearchDomain]


def _read_profile_file(
    path: Path,
    identifier_field: str,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"profile file does not exist: {path}")
    profiles: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                domain = str(record["domain"])
                identifier = str(record[identifier_field])
                profile = str(record["profile"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid profile record at {path}:{line_number}"
                ) from error
            if not domain or not identifier or not profile.strip():
                raise ValueError(f"empty profile field at {path}:{line_number}")
            domain_profiles = profiles.setdefault(domain, {})
            if identifier in domain_profiles:
                raise ValueError(
                    f"duplicate {identifier_field} {identifier!r} in domain {domain!r}"
                )
            domain_profiles[identifier] = profile
    return profiles


def _edges_from_pairs(pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    pair_list = list(pairs)
    if not pair_list:
        return _empty_edges()
    return np.asarray(pair_list, dtype=np.int64).T


def load_compact_dataset(path: str | Path) -> ResearchDataset:
    """Load the repository's CSV/JSONL data format.

    Required files:

    - ``interactions.csv``: ``domain,user_id,item_id,split``
    - ``user_profiles.jsonl``: ``domain,user_id,profile``
    - ``item_profiles.jsonl``: ``domain,item_id,profile``
    """

    root = Path(path)
    user_profiles = _read_profile_file(
        root / "user_profiles.jsonl",
        "user_id",
    )
    item_profiles = _read_profile_file(
        root / "item_profiles.jsonl",
        "item_id",
    )
    interaction_path = root / "interactions.csv"
    if not interaction_path.is_file():
        raise FileNotFoundError(f"interaction file does not exist: {interaction_path}")

    records: dict[str, list[tuple[str, str, str]]] = {}
    with interaction_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_fields = {"domain", "user_id", "item_id", "split"}
        if reader.fieldnames is None or not expected_fields.issubset(reader.fieldnames):
            raise ValueError(
                "interactions.csv must contain domain,user_id,item_id,split"
            )
        for line_number, record in enumerate(reader, start=2):
            domain = str(record["domain"]).strip()
            user_id = str(record["user_id"]).strip()
            item_id = str(record["item_id"]).strip()
            split = str(record["split"]).strip().lower()
            if split == "val":
                split = "validation"
            if split not in {"train", "validation", "test"}:
                raise ValueError(
                    f"invalid split {split!r} at {interaction_path}:{line_number}"
                )
            if domain not in user_profiles or user_id not in user_profiles[domain]:
                raise ValueError(f"missing user profile for {domain}:{user_id}")
            if domain not in item_profiles or item_id not in item_profiles[domain]:
                raise ValueError(f"missing item profile for {domain}:{item_id}")
            records.setdefault(domain, []).append((user_id, item_id, split))

    domains: dict[str, ResearchDomain] = {}
    for domain_name in sorted(records):
        raw_user_ids = tuple(sorted(user_profiles[domain_name]))
        raw_item_ids = tuple(sorted(item_profiles[domain_name]))
        user_to_index = {
            identifier: index for index, identifier in enumerate(raw_user_ids)
        }
        item_to_index = {
            identifier: index for index, identifier in enumerate(raw_item_ids)
        }
        split_pairs: dict[str, list[tuple[int, int]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for user_id, item_id, split in records[domain_name]:
            split_pairs[split].append((user_to_index[user_id], item_to_index[item_id]))
        domains[domain_name] = ResearchDomain(
            name=domain_name,
            raw_user_ids=raw_user_ids,
            raw_item_ids=raw_item_ids,
            user_profiles=tuple(
                user_profiles[domain_name][identifier] for identifier in raw_user_ids
            ),
            item_profiles=tuple(
                item_profiles[domain_name][identifier] for identifier in raw_item_ids
            ),
            train_edges=_edges_from_pairs(split_pairs["train"]),
            validation_edges=_edges_from_pairs(split_pairs["validation"]),
            test_edges=_edges_from_pairs(split_pairs["test"]),
        )
    if not domains:
        raise ValueError("the compact dataset contains no interactions")
    return ResearchDataset(domains=domains)


def _load_pickle_matrix(path: Path, required: bool) -> object | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"EasyRec matrix does not exist: {path}")
        return None
    with path.open("rb") as source:
        return pickle.load(source)


def _matrix_to_edges(
    matrix: object | None, expected_shape: tuple[int, int]
) -> np.ndarray:
    if matrix is None:
        return _empty_edges()
    if not hasattr(matrix, "shape") or tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"interaction matrix shape must be {expected_shape}, got "
            f"{getattr(matrix, 'shape', None)}"
        )
    coo_matrix = matrix.tocoo()
    if coo_matrix.nnz == 0:
        return _empty_edges()
    order = np.lexsort((coo_matrix.col, coo_matrix.row))
    return np.stack(
        (
            np.asarray(coo_matrix.row, dtype=np.int64)[order],
            np.asarray(coo_matrix.col, dtype=np.int64)[order],
        )
    )


def _read_easyrec_profiles(
    path: Path,
    identifier_field: str,
    expected_count: int,
) -> tuple[str, ...]:
    profiles: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identifier = int(record[identifier_field])
                profile = str(record["profile"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid EasyRec profile at {path}:{line_number}"
                ) from error
            if identifier in profiles:
                raise ValueError(f"duplicate {identifier_field} {identifier} in {path}")
            profiles[identifier] = profile
    expected_ids = set(range(expected_count))
    if set(profiles) != expected_ids:
        missing = sorted(expected_ids - set(profiles))
        extra = sorted(set(profiles) - expected_ids)
        raise ValueError(
            f"{path} IDs must cover 0..{expected_count - 1}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return tuple(profiles[index] for index in range(expected_count))


def load_easyrec_domain(
    data_root: str | Path,
    domain: str,
) -> ResearchDomain:
    """Load one domain in the official HKUDS/EasyRec data layout."""

    domain_root = Path(data_root) / domain
    train_matrix = _load_pickle_matrix(domain_root / "trn_mat.pkl", required=True)
    if not hasattr(train_matrix, "shape") or len(train_matrix.shape) != 2:
        raise ValueError("trn_mat.pkl must contain a two-dimensional sparse matrix")
    num_users, num_items = map(int, train_matrix.shape)
    expected_shape = (num_users, num_items)
    validation_matrix = _load_pickle_matrix(
        domain_root / "val_mat.pkl",
        required=False,
    )
    test_matrix = _load_pickle_matrix(
        domain_root / "tst_mat.pkl",
        required=False,
    )
    return ResearchDomain(
        name=domain,
        raw_user_ids=tuple(str(index) for index in range(num_users)),
        raw_item_ids=tuple(str(index) for index in range(num_items)),
        user_profiles=_read_easyrec_profiles(
            domain_root / "user_profile.json",
            "user_id",
            num_users,
        ),
        item_profiles=_read_easyrec_profiles(
            domain_root / "item_profile.json",
            "item_id",
            num_items,
        ),
        train_edges=_matrix_to_edges(train_matrix, expected_shape),
        validation_edges=_matrix_to_edges(validation_matrix, expected_shape),
        test_edges=_matrix_to_edges(test_matrix, expected_shape),
    )
