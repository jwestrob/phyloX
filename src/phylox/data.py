from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConcatenatedEmbeddings:
    embeddings: np.ndarray
    mask: np.ndarray
    dim_to_partition: np.ndarray
    partition_names: tuple[str, ...]
    partition_weights: np.ndarray

    @property
    def n_taxa(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def d_total(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n_partitions(self) -> int:
        return int(len(self.partition_names))

    def partition_indices(self) -> dict[int, np.ndarray]:
        return partition_indices(self.dim_to_partition)


def concatenate_partition_embeddings(
    partitions: Mapping[str, np.ndarray],
    presence_mask_by_partition: Mapping[str, np.ndarray] | None = None,
    weight_scheme: str = "uniform",
) -> ConcatenatedEmbeddings:
    """
    Concatenate partition embeddings into one matrix with explicit mask.

    Each partition matrix must be shape (n_taxa, d_g). Missing values can be encoded
    either by `presence_mask_by_partition[name]` (row-level) or NaN entries.
    """
    if not partitions:
        raise ValueError("partitions must be non-empty")

    partition_names = tuple(partitions.keys())
    n_taxa: int | None = None
    block_values: list[np.ndarray] = []
    block_masks: list[np.ndarray] = []
    dim_to_partition: list[np.ndarray] = []

    for p_idx, name in enumerate(partition_names):
        z = np.asarray(partitions[name], dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"partition '{name}' must be 2D")
        if n_taxa is None:
            n_taxa = int(z.shape[0])
        if z.shape[0] != n_taxa:
            raise ValueError("all partitions must have same number of taxa")

        row_presence: np.ndarray | None = None
        if presence_mask_by_partition is not None and name in presence_mask_by_partition:
            row_presence = np.asarray(presence_mask_by_partition[name], dtype=bool)
            if row_presence.shape != (n_taxa,):
                raise ValueError(
                    f"presence mask for partition '{name}' must have shape (n_taxa,)"
                )

        finite = np.isfinite(z)
        if row_presence is None:
            part_mask = finite
        else:
            part_mask = row_presence[:, None] & finite

        block_values.append(np.where(part_mask, z, 0.0))
        block_masks.append(part_mask)
        dim_to_partition.append(np.full(z.shape[1], p_idx, dtype=np.int64))

    emb = np.concatenate(block_values, axis=1)
    msk = np.concatenate(block_masks, axis=1)
    dim_map = np.concatenate(dim_to_partition, axis=0)
    weights = compute_partition_weights(dim_map, scheme=weight_scheme)
    return ConcatenatedEmbeddings(
        embeddings=emb,
        mask=msk,
        dim_to_partition=dim_map,
        partition_names=partition_names,
        partition_weights=weights,
    )


def split_concatenated_embeddings(
    data: ConcatenatedEmbeddings,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    indices = partition_indices(data.dim_to_partition)
    for p_idx, name in enumerate(data.partition_names):
        idx = indices[p_idx]
        out[name] = (data.embeddings[:, idx], data.mask[:, idx])
    return out


def partition_indices(dim_to_partition: np.ndarray) -> dict[int, np.ndarray]:
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    if dim_to_partition.ndim != 1:
        raise ValueError("dim_to_partition must be a 1D array")
    if dim_to_partition.size == 0:
        return {}
    if np.any(dim_to_partition < 0):
        raise ValueError("dim_to_partition must be non-negative")

    out: dict[int, np.ndarray] = {}
    n_partitions = int(dim_to_partition.max()) + 1
    for p in range(n_partitions):
        out[p] = np.flatnonzero(dim_to_partition == p)
    return out


def compute_partition_weights(
    dim_to_partition: np.ndarray,
    scheme: str = "uniform",
) -> np.ndarray:
    """
    Build partition-level weights.

    Schemes:
    - uniform: each partition has weight 1
    - inverse_dims: each partition has weight 1 / d_g
    """
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    if dim_to_partition.ndim != 1:
        raise ValueError("dim_to_partition must be 1D")
    if dim_to_partition.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any(dim_to_partition < 0):
        raise ValueError("dim_to_partition must be non-negative")

    n_partitions = int(dim_to_partition.max()) + 1
    if scheme == "uniform":
        return np.ones(n_partitions, dtype=np.float64)
    if scheme == "inverse_dims":
        weights = np.zeros(n_partitions, dtype=np.float64)
        for p in range(n_partitions):
            d_p = np.count_nonzero(dim_to_partition == p)
            if d_p <= 0:
                raise ValueError(f"partition {p} has no dimensions")
            weights[p] = 1.0 / float(d_p)
        return weights
    raise ValueError(f"unsupported weight scheme '{scheme}'")


def partition_presence_from_mask(mask: np.ndarray, dim_to_partition: np.ndarray) -> np.ndarray:
    """
    Returns bool matrix shape (n_taxa, n_partitions) indicating taxa-partition presence.
    A taxon is present for partition g if it has at least one observed dim in g.
    """
    m = np.asarray(mask, dtype=bool)
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    if dim_to_partition.shape != (m.shape[1],):
        raise ValueError("dim_to_partition length mismatch")
    n_partitions = int(dim_to_partition.max()) + 1 if dim_to_partition.size > 0 else 0
    out = np.zeros((m.shape[0], n_partitions), dtype=bool)
    for p in range(n_partitions):
        idx = np.flatnonzero(dim_to_partition == p)
        if idx.size == 0:
            continue
        out[:, p] = np.any(m[:, idx], axis=1)
    return out


def partition_weights_from_mapping(
    partition_names: Sequence[str],
    mapping: Mapping[str, float],
    default: float = 1.0,
) -> np.ndarray:
    weights = np.full(len(partition_names), float(default), dtype=np.float64)
    for i, name in enumerate(partition_names):
        if name in mapping:
            weights[i] = float(mapping[name])
    return weights
