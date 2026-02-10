from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from .optimize import OUModelParameters
from .tree import PhyloTree


@dataclass(frozen=True)
class TorchScoreBundle:
    score_fn: callable
    batch_score_fn: callable
    device: str


def torch_available(require_cuda: bool = False) -> bool:
    if torch is None:
        return False
    if require_cuda:
        return bool(torch.cuda.is_available())
    return True


def resolve_torch_device(preferred: str = "cuda") -> str:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    pref = preferred.lower()
    if pref in {"auto", "best"}:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def build_torch_score_bundle(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    device: str = "cuda",
    dtype: str = "float32",
) -> TorchScoreBundle:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")

    dev = resolve_torch_device(device)
    if dev == "mps":
        # Allow unsupported ops to fall back to CPU on Apple Silicon.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch_dtype = _resolve_torch_dtype_for_device(dtype=dtype, device=dev)

    prep = _prepare_static_tensors(
        embeddings=embeddings,
        mask=mask,
        dim_to_partition=dim_to_partition,
        params=params,
        device=dev,
        dtype=torch_dtype,
    )

    def score_fn(tree: PhyloTree) -> float:
        return _ou_log_likelihood_torch_prepared(
            tree=tree,
            prepared=prep,
            alpha=np.asarray(params.alpha_by_partition, dtype=np.float64),
            sigma2=np.asarray(params.sigma2_by_partition, dtype=np.float64),
            gamma_rates_by_partition=params.gamma_rates_by_partition,
        )

    def batch_score_fn(trees: list[PhyloTree]) -> np.ndarray:
        out = np.empty(len(trees), dtype=np.float64)
        for i, t in enumerate(trees):
            out[i] = score_fn(t)
        return out

    return TorchScoreBundle(score_fn=score_fn, batch_score_fn=batch_score_fn, device=dev)


def ou_log_likelihood_torch(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    alpha_by_partition: Sequence[float],
    sigma2_by_partition: Sequence[float],
    partition_weights: Sequence[float] | None = None,
    coverage_scale: np.ndarray | None = None,
    rate_by_dim: np.ndarray | None = None,
    precision_weights: np.ndarray | None = None,
    gamma_rates_by_partition: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    root: int | None = None,
    device: str = "cuda",
    dtype: str = "float32",
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    params = OUModelParameters(
        alpha_by_partition=np.asarray(alpha_by_partition, dtype=np.float64),
        sigma2_by_partition=np.asarray(sigma2_by_partition, dtype=np.float64),
        partition_weights=None if partition_weights is None else np.asarray(partition_weights, dtype=np.float64),
        coverage_scale=None if coverage_scale is None else np.asarray(coverage_scale, dtype=np.float64),
        rate_by_dim=None if rate_by_dim is None else np.asarray(rate_by_dim, dtype=np.float64),
        precision_weights=(
            None if precision_weights is None else np.asarray(precision_weights, dtype=np.float64)
        ),
        gamma_rates_by_partition=gamma_rates_by_partition,
    )
    bundle = build_torch_score_bundle(
        embeddings=embeddings,
        mask=mask,
        dim_to_partition=dim_to_partition,
        params=params,
        device=device,
        dtype=dtype,
    )
    if root is None:
        return float(bundle.score_fn(tree))

    # For custom root we need a direct call bypassing closure defaults.
    prep = _prepare_static_tensors(
        embeddings=embeddings,
        mask=mask,
        dim_to_partition=dim_to_partition,
        params=params,
        device=bundle.device,
        dtype=_resolve_torch_dtype_for_device(dtype=dtype, device=bundle.device),
    )
    return _ou_log_likelihood_torch_prepared(
        tree=tree,
        prepared=prep,
        alpha=np.asarray(alpha_by_partition, dtype=np.float64),
        sigma2=np.asarray(sigma2_by_partition, dtype=np.float64),
        gamma_rates_by_partition=gamma_rates_by_partition,
        root=root,
    )


@dataclass(frozen=True)
class _Prepared:
    embeddings: "torch.Tensor"
    mask: "torch.Tensor"
    dim_to_partition: np.ndarray
    partition_weights: "torch.Tensor"
    coverage: "torch.Tensor"
    dim_rate: "torch.Tensor"
    precision: "torch.Tensor"
    n_partitions: int
    device: str
    dtype: "torch.dtype"


def _prepare_static_tensors(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    device: str,
    dtype,
) -> _Prepared:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask shape mismatch")
    if dim_map.shape != (z.shape[1],):
        raise ValueError("dim_to_partition shape mismatch")
    if np.any(dim_map < 0):
        raise ValueError("dim_to_partition must be non-negative")
    n_taxa, d_total = z.shape
    n_partitions = int(dim_map.max()) + 1 if d_total > 0 else 0

    if params.partition_weights is None:
        p_weights = np.ones(n_partitions, dtype=np.float64)
    else:
        p_weights = np.asarray(params.partition_weights, dtype=np.float64)
        if p_weights.shape != (n_partitions,):
            raise ValueError("partition_weights length mismatch")

    if params.coverage_scale is None:
        coverage = np.ones((n_taxa, n_partitions), dtype=np.float64)
    else:
        coverage = np.asarray(params.coverage_scale, dtype=np.float64)
        if coverage.shape != (n_taxa, n_partitions):
            raise ValueError("coverage_scale shape mismatch")

    if params.rate_by_dim is None:
        dim_rate = np.ones(d_total, dtype=np.float64)
    else:
        dim_rate = np.asarray(params.rate_by_dim, dtype=np.float64)
        if dim_rate.shape != (d_total,):
            raise ValueError("rate_by_dim shape mismatch")

    if params.precision_weights is None:
        precision = np.ones((n_taxa, d_total), dtype=np.float64)
    else:
        precision = np.asarray(params.precision_weights, dtype=np.float64)
        if precision.shape != (n_taxa, d_total):
            raise ValueError("precision_weights shape mismatch")

    return _Prepared(
        embeddings=torch.as_tensor(z, dtype=dtype, device=device),
        mask=torch.as_tensor(m, dtype=torch.bool, device=device),
        dim_to_partition=dim_map,
        partition_weights=torch.as_tensor(p_weights, dtype=dtype, device=device),
        coverage=torch.as_tensor(coverage, dtype=dtype, device=device),
        dim_rate=torch.as_tensor(dim_rate, dtype=dtype, device=device),
        precision=torch.as_tensor(precision, dtype=dtype, device=device),
        n_partitions=n_partitions,
        device=device,
        dtype=dtype,
    )


def _ou_log_likelihood_torch_prepared(
    tree: PhyloTree,
    prepared: _Prepared,
    alpha: np.ndarray,
    sigma2: np.ndarray,
    gamma_rates_by_partition: Mapping[int, tuple[np.ndarray, np.ndarray]] | None,
    root: int | None = None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if alpha.shape != (prepared.n_partitions,):
        raise ValueError("alpha length mismatch")
    if sigma2.shape != (prepared.n_partitions,):
        raise ValueError("sigma2 length mismatch")

    rooted = tree.rooted(root=root)
    p_terms = torch.zeros(prepared.n_partitions, dtype=prepared.dtype, device=prepared.device)

    for p in range(prepared.n_partitions):
        dim_idx_np = np.flatnonzero(prepared.dim_to_partition == p)
        if dim_idx_np.size == 0:
            continue
        dim_idx = torch.as_tensor(dim_idx_np, dtype=torch.long, device=prepared.device)
        rate = prepared.dim_rate[dim_idx]
        if gamma_rates_by_partition is not None and p in gamma_rates_by_partition:
            gr, gw = gamma_rates_by_partition[p]
            ll = _partition_ll_discrete_gamma_torch(
                rooted=rooted,
                prepared=prepared,
                dim_idx=dim_idx,
                alpha=float(alpha[p]),
                sigma2=float(sigma2[p]),
                dim_rate=rate,
                gamma_rates=np.asarray(gr, dtype=np.float64),
                gamma_weights=np.asarray(gw, dtype=np.float64),
            )
        else:
            ll = _partition_dim_ll_torch(
                rooted=rooted,
                prepared=prepared,
                dim_idx=dim_idx,
                alpha=float(alpha[p]),
                sigma2=float(sigma2[p]),
                dim_rate=rate,
            ).sum()
        p_terms[p] = ll
    total = torch.dot(prepared.partition_weights, p_terms)
    return float(total.detach().cpu().item())


def _partition_dim_ll_torch(
    rooted,
    prepared: _Prepared,
    dim_idx: "torch.Tensor",
    alpha: float,
    sigma2: float,
    dim_rate: "torch.Tensor",
) -> "torch.Tensor":
    n_nodes = rooted.num_nodes
    n_taxa = prepared.embeddings.shape[0]
    m = int(dim_idx.shape[0])
    if m == 0:
        return torch.zeros(0, dtype=prepared.dtype, device=prepared.device)

    J = torch.zeros((n_nodes, m), dtype=prepared.dtype, device=prepared.device)
    h = torch.zeros((n_nodes, m), dtype=prepared.dtype, device=prepared.device)
    c = torch.zeros((n_nodes, m), dtype=prepared.dtype, device=prepared.device)

    obs = prepared.mask[:, dim_idx]
    y = prepared.embeddings[:, dim_idx]
    precision = prepared.precision[:, dim_idx]

    part_ids = prepared.dim_to_partition[dim_idx.detach().cpu().numpy()]
    p = int(part_ids[0])
    cov = prepared.coverage[:, p : p + 1]
    var = (sigma2 * cov) / precision
    inv_var = torch.where(obs, 1.0 / var, torch.zeros_like(var))

    J[:n_taxa] = inv_var
    h[:n_taxa] = y * inv_var
    two_pi = torch.tensor(2.0 * np.pi, dtype=prepared.dtype, device=prepared.device)
    c[:n_taxa] = torch.where(
        obs,
        -0.5 * (y * y * inv_var + torch.log(two_pi * var)),
        torch.zeros_like(y),
    )

    for node in rooted.postorder:
        if node == rooted.root:
            continue
        parent = int(rooted.parent[node])
        t = float(rooted.branch_length_to_parent[node])
        a = torch.exp((-alpha * t) * dim_rate)
        q = 1.0 - a * a

        Jc = J[node]
        hc = h[node]
        cc = c[node]
        denom = 1.0 + q * Jc

        Jmsg = (a * a) * Jc / denom
        hmsg = a * hc / denom
        cmsg = cc - 0.5 * torch.log(denom) + 0.5 * (hc * hc) * q / denom

        J[parent] += Jmsg
        h[parent] += hmsg
        c[parent] += cmsg

    r = int(rooted.root)
    Jr = J[r]
    hr = h[r]
    cr = c[r]
    return cr - 0.5 * torch.log(1.0 + Jr) + 0.5 * (hr * hr) / (1.0 + Jr)


def _partition_ll_discrete_gamma_torch(
    rooted,
    prepared: _Prepared,
    dim_idx: "torch.Tensor",
    alpha: float,
    sigma2: float,
    dim_rate: "torch.Tensor",
    gamma_rates: np.ndarray,
    gamma_weights: np.ndarray,
) -> "torch.Tensor":
    if gamma_rates.ndim != 1 or gamma_weights.ndim != 1:
        raise ValueError("gamma rates/weights must be 1D")
    if gamma_rates.shape != gamma_weights.shape:
        raise ValueError("gamma rates/weights shape mismatch")
    if np.any(gamma_rates <= 0) or np.any(gamma_weights <= 0):
        raise ValueError("gamma rates/weights must be > 0")

    rates = torch.as_tensor(gamma_rates, dtype=prepared.dtype, device=prepared.device)
    w = torch.as_tensor(gamma_weights, dtype=prepared.dtype, device=prepared.device)
    w = w / torch.sum(w)
    logw = torch.log(w)

    ll_cat = []
    for i in range(int(rates.shape[0])):
        ll = _partition_dim_ll_torch(
            rooted=rooted,
            prepared=prepared,
            dim_idx=dim_idx,
            alpha=alpha,
            sigma2=sigma2,
            dim_rate=dim_rate * rates[i],
        )
        ll_cat.append(ll + logw[i])
    stack = torch.stack(ll_cat, dim=0)
    return torch.logsumexp(stack, dim=0).sum()


def _parse_torch_dtype(dtype: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    d = dtype.lower()
    if d in {"float32", "fp32"}:
        return torch.float32
    if d in {"float64", "fp64"}:
        return torch.float64
    if d in {"float16", "fp16"}:
        return torch.float16
    if d in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype '{dtype}'")


def _resolve_torch_dtype_for_device(dtype: str, device: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    requested = _parse_torch_dtype(dtype)

    candidates = [requested]
    # MPS commonly has dtype limitations; ensure we pick a supported one.
    if device == "mps":
        for alt in (torch.float32, torch.float16):
            if alt not in candidates:
                candidates.append(alt)

    for cand in candidates:
        try:
            _ = torch.empty((1,), device=device, dtype=cand)
            return cand
        except Exception:
            continue
    raise RuntimeError(f"no supported dtype found for device '{device}'")
