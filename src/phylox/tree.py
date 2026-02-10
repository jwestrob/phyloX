from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootedTree:
    num_nodes: int
    root: int
    parent: np.ndarray
    children: list[list[int]]
    branch_length_to_parent: np.ndarray
    preorder: np.ndarray
    postorder: np.ndarray


@dataclass(frozen=True)
class PhyloTree:
    """Undirected tree with branch lengths and leaf ids 0..leaf_count-1."""

    num_nodes: int
    edges: list[tuple[int, int, float]]
    leaf_count: int

    def __post_init__(self) -> None:
        if self.leaf_count <= 0:
            raise ValueError("leaf_count must be positive")
        if self.num_nodes < self.leaf_count:
            raise ValueError("num_nodes must be >= leaf_count")
        if len(self.edges) != self.num_nodes - 1:
            raise ValueError("a tree with N nodes must contain N-1 edges")
        for u, v, t in self.edges:
            if u < 0 or v < 0 or u >= self.num_nodes or v >= self.num_nodes:
                raise ValueError("edge endpoint out of range")
            if u == v:
                raise ValueError("self-loops are not allowed")
            if t < 0:
                raise ValueError("branch lengths must be non-negative")
        if not self.is_connected():
            raise ValueError("tree is not connected")

    def adjacency(self) -> list[list[tuple[int, float]]]:
        adj: list[list[tuple[int, float]]] = [[] for _ in range(self.num_nodes)]
        for u, v, t in self.edges:
            adj[u].append((v, t))
            adj[v].append((u, t))
        return adj

    def is_connected(self) -> bool:
        adj = self.adjacency()
        seen = np.zeros(self.num_nodes, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            node = stack.pop()
            for nxt, _ in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        return bool(seen.all())

    def rooted(self, root: int | None = None) -> RootedTree:
        if root is None:
            root = self.default_root()
        if root < 0 or root >= self.num_nodes:
            raise ValueError("root out of range")

        adj = self.adjacency()
        parent = np.full(self.num_nodes, -1, dtype=np.int64)
        branch = np.zeros(self.num_nodes, dtype=np.float64)
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]

        stack: list[tuple[int, int]] = [(root, -1)]
        preorder_list: list[int] = []
        while stack:
            node, par = stack.pop()
            if parent[node] != -1:
                continue
            parent[node] = node if par == -1 else par
            preorder_list.append(node)
            for nxt, t in adj[node]:
                if nxt == par:
                    continue
                if parent[nxt] != -1:
                    continue
                children[node].append(nxt)
                branch[nxt] = float(t)
                stack.append((nxt, node))

        if np.any(parent == -1):
            raise ValueError("failed to orient tree from root")

        preorder = np.asarray(preorder_list, dtype=np.int64)
        postorder = preorder[::-1].copy()
        return RootedTree(
            num_nodes=self.num_nodes,
            root=root,
            parent=parent,
            children=children,
            branch_length_to_parent=branch,
            preorder=preorder,
            postorder=postorder,
        )

    def default_root(self) -> int:
        return self.leaf_count if self.leaf_count < self.num_nodes else 0


def pairwise_leaf_distances(tree: PhyloTree) -> np.ndarray:
    """Pairwise patristic distances between leaves [0..leaf_count-1]."""
    adj = tree.adjacency()
    n = tree.leaf_count
    out = np.zeros((n, n), dtype=np.float64)

    for src in range(n):
        dist = np.full(tree.num_nodes, np.nan, dtype=np.float64)
        dist[src] = 0.0
        stack = [src]
        while stack:
            node = stack.pop()
            for nxt, t in adj[node]:
                if np.isnan(dist[nxt]):
                    dist[nxt] = dist[node] + t
                    stack.append(nxt)
        out[src, :] = dist[:n]
    return out
