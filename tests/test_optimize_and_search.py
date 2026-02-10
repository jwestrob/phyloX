import numpy as np

from phylox.optimize import OUModelParameters, optimize_branch_lengths_coordinate, optimize_partition_parameters, score_ou_model
from phylox.search import apply_nni_swap, apply_spr_move, enumerate_nni_moves, nni_hillclimb
from phylox.simulate import simulate_ou_embeddings, simulate_random_binary_tree
from phylox.tree import PhyloTree


def _small_tree() -> PhyloTree:
    edges = [
        (4, 0, 0.2),
        (4, 1, 0.3),
        (4, 5, 0.1),
        (5, 2, 0.25),
        (5, 3, 0.35),
    ]
    return PhyloTree(num_nodes=6, edges=edges, leaf_count=4)


def test_branch_length_optimization_improves_or_matches_likelihood():
    rng = np.random.default_rng(123)
    tree = _small_tree()
    dim_map = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.9, 1.2], dtype=np.float64)
    sigma2 = np.asarray([0.08, 0.12], dtype=np.float64)
    z, m = simulate_ou_embeddings(tree, dim_map, alpha, sigma2, seed=10)

    perturbed_edges = [(u, v, t * float(rng.uniform(0.5, 1.8))) for u, v, t in tree.edges]
    start_tree = PhyloTree(num_nodes=tree.num_nodes, edges=perturbed_edges, leaf_count=tree.leaf_count)
    params = OUModelParameters(alpha_by_partition=alpha, sigma2_by_partition=sigma2)

    ll0 = score_ou_model(start_tree, z, m, dim_map, params)
    out_tree, ll1 = optimize_branch_lengths_coordinate(start_tree, z, m, dim_map, params, rounds=2)
    _ = out_tree
    assert ll1 >= ll0 - 1e-8


def test_partition_parameter_optimization_improves_or_matches_likelihood():
    tree = _small_tree()
    dim_map = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    alpha_true = np.asarray([0.8, 1.4], dtype=np.float64)
    sigma2_true = np.asarray([0.05, 0.2], dtype=np.float64)
    z, m = simulate_ou_embeddings(tree, dim_map, alpha_true, sigma2_true, seed=21)

    init = OUModelParameters(
        alpha_by_partition=np.asarray([0.2, 2.5], dtype=np.float64),
        sigma2_by_partition=np.asarray([0.3, 0.03], dtype=np.float64),
    )
    ll0 = score_ou_model(tree, z, m, dim_map, init)
    params1, ll1 = optimize_partition_parameters(tree, z, m, dim_map, init, rounds=2)
    _ = params1
    assert ll1 >= ll0 - 1e-8


def test_nni_hillclimb_non_decreasing():
    tree = simulate_random_binary_tree(n_taxa=8, seed=99)
    dim_map = np.repeat(np.arange(3), repeats=[20, 20, 20]).astype(np.int64)
    alpha = np.asarray([0.8, 1.0, 1.2], dtype=np.float64)
    sigma2 = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
    z, m = simulate_ou_embeddings(tree, dim_map, alpha, sigma2, seed=101)
    params = OUModelParameters(alpha_by_partition=alpha, sigma2_by_partition=sigma2)

    # Build a perturbed topology from one NNI alternative.
    iedge = tree.internal_edge_indices()[0]
    u, v, _ = tree.edges[iedge]
    u_nei = [x for x, _ in tree.neighbors(u) if x != v]
    v_nei = [x for x, _ in tree.neighbors(v) if x != u]
    start_tree = apply_nni_swap(tree, u, v, u_nei[0], v_nei[0])

    score = lambda t: score_ou_model(t, z, m, dim_map, params)
    ll0 = score(start_tree)
    result = nni_hillclimb(start_tree, score, max_rounds=6, edge_disjoint_batch=True)
    assert result.log_likelihood >= ll0 - 1e-8


def test_nni_enumeration_batch_matches_scalar():
    tree = simulate_random_binary_tree(n_taxa=8, seed=31)
    dim_map = np.repeat(np.arange(2), repeats=[10, 10]).astype(np.int64)
    alpha = np.asarray([0.9, 1.1], dtype=np.float64)
    sigma2 = np.asarray([0.07, 0.09], dtype=np.float64)
    z, m = simulate_ou_embeddings(tree, dim_map, alpha, sigma2, seed=32)
    params = OUModelParameters(alpha_by_partition=alpha, sigma2_by_partition=sigma2)
    score = lambda t: score_ou_model(t, z, m, dim_map, params)

    scalar = enumerate_nni_moves(tree, score)
    batch = enumerate_nni_moves(
        tree,
        score,
        batch_score_fn=lambda trees: np.asarray([score(t) for t in trees], dtype=np.float64),
    )

    sig_scalar = [(mv.edge, mv.swap) for mv in scalar]
    sig_batch = [(mv.edge, mv.swap) for mv in batch]
    assert sig_scalar == sig_batch


def test_apply_spr_move_preserves_tree_shape():
    tree = simulate_random_binary_tree(n_taxa=7, seed=17)
    deg = tree.degrees()
    attach = int(np.flatnonzero(deg >= 3)[0])
    subtree = tree.neighbors(attach)[0][0]

    # Find a target edge away from attach/subtree.
    subtree_nodes = {subtree}
    stack = [subtree]
    seen = {attach}
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        subtree_nodes.add(node)
        for nxt, _ in tree.neighbors(node):
            if nxt not in seen:
                stack.append(nxt)

    target = None
    for u, v, _ in tree.edges:
        if u == attach or v == attach:
            continue
        if u in subtree_nodes or v in subtree_nodes:
            continue
        target = (u, v)
        break
    assert target is not None

    moved = apply_spr_move(tree, attach_node=attach, subtree_root=subtree, target_edge=target)
    assert moved.num_nodes == tree.num_nodes
    assert len(moved.edges) == len(tree.edges)
    assert moved.leaf_count == tree.leaf_count
