from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random


""" def update_edges_by_cosine(
    data,
    high_thresh=0.75,
    low_thresh=0.45,
    percent=0.2,
    freeze_mask: Optional[torch.Tensor] = None,
):
    x = F.normalize(data.x, dim=1)
    sim = torch.mm(x, x.t())
    n = sim.size(0)
    edge_set = set([(int(u), int(v)) for u, v in data.edge_index.t().tolist()])

    # freeze edges
    frozen_edges = set()
    if freeze_mask is not None:
        frozen_edges = set([(int(u), int(v)) for u, v in freeze_mask.t().tolist()])

    new_edges = set()
    total_pairs = n * (n - 1) // 2
    num_samples = int(total_pairs * percent)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_pairs = (
        random.sample(all_pairs, num_samples)
        if num_samples < total_pairs
        else all_pairs
    )

    for i, j in sampled_pairs:
        if (i, j) in frozen_edges or (j, i) in frozen_edges:
            continue  # skip frozen edges

        s = sim[i, j].item()
        if s > high_thresh:
            new_edges.add((i, j))
            new_edges.add((j, i))
        elif s < low_thresh and (i, j) in edge_set:
            edge_set.discard((i, j))
            edge_set.discard((j, i))

    # combine existing edges, new edges, and frozen edges
    updated_edges = list(edge_set.union(new_edges).union(frozen_edges))
    if len(updated_edges) == 0:
        updated_edges = [(0, 0)]

    edge_index = torch.tensor(updated_edges, dtype=torch.long).t().contiguous()

    # return updated data and updated freeze mask
    return Data(x=data.x, edge_index=edge_index), (
        torch.tensor(list(frozen_edges), dtype=torch.long).t().contiguous()
        if frozen_edges
        else None
    )
 """

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random


""" def update_edges_by_cosine(
    data,
    weight_matrix: torch.Tensor,
    high_thresh=0.75,
    low_thresh=0.45,
    percent=0.2,
    lr=0.1,
    decay=0.95,
):
    x = F.normalize(data.x, dim=1)
    sim = torch.mm(x, x.t())
    n = sim.size(0)
    edge_set = set([(int(u), int(v)) for u, v in data.edge_index.t().tolist()])

    total_pairs = n * (n - 1) // 2
    num_samples = int(total_pairs * percent)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_pairs = (
        random.sample(all_pairs, num_samples)
        if num_samples < total_pairs
        else all_pairs
    )

    for i, j in sampled_pairs:
        s = sim[i, j].item()

        # Update weights using cosine similarity feedback
        if s > high_thresh:
            weight_matrix[i, j] = weight_matrix[i, j] * decay + lr * s
            weight_matrix[j, i] = weight_matrix[j, i] * decay + lr * s
            edge_set.add((i, j))
            edge_set.add((j, i))
        elif s < low_thresh:
            weight_matrix[i, j] = weight_matrix[i, j] * decay
            weight_matrix[j, i] = weight_matrix[j, i] * decay
            if (i, j) in edge_set:
                edge_set.discard((i, j))
                edge_set.discard((j, i))
        else:
            weight_matrix[i, j] *= decay
            weight_matrix[j, i] *= decay

    # Threshold weights to optionally prune very low-weight connections
    weight_thresh = 0.05
    for i in range(n):
        for j in range(i + 1, n):
            if weight_matrix[i, j] < weight_thresh and (i, j) in edge_set:
                edge_set.discard((i, j))
                edge_set.discard((j, i))

    updated_edges = list(edge_set)
    if len(updated_edges) == 0:
        updated_edges = [(0, 0)]

    edge_index = torch.tensor(updated_edges, dtype=torch.long).t().contiguous()

    return Data(x=data.x, edge_index=edge_index), weight_matrix
 """


def update_edges_by_cosine(
    data,
    weight_matrix: torch.Tensor,
    high_thresh=0.75,
    low_thresh=0.45,
    percent=0.2,
    lr=0.1,
    decay=0.95,
    weight_thresh=0.05,
):
    x = F.normalize(data.x, dim=1)
    n = x.size(0)

    # Compute cosine similarity (GPU-accelerated)
    sim = torch.mm(x, x.t())

    # Get upper-triangle indices (avoid duplicates)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1).to(x.device)

    # Optional random sampling to reduce computation
    total_pairs = idx_i.size(0)
    num_samples = int(total_pairs * percent)
    if num_samples < total_pairs:
        perm = torch.randperm(total_pairs, device=x.device)[:num_samples]
        idx_i, idx_j = idx_i[perm], idx_j[perm]

    s = sim[idx_i, idx_j]

    # Prepare masks
    high_mask = s > high_thresh
    low_mask = s < low_thresh
    mid_mask = (~high_mask) & (~low_mask)

    # Update weights using vectorized operations
    weight_matrix[idx_i[high_mask], idx_j[high_mask]] = (
        weight_matrix[idx_i[high_mask], idx_j[high_mask]] * decay + lr * s[high_mask]
    )
    weight_matrix[idx_j[high_mask], idx_i[high_mask]] = (
        weight_matrix[idx_j[high_mask], idx_i[high_mask]] * decay + lr * s[high_mask]
    )

    weight_matrix[idx_i[low_mask], idx_j[low_mask]] *= decay
    weight_matrix[idx_j[low_mask], idx_i[low_mask]] *= decay

    weight_matrix[idx_i[mid_mask], idx_j[mid_mask]] *= decay
    weight_matrix[idx_j[mid_mask], idx_i[mid_mask]] *= decay

    # Build edge set based on thresholds (tensor form)
    keep_mask = weight_matrix > weight_thresh
    edge_index = keep_mask.nonzero(as_tuple=False).t().contiguous()

    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=x.device)

    return Data(x=data.x, edge_index=edge_index), weight_matrix
