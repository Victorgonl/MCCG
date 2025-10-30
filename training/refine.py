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


import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def update_edges_by_cosine(
    data: Data,
    weight_matrix: torch.Tensor,
    high_thresh=0.95,  # to create an edge
    low_thresh=0.25,  # to remove an edge
    percent=0.3,  # percent of node pairs to sample
    alpha=0.05,  # weight update rate
):
    """
    Updates the graph structure and a weight matrix based on cosine similarity
    of node features.

    1. Samples a subset of node pairs.
    2. Updates their entries in `weight_matrix` via EMA of cosine similarity.
    3. Removes existing edges if their weight is < low_thresh.
    4. Adds new edges if a sampled pair's weight is > high_thresh.
    """
    x = data.x
    num_nodes = x.size(0)
    device = x.device

    # Handle empty or trivial graphs
    if num_nodes < 2:
        return data, weight_matrix

    # --- 1. Sample Node Pairs ---
    num_samples = int(num_nodes * num_nodes * percent)

    # If sampling is too low, just return
    if num_samples == 0:
        # Still need to filter existing edges based on low_thresh
        if data.edge_index.numel() > 0:
            current_edge_weights = weight_matrix[data.edge_index[0], data.edge_index[1]]
            keep_mask = current_edge_weights >= low_thresh
            new_edge_index = data.edge_index[:, keep_mask]

            new_data = Data(x=x, edge_index=new_edge_index)
            # Copy other attributes
            for key, value in data:
                if key not in ["x", "edge_index"]:
                    new_data[key] = value
            return new_data, weight_matrix
        else:
            return data, weight_matrix

    # Sample source and target nodes
    i = torch.randint(0, num_nodes, (num_samples,), device=device)
    j = torch.randint(0, num_nodes, (num_samples,), device=device)

    # Filter out self-loops (i == j)
    mask = i != j
    i, j = i[mask], j[mask]

    # If mask filtered everything
    if i.numel() == 0:
        return data, weight_matrix

    # --- 2. Calculate Cosine Similarity & Update Weights (for sampled pairs) ---

    # Normalize features for efficient cosine similarity calculation
    x_norm = F.normalize(x, p=2, dim=1)

    # Get features for sampled pairs
    feat_i = x_norm[i]
    feat_j = x_norm[j]

    # Cosine similarity is the dot product of normalized vectors
    # Detach to prevent gradients from flowing back from this update
    cos_sim = (feat_i * feat_j).sum(dim=1).detach()

    # Update weight_matrix using Exponential Moving Average (EMA)
    # W_new = (1 - alpha) * W_old + alpha * S_current
    weight_matrix[i, j] = (1 - alpha) * weight_matrix[i, j] + alpha * cos_sim
    # Enforce symmetry
    weight_matrix[j, i] = weight_matrix[i, j]

    # --- 3. Edge Removal (from *existing* edges) ---

    if data.edge_index.numel() > 0:
        # Get weights for all current edges
        current_edge_weights = weight_matrix[data.edge_index[0], data.edge_index[1]]

        # Keep edges that are *at or above* the low threshold
        keep_mask = current_edge_weights >= low_thresh
        kept_edges = data.edge_index[:, keep_mask]
    else:
        # No existing edges to keep
        kept_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    # --- 4. Edge Addition (from *sampled* pairs) ---

    # Check the *updated* weights of the pairs we just sampled
    updated_sampled_weights = weight_matrix[i, j]

    # Find pairs that are *above* the high threshold
    add_mask = updated_sampled_weights > high_thresh

    i_add = i[add_mask]
    j_add = j[add_mask]

    # Stack new edges, adding both directions (i, j) and (j, i)
    new_edges_to_add = torch.stack(
        [torch.cat([i_add, j_add]), torch.cat([j_add, i_add])], dim=0
    )

    # --- 5. Combine and Coalesce ---

    # Concatenate the edges we kept with the new ones we're adding
    new_edge_index = torch.cat([kept_edges, new_edges_to_add], dim=1)

    # Use coalesce to remove duplicate edges and sort them
    new_edge_index = coalesce(new_edge_index, num_nodes=num_nodes)

    # Create a new Data object with the updated edge_index
    new_data = Data(x=x, edge_index=new_edge_index)

    # Copy over all other attributes (like y, masks, etc.)
    for key, value in data:
        if key not in ["x", "edge_index"]:
            new_data[key] = value

    return new_data, weight_matrix
