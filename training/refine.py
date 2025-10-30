import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random

def update_edges_by_cosine(data, high_thresh=0.75, low_thresh=0.45, percent=0.2, freeze_mask=None):
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
    sampled_pairs = random.sample(all_pairs, num_samples) if num_samples < total_pairs else all_pairs

    for i, j in sampled_pairs:
        if (i, j) in frozen_edges or (j, i) in frozen_edges:
            continue # skip frozen edges

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
    return Data(x=data.x, edge_index=edge_index), torch.tensor(list(frozen_edges), dtype=torch.long).t().contiguous() if frozen_edges else None
