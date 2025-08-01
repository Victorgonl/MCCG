import torch
from torch_scatter import scatter
from torch_geometric.utils import degree, to_networkx
import networkx as nx
from params import set_params

_, args = set_params()


def compute_pr(data, damp: float = 0.85, k: int = 10):
    """
    Compute PageRank scores for each node in the graph.

    Args:
        data (torch_geometric.data.Data): Input graph data with edge_index.
        damp (float): Damping factor, usually between 0.85 and 0.9.
        k (int): Number of power iterations to run.

    Returns:
        torch.Tensor: PageRank scores for each node.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    deg_out = degree(edge_index[0], num_nodes=num_nodes)
    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce="sum", dim_size=num_nodes)
        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    """
    Compute eigenvector centrality for each node in the graph.

    Args:
        data (torch_geometric.data.Data): Input graph data with edge_index.

    Returns:
        torch.Tensor: Eigenvector centrality values per node.
    """
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature(x, drop_prob):
    """
    Randomly drops features across all nodes with uniform probability.

    Args:
        x (torch.Tensor): Node features [num_nodes, num_features].
        drop_prob (float): Probability of dropping each feature.

    Returns:
        torch.Tensor: Modified feature matrix with some features dropped.
    """
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    """
    Drop features based on importance weights and a probability threshold.

    Args:
        x (torch.Tensor): Node features [num_nodes, num_features].
        w (torch.Tensor): Feature importance weights.
        p (float): Base drop probability.
        threshold (float): Maximum drop probability.

    Returns:
        torch.Tensor: Feature matrix with weighted dropped features.
    """
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[drop_mask] = 0.0
    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    """
    Drop features based on global importance weights (non-repeated across nodes).

    Args:
        x (torch.Tensor): Node features [num_nodes, num_features].
        w (torch.Tensor): Feature importance weights.
        p (float): Base drop probability.
        threshold (float): Maximum drop probability.

    Returns:
        torch.Tensor: Feature matrix with weighted dropped features.
    """
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x


def feature_drop_weights(x, node_c):
    """
    Compute feature drop weights based on binary feature presence.

    Args:
        x (torch.Tensor): Binary feature matrix.
        node_c (torch.Tensor): Centrality scores per node.

    Returns:
        torch.Tensor: Drop weights for each feature.
    """
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def feature_drop_weights_dense(x, node_c):
    """
    Compute feature drop weights using absolute values (for dense features).

    Args:
        x (torch.Tensor): Dense feature matrix.
        node_c (torch.Tensor): Centrality scores per node.

    Returns:
        torch.Tensor: Drop weights for each feature.
    """
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def drop_edge_weighted_by_label(edge_index, edge_weights, p: float, label):
    """
    Drop edges based on edge importance weights and a drop probability,
    conditioned on node labels to preserve label-consistent connections.

    Args:
        edge_index (torch.Tensor): Edge indices [2, num_edges].
        edge_weights (torch.Tensor): Edge weights.
        label (torch.Tensor): Node labels [num_nodes].
        p (float): Base drop probability.

    Returns:
        torch.Tensor: Filtered edge indices after dropout.
    """
    edge_weights = edge_weights / edge_weights.mean() * p
    src, dst = edge_index
    label_mask = label[src] == label[dst]
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    final_mask = sel_mask | label_mask
    return edge_index[:, final_mask]


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    """
    Drop edges based on edge importance weights and a drop probability.

    Args:
        edge_index (torch.Tensor): Edge indices [2, num_edges].
        edge_weights (torch.Tensor): Edge weights.
        p (float): Base drop probability.
        threshold (float): Maximum weight threshold.

    Returns:
        torch.Tensor: Filtered edge indices after dropout.
    """
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]


def degree_drop_weights(data):
    """
    Compute edge weights based on the degree of target nodes.

    Args:
        data (torch_geometric.data.Data): Input graph data.

    Returns:
        torch.Tensor: Drop weights based on node degree.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    deg = degree(edge_index[1], num_nodes=num_nodes)
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    return weights


def pr_drop_weights(data, aggr: str = "sink", k: int = 10):
    """
    Compute edge drop weights based on PageRank values.

    Args:
        data (torch_geometric.data.Data): Input graph data.
        aggr (str): Aggregation method: 'sink', 'source', or 'mean'.
        k (int): Number of iterations for PageRank.

    Returns:
        torch.Tensor: Edge drop weights based on PageRank.
    """
    edge_index = data.edge_index
    pv = compute_pr(data, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == "sink":
        s = s_col
    elif aggr == "source":
        s = s_row
    elif aggr == "mean":
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())
    return weights


def evc_drop_weights(data):
    """
    Compute edge drop weights based on eigenvector centrality of target nodes.

    Args:
        data (torch_geometric.data.Data): Input graph data.

    Returns:
        torch.Tensor: Edge drop weights based on eigenvector centrality.
    """
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()
    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col
    return (s.max() - s) / (s.max() - s.mean())
