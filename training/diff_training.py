import os
import time
import random
from datetime import datetime

import torch
import torch.nn as nn

from dataset.enhance_graph import *
from dataset.load_data import load_dataset, load_graph
from model.mccg_model import GAT
from .utils import *
from os.path import join
from params import set_params

# Parameters
_, args = set_params()

device = torch.device(
    ("cuda:" + str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu"
)

print("Device:", device)
print("=" * 15, "\n")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_pairs(label, num_pos=5000, num_neg=5000):
    """Sample positive and negative pairs from multi-label data and shuffle them."""
    label = label.cpu().numpy()
    n = len(label)
    pos_pairs, neg_pairs = [], []

    for i in range(n):
        for j in range(i + 1, n):
            same_label = (label[i] * label[j]).sum() > 0
            if same_label:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    pos_pairs = pos_pairs[: min(num_pos, len(pos_pairs))]
    neg_pairs = neg_pairs[: min(num_neg, len(neg_pairs))]

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    # Shuffle pairs and labels together
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)

    return torch.tensor(pairs, dtype=torch.long, device=device), torch.tensor(
        labels, dtype=torch.float32, device=device
    )


class DiffModel(nn.Module):
    def __init__(self, encoder, dim_hidden):
        super(DiffModel, self).__init__()
        self.encoder = encoder
        self.diff_classifier = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, 1),
        )

    def forward(self, x1, adj1, M1, x2, adj2, M2, pairs):
        z1 = self.encoder(x1, adj1, M1)
        z2 = self.encoder(x2, adj2, M2)
        diff = torch.abs(z1[pairs[:, 0]] - z2[pairs[:, 1]])
        return self.diff_classifier(diff).squeeze(-1)


class DiffTrainer:
    def __init__(self) -> None:
        pass

    def fit(
        self,
        logger,
        mode,
        combin_num,
        layer_shape,
        dim_proj_multiview,
        dim_proj_cluster,
        drop_scheme,
        drop_feature_rate_view1,
        drop_feature_rate_view2,
        drop_edge_rate_view1,
        drop_edge_rate_view2,
        th_a,
        th_o,
        th_v,
        db_eps,
        db_min,
        l2_coef,
        w_cluster,
        t_multiview,
        t_cluster,
    ):

        set_seed(42)  # Optional reproducibility
        names, pubs = load_dataset("train")
        criterion = nn.BCEWithLogitsLoss()

        encoder = GAT(layer_shape[0], layer_shape[1], layer_shape[2])
        model = DiffModel(encoder, dim_hidden=layer_shape[2]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=l2_coef
        )

        graph_data = {}
        for name in names:
            label, ft_list, data = load_graph(name, "train", th_a, th_o, th_v)
            label, ft_list, data = (
                label.to(device).float(),
                ft_list.float().to(device),
                data.to(device),
            )
            adj = get_adj(data.edge_index, data.num_nodes)
            M = get_M(adj, t=2)
            pairs, pair_labels = sample_pairs(label, 5000, 5000)

            # Drop scheme
            if drop_scheme == "degree":
                edge_weights = degree_drop_weights(data).to(device)
                node_deg = degree(data.edge_index[1], num_nodes=data.num_nodes)
                feature_weights = feature_drop_weights_dense(
                    ft_list, node_c=node_deg
                ).to(device)
            elif drop_scheme == "pr":
                edge_weights = pr_drop_weights(data, aggr="sink", k=200).to(device)
                node_pr = compute_pr(data)
                feature_weights = feature_drop_weights_dense(
                    ft_list, node_c=node_pr
                ).to(device)
            elif drop_scheme == "evc":
                edge_weights = evc_drop_weights(data).to(device)
                node_evc = eigenvector_centrality(data)
                feature_weights = feature_drop_weights_dense(
                    ft_list, node_c=node_evc
                ).to(device)
            else:
                raise ValueError(f"undefined drop scheme: {drop_scheme}.")

            graph_data[name] = (
                ft_list,
                data,
                adj,
                M,
                pairs,
                pair_labels,
                edge_weights,
                feature_weights,
            )

        # Training loop
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            start_epoch_time = datetime.now()

            for p, name in enumerate(names, 1):
                (
                    ft_list,
                    data,
                    adj,
                    M,
                    pairs,
                    pair_labels,
                    edge_weights,
                    feature_weights,
                ) = graph_data[name]

                edge_index1 = drop_edge_weighted(
                    data.edge_index, edge_weights, p=drop_edge_rate_view1, threshold=0.7
                )
                edge_index2 = drop_edge_weighted(
                    data.edge_index, edge_weights, p=drop_edge_rate_view2, threshold=0.7
                )

                adj1, adj2 = get_adj(edge_index1, data.num_nodes), get_adj(
                    edge_index2, data.num_nodes
                )
                M1, M2 = get_M(adj1, t=2), get_M(adj2, t=2)
                x1 = drop_feature_weighted_2(
                    ft_list, feature_weights, drop_feature_rate_view1, threshold=0.7
                )
                x2 = drop_feature_weighted_2(
                    ft_list, feature_weights, drop_feature_rate_view2, threshold=0.7
                )

                optimizer.zero_grad()
                pred = model(x1, adj1, M1, x2, adj2, M2, pairs)
                loss_train = criterion(pred, pair_labels)
                loss_train.backward()
                optimizer.step()

                epoch_loss += loss_train.item()

            duration = (datetime.now() - start_epoch_time).seconds
            logger.info(
                f"Epoch {epoch}/{args.epochs} | Avg Loss: {epoch_loss / len(names):.4f} | Duration: {duration}s"
            )

        # Final Evaluation
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for name in names:
                (
                    ft_list,
                    data,
                    adj,
                    M,
                    pairs,
                    pair_labels,
                    edge_weights,
                    feature_weights,
                ) = graph_data[name]

                x1 = drop_feature_weighted_2(
                    ft_list, feature_weights, drop_feature_rate_view1, threshold=0.7
                )
                x2 = drop_feature_weighted_2(
                    ft_list, feature_weights, drop_feature_rate_view2, threshold=0.7
                )
                adj1 = get_adj(data.edge_index, data.num_nodes)
                M1 = get_M(adj1, t=2)

                pred = model(x1, adj1, M1, x2, adj1, M1, pairs)
                pred_labels = (pred.sigmoid() > 0.5).long()
                all_preds.append(pred_labels.cpu())
                all_labels.append(pair_labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        pre, rec, f1 = evaluate(all_preds, all_labels)
        logger.info(f"AVERAGE: Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        os.makedirs(".expert_record", exist_ok=True)
        with open(
            join(".expert_record", args.predict_result), "a", encoding="utf-8"
        ) as f:
            msg = f"combin_num: {combin_num}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}\n"
            logger.info(msg)
            f.write(msg)

        return model.encoder


def evaluate(pred_labels: torch.Tensor, true_labels: torch.Tensor):
    pred_labels = pred_labels.view(-1).cpu()
    true_labels = true_labels.view(-1).cpu()

    tp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1
