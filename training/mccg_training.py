import time
from datetime import datetime

import hdbscan
from numpy import indices
from sklearn.metrics.pairwise import pairwise_distances
from dataset.enhance_graph import *
from dataset.load_data import load_dataset, load_graph
from dataset.save_results import get_results
from evaluation.eval import evaluate
from model.mccg_model import MCCG, GAT
from .utils import *
from os.path import join
from params import set_params

_, args = set_params()

device = torch.device(
    ("cuda:" + str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu"
)


class MCCG_Trainer:
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
    w_diff,
    t_multiview,
    t_cluster,
    refine,
    semi_supervised=True, 
):
    train_names, train_pubs = load_dataset("train")
    eval_names, eval_pubs = load_dataset(mode)
    results = {}

    for p, name in enumerate(train_names, 1):        
        start_time = datetime.now()
        logger.info(
            f"Training {p}/{len(train_names)}: {name} | Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        name_start = time.time()

        label, ft_list, data = load_graph(name, "train", th_a, th_o, th_v)
        ft_list = ft_list.float().to(device)
        data = data.to(device)
        adj = get_adj(data.edge_index, data.num_nodes)
        M = get_M(adj, t=2)

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

        edge_index1 = drop_edge_weighted(
            data.edge_index, edge_weights, p=drop_edge_rate_view1, threshold=0.7
        )
        edge_index2 = drop_edge_weighted(
            data.edge_index, edge_weights, p=drop_edge_rate_view2, threshold=0.7
        )
        adj1 = get_adj(edge_index1, data.num_nodes)
        adj2 = get_adj(edge_index2, data.num_nodes)
        M1 = get_M(adj1, t=2)
        M2 = get_M(adj2, t=2)

        x1 = drop_feature_weighted_2(
            ft_list, feature_weights, drop_feature_rate_view1, threshold=0.7
        )
        x2 = drop_feature_weighted_2(
            ft_list, feature_weights, drop_feature_rate_view2, threshold=0.7
        )

        encoder = GAT(layer_shape[0], layer_shape[1], layer_shape[2])
        model = MCCG(
            encoder,
            dim_hidden=layer_shape[2],
            dim_proj_multiview=dim_proj_multiview,
            dim_proj_cluster=dim_proj_cluster,
            refine=refine,
        )
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=l2_coef
        )

        for epoch in range(1, args.epochs + 1):

            model.train()
            optimizer.zero_grad()

            # ======= Optional semi-supervised ground truth ============
            if semi_supervised and label is not None:
                diff_labels = label.to(device)
            else:
                diff_labels= None
            # ===========================================================

            embd_multiview, embd_cluster, loss_diff, pair_indices = model(
                x1, adj1, M1, x2, adj2, M2, labels=diff_labels
            )

            dis = pairwise_distances(
                embd_cluster.cpu().detach().numpy(), metric="cosine"
            )
            pred_labels = hdbscan.HDBSCAN(
                cluster_selection_epsilon=db_eps,
                min_samples=db_min,
                min_cluster_size=db_min,
                metric="precomputed",
            ).fit_predict(dis.astype("double"))

            # ======= Optional semi-supervised ground truth ============
            if semi_supervised and label is not None:
                labels = label.to(device)
            else:
                labels = torch.from_numpy(pred_labels).to(device)
            # ===========================================================

            loss_cluster = model.SelfSupConLoss(
                embd_cluster.unsqueeze(1),
                labels,
                contrast_mode="one",
                temperature=t_cluster,
            )
            loss_multiview = model.SelfSupConLoss(
                embd_multiview, labels, contrast_mode="all", temperature=t_multiview
            )

            w_multiview = 1 - w_cluster - w_diff
            loss_train = (
                w_cluster * loss_cluster
                + w_diff * loss_diff
                + w_multiview * loss_multiview
            )

            loss_train.backward()
            optimizer.step()

            if epoch == args.epochs:
                name_end = time.time()
                duration_seconds = name_end - name_start
                formatted_duration = str(datetime.timedelta(seconds=int(duration_seconds)))

                logger.info(
                    f"Epochs: {epoch}/{args.epochs} | Runtime: {formatted_duration} | "
                    f"Diff Loss: {loss_diff.item():.4f} (pairs: {pair_indices.size(0)}) | "
                    f"MultiView Loss: {loss_multiview.item():.4f} | "
                    f"Cluster Loss: {loss_cluster.item():.4f} | Total Loss: {loss_train.item():.4f}"
                    f"\n{'-'*100}"
                )

        eval_results = {}

        for ename in eval_names:
            _, eft_list, edata = load_graph(ename, mode, th_a, th_o, th_v)
            eft_list = eft_list.float().to(device)
            edata = edata.to(device)

            adj_eval = get_adj(edata.edge_index, edata.num_nodes)
            M_eval = get_M(adj_eval, t=2)

            model.eval()
            with torch.no_grad():
                embd = model.encoder(eft_list, adj_eval, M_eval)
                embd = F.normalize(model.cluster_projector(embd), dim=1)
                lc_dis = pairwise_distances(
                    embd.cpu().detach().numpy(), metric="cosine"
                )
                eval_labels = hdbscan.HDBSCAN(
                    cluster_selection_epsilon=db_eps,
                    min_samples=db_min,
                    min_cluster_size=db_min,
                    metric="precomputed",
                ).fit_predict(lc_dis.astype("double"))

                cm = torch.from_numpy(onehot_encoder(eval_labels))
                soft_labels = torch.mm(cm, cm.t())
                eval_results[ename] = matx2list(soft_labels)

        prediction = get_results(eval_names, eval_pubs, eval_results)
        pre, rec, f1 = evaluate(prediction, args.ground_truth_file, print_names=True)

        logger.info(
            f"[{mode.upper()}] AVERAGE: Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
        )

        with open(
            join(".expert_record", args.predict_result), "a", encoding="utf-8"
        ) as f:
            msg = f"combin_num: {combin_num}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}\n"
            logger.info(msg)
            f.write(msg)
