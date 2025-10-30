import time
from datetime import datetime, timedelta
from torch_geometric.data import Data
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
from dataset.enhance_graph import *
from dataset.load_data import load_dataset, load_graph, load_json
from dataset.save_results import get_results
from evaluation.eval import evaluate
from model.mccg_model import MCCG, GAT
from .utils import *
from os.path import join
from params import set_params
from .refine import update_edges_by_cosine

_, args = set_params()

device = torch.device(
    ("cuda:" + str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu"
)

print("Device:", device)
print("=" * 15, "\n")


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
        t_multiview,
        t_cluster,
        refine=True,
    ):

        names, pubs = load_dataset(mode)
        results = {}

        for p, name in enumerate(names, 1):
            start_time = datetime.now()
            logger.info(
                f"Training {p}/{len(names)}: {name} | Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            name_start = time.time()

            results[name] = []

            label, ft_list, data = load_graph(name, mode, th_a, th_o, th_v)
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
            )
            model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=l2_coef
            )

            if refine:
                # ones for initial weight matrix for nodes that have edges
                weight_matrix = torch.zeros(data.num_nodes, data.num_nodes).to(device)
                for i, j in data.edge_index.t():
                    weight_matrix[i, j] = 1.0
                    weight_matrix[j, i] = 1.0
                weight_matrix_1 = weight_matrix.clone()
                weight_matrix_2 = weight_matrix.clone()

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()

                data1 = Data(x=x1, edge_index=edge_index1)
                data2 = Data(x=x2, edge_index=edge_index2)

                if refine:
                    data1, weight_matrix_1 = update_edges_by_cosine(
                        data1, weight_matrix=weight_matrix_1
                    )
                    data2, weight_matrix_2 = update_edges_by_cosine(
                        data2, weight_matrix=weight_matrix_2
                    )

                data1 = data1.to(device)
                data2 = data2.to(device)

                adj1 = get_adj(data1.edge_index, data1.num_nodes)
                adj2 = get_adj(data2.edge_index, data2.num_nodes)
                M1 = get_M(adj1, t=2)
                M2 = get_M(adj2, t=2)

                embd_multiview, embd_cluster = model(
                    data1.x, adj1, M1, data2.x, adj2, M2
                )

                dis = pairwise_distances(
                    embd_cluster.cpu().detach().numpy(), metric="cosine"
                )
                labels = hdbscan.HDBSCAN(
                    cluster_selection_epsilon=db_eps,
                    min_samples=db_min,
                    min_cluster_size=db_min,
                    metric="precomputed",
                ).fit_predict(dis.astype("double"))
                labels = torch.from_numpy(labels).to(device)

                loss_cluster = model.SelfSupConLoss(
                    embd_cluster.unsqueeze(1),
                    labels,
                    contrast_mode="one",
                    temperature=t_cluster,
                )
                loss_multiview = model.SelfSupConLoss(
                    embd_multiview,
                    labels,
                    contrast_mode="all",
                    temperature=t_multiview,
                )

                w_multiview = 1 - w_cluster
                loss_train = w_cluster * loss_cluster + w_multiview * loss_multiview

                loss_train.backward()
                optimizer.step()

                if epoch == args.epochs:
                    name_end = time.time()
                    duration_seconds = name_end - name_start
                    formatted_duration = str(timedelta(seconds=int(duration_seconds)))

                    logger.info(
                        f"Epochs: {epoch}/{args.epochs} | Runtime: {formatted_duration} | "
                        f"MultiView Loss: {loss_multiview.item():.4f} | "
                        f"Cluster Loss: {loss_cluster.item():.4f} | Total Loss: {loss_train.item():.4f}"
                        f"\n{'-'*100}"
                    )

            with torch.no_grad():
                model.eval()
                embd = model.encoder(ft_list, adj, M)
                embd = F.normalize(model.cluster_projector(embd), dim=1)
                lc_dis = pairwise_distances(
                    embd.cpu().detach().numpy(), metric="cosine"
                )
                labels = hdbscan.HDBSCAN(
                    cluster_selection_epsilon=db_eps,
                    min_samples=db_min,
                    min_cluster_size=db_min,
                    metric="precomputed",
                ).fit_predict(lc_dis.astype("double"))

                class_matrix = torch.from_numpy(onehot_encoder(labels))
                labels = torch.mm(class_matrix, class_matrix.t())
                pred = matx2list(labels)

                results[name] = pred

                predict = get_results([name], pubs, results)

                pre, rec, f1 = evaluate(
                    predict, args.ground_truth_file, print_names=True
                )

        predict = get_results(names, pubs, results)

        pre, rec, f1 = evaluate(predict, args.ground_truth_file, print_names=True)

        logger.info(
            f"[{mode.upper()}] AVERAGE: Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
        )

        with open(
            join(".expert_record", args.predict_result), "a", encoding="utf-8"
        ) as f:
            msg = f"combin_num: {combin_num}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}\n"
            logger.info(msg)
            f.write(msg)
