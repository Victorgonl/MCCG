import os
import torch
import numpy as np
from os.path import join
from tqdm import tqdm
from dataset.load_data import load_data, load_json
from dataset.save_results import check_mkdir
from params import set_params
from itertools import combinations
from os.path import join

_, args = set_params()


def gen_relations(name, mode, target):
    dirpath = join(args.save_path, "relations", mode, name)

    temp = set()
    paper_info = dict()

    if target == "author":
        filename = "paper_author.txt"
    elif target == "org":
        filename = "paper_org.txt"
    elif target == "venue":
        filename = "paper_venue.txt"

    with open(join(dirpath, filename), "r", encoding="utf-8") as f:
        for line in f:
            temp.add(line)

    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_info:
                paper_info[p] = []
            paper_info[p].append(a)

    temp.clear()

    return paper_info


def save_label_pubs(mode, name, raw_pubs, save_path):
    if mode == "train":
        label_dict = {}
        pubs = []
        ilabel = 0
        for aid in raw_pubs[name]:
            pubs.extend(raw_pubs[name][aid])
            for pid in raw_pubs[name][aid]:
                label_dict[pid] = ilabel
            ilabel += 1

        file_path = join(save_path, "p_label.npy")
        np.save(file_path, label_dict)
    else:
        pubs = []
        for pid in raw_pubs[name]:
            pubs.append(pid)

    return pubs


""" def save_graph(name, pubs, save_path, mode):
    # init node mapping & edge mapping
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    cp_a, cp_o = set(), set()

    paper_rel_ath = gen_relations(name, mode, 'author')
    paper_rel_org = gen_relations(name, mode, 'org')
    paper_rel_ven = gen_relations(name, mode, 'venue')
    for pid in paper_dict:
        if pid not in paper_rel_ath:
            cp_a.add(paper_dict[pid])
    for pid in paper_dict:
        if pid not in paper_rel_org:
            cp_o.add(paper_dict[pid])

    # mark paper w/o coauthor and coorg as outlier
    cp = cp_a & cp_o

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:

        for p1 in paper_dict:
            p1_idx = paper_dict[p1]
            for p2 in paper_dict:
                p2_idx = paper_dict[p2]
                if p1 != p2:
                    co_aths, co_orgs, co_vens = 0, 0, 0
                    org_attr, org_attr_jaccard, org_jaccard2, ven_attr, ven_attr_jaccard, venue_jaccard2 = 0, 0, 0, 0, 0, 0
                    if p1 in paper_rel_ath:
                        for k in paper_rel_ath[p1]:
                            if p2 in paper_rel_ath:
                                if k in paper_rel_ath[p2]:
                                    co_aths += 1

                    if p1 in paper_rel_org:
                        for k in paper_rel_org[p1]:
                            if p2 in paper_rel_org:
                                if k in paper_rel_org[p2]:
                                    co_orgs += 1

                    if p1 in paper_rel_ven:
                        for k in paper_rel_ven[p1]:
                            if p2 in paper_rel_ven:
                                if k in paper_rel_ven[p2]:
                                    co_vens += 1

                    if co_orgs > 0:
                        all_words_p1 = len(paper_rel_org[p1])
                        all_words_p2 = len(paper_rel_org[p2])
                        org_attr = co_orgs / max(all_words_p1, all_words_p2)
                        org_attr_jaccard = co_orgs / (all_words_p1 + all_words_p2 - co_orgs)

                    if co_vens > 0:
                        all_words_p1 = len(paper_rel_ven[p1])
                        all_words_p2 = len(paper_rel_ven[p2])
                        ven_attr = co_vens / max(all_words_p1, all_words_p2)
                        ven_attr_jaccard = co_vens / (all_words_p1 + all_words_p2 - co_vens)

                    if (co_aths + co_orgs) > 0:
                        f.write(f'{p1_idx}\t{p2_idx}\t{co_aths}\t'
                                f'{co_orgs}\t{org_attr_jaccard}\t'
                                f'{co_vens}\t{ven_attr_jaccard}\n')

    f.close()

    with open(join(save_path, 'rel_cp.txt'), 'w') as out_f:
        for i in cp:
            out_f.write(f'{i}\n')
    out_f.close() """


def save_graph(name, pubs, save_path, mode, loop=None):
    if loop:
        loop.set_postfix(name=name, step="2/3: Saving graph")

    assert pubs, f"[save_graph] Empty `pubs` list for {name}, cannot build graph."

    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}
    assert paper_dict, f"[save_graph] No valid papers for {name}"

    cp_a, cp_o = set(), set()

    paper_rel_ath = gen_relations(name, mode, "author")
    paper_rel_org = gen_relations(name, mode, "org")
    paper_rel_ven = gen_relations(name, mode, "venue")

    for pid in paper_dict:
        if pid not in paper_rel_ath:
            cp_a.add(paper_dict[pid])
        if pid not in paper_rel_org:
            cp_o.add(paper_dict[pid])

    cp = cp_a & cp_o

    adj_attr_file = join(save_path, "adj_attr.txt")
    with open(adj_attr_file, "w") as f:
        edge_count = 0
        for p1, p2 in combinations(paper_dict, 2):
            p1_idx = paper_dict[p1]
            p2_idx = paper_dict[p2]

            authors_p1 = set(paper_rel_ath.get(p1, []))
            authors_p2 = set(paper_rel_ath.get(p2, []))
            co_aths = len(authors_p1 & authors_p2)

            orgs_p1 = set(paper_rel_org.get(p1, []))
            orgs_p2 = set(paper_rel_org.get(p2, []))
            co_orgs = len(orgs_p1 & orgs_p2)

            vens_p1 = set(paper_rel_ven.get(p1, []))
            vens_p2 = set(paper_rel_ven.get(p2, []))
            co_vens = len(vens_p1 & vens_p2)

            if co_aths + co_orgs > 0:
                org_attr_jaccard = (
                    co_orgs / (len(orgs_p1 | orgs_p2)) if co_orgs > 0 else 0
                )
                ven_attr_jaccard = (
                    co_vens / (len(vens_p1 | vens_p2)) if co_vens > 0 else 0
                )

                f.write(
                    f"{p1_idx}\t{p2_idx}\t{co_aths}\t{co_orgs}\t"
                    f"{org_attr_jaccard}\t{co_vens}\t{ven_attr_jaccard}\n"
                )
                edge_count += 1

    assert edge_count > 0, f"[save_graph] No edges saved for {name}"

    with open(join(save_path, "rel_cp.txt"), "w") as out_f:
        for i in cp:
            out_f.write(f"{i}\n")


def save_emb(name, pubs, save_path):
    assert pubs, f"[save_emb] No publications found for {name}"

    # build mapping of paper & index-id
    mapping = {idx: pid for idx, pid in enumerate(pubs)}

    # load paper embedding
    ptext_emb = load_data(join(args.save_path, "paper_emb", name, "ptext_emb.pkl"))

    ft = dict()
    for pidx in mapping:
        pid = mapping[pidx]
        assert (
            pid in ptext_emb
        ), f"[save_emb] Missing embedding for paper ID {pid} in {name}"
        tensor = torch.from_numpy(ptext_emb[pid])
        assert (
            tensor.numel() > 0
        ), f"[save_emb] Empty embedding for paper ID {pid} in {name}"
        assert not torch.isnan(
            tensor
        ).any(), f"[save_emb] NaN found in embedding for paper ID {pid} in {name}"
        ft[pidx] = tensor

    assert ft, f"[save_emb] No features generated for {name}"

    feats_file_path = join(save_path, "feats_p.npy")
    np.save(feats_file_path, ft)


def build_graph(force_rebuild=False):
    for mode in ["train", "valid", "test"]:
        print("preprocess dataset: ", mode)
        data_base = join(args.save_path, "src")
        if mode == "train":
            raw_pubs = load_json(join(data_base, "train", "train_author.json"))
        elif mode == "valid":
            raw_pubs = load_json(join(data_base, "valid", "sna_valid_raw.json"))
        elif mode == "test":
            raw_pubs = load_json(join(data_base, "test", "sna_test_raw.json"))

        print(f"{mode} raw pubs loaded")

        loop = tqdm(raw_pubs)
        for name in loop:
            loop.set_postfix(name=name)

            save_path = join(args.save_path, "graph", mode, name)

            if (
                not force_rebuild
                and os.path.exists(save_path)
                and len(os.listdir(save_path)) > 0
            ):
                continue

            check_mkdir(save_path)
            loop.set_postfix(name=name, step="1/3: Saving label pubs")
            pubs = save_label_pubs(mode, name, raw_pubs, save_path)
            loop.set_postfix(name=name, step="2/3: Saving graph")
            save_graph(name, pubs, save_path, mode, loop=loop)
            loop.set_postfix(name=name, step="3/3: Saving embeddings")
            save_emb(name, pubs, save_path)


if __name__ == "__main__":
    build_graph()
