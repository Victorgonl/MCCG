import codecs
import os
import pickle
import re
import numpy as np
from tqdm import tqdm
from os.path import join
from gensim.models import word2vec
from params import set_params
from dataset.dump_graph import build_graph
from dataset.load_data import load_json
from dataset.save_results import dump_json, check_mkdir
from character.match_name import match_name

_, args = set_params()

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
             'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'universities', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing','journal', 'science',
                    'sciences', 'international', 'key', 'research', 'engineering','academy', 'state', 'center',
                    'xuebao', 'conference', 'proceedings', 'technology', 'jishu', 'ieee','acta', 'applied',
                    'letters', 'society', 'communications', 'daxue', 'sinica', 'yu', 'gongcheng','usa','xi',
                    'guangzhou','tianjing','pr','wuhan','chengdu','lanzhou','sichuan','dalian']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                   'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                   'p', 'results','people', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'ministry'
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system', 'sci', 'affiliated'
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                   'time', 'zhejiang', 'used', 'data', 'these', 'chemistry', 'chemical', 'physics', 'medical',
                   'hospital', 'national', 'information', 'beijing', 'lab', 'education','edu', 'ltd', 'co', ]


def save_pickle(data, *paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pubinfo(mode):
    """
    Read pubs' meta-information.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        pubs = load_json(join(base, "train", "train_pub.json"))
    elif mode == 'valid':
        pubs = load_json(join(base, "valid", "sna_valid_pub.json"))
    elif mode == 'test':
        pubs = load_json(join(base, 'test', 'sna_test_pub.json'))
    else:
        raise ValueError('choose right mode')

    return pubs


def read_raw_pubs(mode):
    """
    Read raw pubs.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        raw_pubs = load_json(join(base, "train", "train_author.json"))
    elif mode == 'valid':
        raw_pubs = load_json(join(base, "valid", "sna_valid_raw.json"))
    elif mode == 'test':
        raw_pubs = load_json(join(base, "test", "sna_test_raw.json"))
    else:
        raise ValueError('choose right mode')

    return raw_pubs


def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = ''.join(token[:])
        name_reverse = token[-1] + ''.join(token[:-1])

    return name, name_reverse

def dump_plain_texts_to_file(raw_data_root, processed_data_root):
    """
    Dump raw publication data to files with caching.
    """
    cache_path = os.path.join(processed_data_root, 'extract_text', 'plain_text.txt')
    if os.path.exists(cache_path):
        print(f'Cache found at {cache_path}, skipping processing.')
        return

    train_pubs_dict = load_json(os.path.join(raw_data_root, 'train', 'train_pub.json'))
    valid_pubs_dict = load_json(os.path.join(raw_data_root, 'valid', 'sna_valid_pub.json'))

    pubs_dict = {}
    pubs_dict.update(train_pubs_dict)
    pubs_dict.update(valid_pubs_dict)

    try:
        test_pubs_dict = load_json(os.path.join(raw_data_root, 'test', 'sna_test_pub.json'))
        pubs_dict.update(test_pubs_dict)
    except:
        pass

    texts_dir = os.path.join(processed_data_root, 'extract_text')
    os.makedirs(texts_dir, exist_ok=True)
    wf = codecs.open(cache_path, 'w', encoding='utf-8')

    for i, pid in enumerate(tqdm(pubs_dict)):
        paper_features = []
        pub = pubs_dict[pid]

        # Save title
        title = pub["title"]
        pstr = title.strip()
        pstr = pstr.lower()
        pstr = re.sub(puncs, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        title_features = pstr

        # Save keywords
        keywd_features = ""
        word_list = []
        if "keywords" in pub:
            for word in pub["keywords"]:
                word_list.append(word)
            pstr = " ".join(word_list)
            keywd_features = pstr

        org_list = []
        for author in pub["authors"]:
            # Save org (every author's organization)
            if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                if pstr:
                    org_list.append(pstr)

        pstr = " ".join(org_list)
        org_features = pstr

        # Save venue
        venue_features = ''
        if "venue" in pub and type(pub["venue"]) is str:
            venue = pub["venue"]
            pstr = venue.strip()
            pstr = pstr.lower()
            pstr = re.sub(puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            venue_features = pstr

        # Save abstract
        abstract_features = ''
        if "abstract" in pub and type(pub["abstract"]) is str:
            abstract = pub["abstract"]
            pstr = abstract.strip()
            pstr = pstr.lower()
            pstr = re.sub(puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            pstr = pstr.replace('\n', '')
            abstract_features = pstr

        paper_features.append(org_features + ' ' + title_features + ' ' + keywd_features + ' ' + venue_features + ' ' + abstract_features + ' ')
        wf.write(' '.join(paper_features) + '\n')

    print(f'All paper texts extracted and saved to cache.')
    wf.close()


def train_w2v_model(processed_data_root):
    texts_dir = join(processed_data_root, 'extract_text')
    model_path = join(processed_data_root, 'w2v_model')
    os.makedirs(model_path, exist_ok=True)
    model_file = join(model_path, 'tvt.model')

    if os.path.exists(model_file):
        model = word2vec.Word2Vec.load(model_file)
        print(f'Loaded cached word2vec model from {model_file}.')
        return model

    sentences = word2vec.Text8Corpus(join(texts_dir, 'plain_text.txt'))
    model = word2vec.Word2Vec(sentences, size=100, negative=5, min_count=5, window=5)
    model.save(model_file)
    print(f'Finish word2vec training and saved model to {model_file}.')
    return model


def dump_paper_emb(processed_data_root, model_name):
    """
    dump paper's [title, org, keywords] average word-embedding as semantic feature with caching.
    """
    model_path = join(processed_data_root, 'w2v_model')
    w2v_model = word2vec.Word2Vec.load(join(model_path, f'{model_name}.model'))

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in enumerate(tqdm(raw_pubs)):
            text_feature_path = join(processed_data_root, 'paper_emb', name)
            ptext_emb_file = join(text_feature_path, 'ptext_emb.pkl')
            tcp_file = join(text_feature_path, 'tcp.pkl')
            if os.path.exists(ptext_emb_file) and os.path.exists(tcp_file):
                continue

            name_pubs = load_json(join(processed_data_root, 'names_pub', mode, name + '.json'))
            os.makedirs(text_feature_path, exist_ok=True)

            ori_name = name
            name, name_reverse = unify_name_order(name)

            authorname_dict = {}
            ptext_emb = {}
            tcp = set()

            for i, pid in enumerate(name_pubs):
                pub = name_pubs[pid]
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                    taken = authorname.split(" ")
                    if len(taken) == 2:
                        authorname = taken[0] + taken[1]
                        authorname_reverse = taken[1] + taken[0]

                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        pass
                    else:
                        if "org" in author:
                            org = author["org"]
                            find_author = True
                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            if "org" in author:
                                org = author['org']
                                break

                pstr = ""
                keyword = ""
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        keyword = keyword + word + " "

                pstr = pub["title"] + " " + keyword + " " + org
                pstr = pstr.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 2]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]

                words_vec = []
                for word in pstr:
                    if word in w2v_model:
                        words_vec.append(w2v_model[word])
                if len(words_vec) < 1:
                    words_vec.append(np.zeros(100))
                    tcp.add(i)

                ptext_emb[pid] = np.mean(words_vec, 0)

            save_pickle(ptext_emb, ptext_emb_file)
            save_pickle(tcp, tcp_file)

    print("Finishing dump all paper embd into files.")


def dump_name_pubs():
    """
    Split publications informations by {name} and dump files as {name}.json with caching.
    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        pub_info = read_pubinfo(mode)
        file_path = join(args.save_path, 'names_pub', mode)
        if not os.path.exists(file_path):
            check_mkdir(file_path)
        for name in tqdm(raw_pubs):
            name_file = join(file_path, name + '.json')
            if os.path.exists(name_file):
                continue

            name_pubs_raw = {}
            if mode != "train":
                for i, pid in enumerate(raw_pubs[name]):
                    name_pubs_raw[pid] = pub_info[pid]
            else:
                pids = []
                for aid in raw_pubs[name]:
                    pids.extend(raw_pubs[name][aid])
                for pid in pids:
                    name_pubs_raw[pid] = pub_info[pid]

            dump_json(name_pubs_raw, name_file, indent=4)

    print("Finishing dump pubs according to names.")


def dump_features_relations_to_file():
    """
    Generate paper features and relations by raw publication data and dump to files with caching.
    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in tqdm(enumerate(raw_pubs)):

            file_path = join(args.save_path, 'relations', mode, name)
            cot_path = join(file_path, 'paper_title.txt')
            coa_path = join(file_path, 'paper_author.txt')
            cov_path = join(file_path, 'paper_venue.txt')
            coo_path = join(file_path, 'paper_org.txt')

            # Skip if all files exist (considered cached)
            if os.path.exists(cot_path) and os.path.exists(coa_path) and os.path.exists(cov_path) and os.path.exists(coo_path):
                continue

            check_mkdir(file_path)
            coa_file = open(coa_path, 'w', encoding='utf-8')
            cov_file = open(cov_path, 'w', encoding='utf-8')
            cot_file = open(cot_path, 'w', encoding='utf-8')
            coo_file = open(coo_path, 'w', encoding='utf-8')

            authorname_dict = {}
            pubs_dict = load_json(join(args.save_path, 'names_pub', mode, name + '.json'))

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
                pub = pubs_dict[pid]

                # Save title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + '\t' + word + '\n')

                # Save keywords
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(' +', ' ', pstr)
                keyword = pstr

                # Save org (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()

                    token = authorname.split(" ")
                    if len(token) == 2:
                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        coa_file.write(pid + '\t' + authorname + '\n')
                    else:
                        if "org" in author:
                            org = author["org"]
                            find_author = True

                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            if "org" in author:
                                org = author['org']
                                break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + '\t' + word + '\n')

                # Save venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    pstr = pstr.split(' ')
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + '\t' + word + '\n')
                    if len(pstr) == 0:
                        cov_file.write(pid + '\t' + 'null' + '\n')

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')


def preprocess_data():
    preprocessing_text= f"Preprocessing data: {args.save_path}"
    print(preprocessing_text)
    print("-" * len(preprocessing_text), "\n")
    print("-> Loading raw data")
    raw_data_root = join(args.save_path, 'src')
    processed_data_root = args.save_path
    print("-> Dumping plain text files")
    dump_plain_texts_to_file(raw_data_root, processed_data_root)
    dump_name_pubs()
    print("-> Training word2vec model")
    train_w2v_model(processed_data_root)
    print("-> Dumping papers embeddings")
    dump_paper_emb(processed_data_root, model_name='tvt')
    print("-> Dumping relations features to files")
    dump_features_relations_to_file()
    print("-> Building papers graph")
    build_graph(force_rebuild=False)


if __name__ == "__main__":
    """
    some pre-processing
    """
    preprocess_data()
