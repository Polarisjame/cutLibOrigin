import numpy as np
import os
import torch
import json

def load_data_from_infer(data):
    # data: list of [delay, feats]
    delays = []
    feats = []
    indexes = []
    for delay, f, index in data:
        delays.append(delay)
        feats.append(f)
        indexes.append(index)
    X = np.asarray(feats, dtype=np.float64)   # (N, D)
    y = np.asarray(delays, dtype=np.float64)  # (N,)
    y = y.reshape(-1, 1)  # (N, 1)
    return X, y, indexes

def load_data_from_subg(data_path, logger=None):
    x = []
    y = []
    for data_pt in os.listdir(data_path):
        if data_pt.endswith('.pt'):
            data = torch.load(os.path.join(data_path, data_pt))
            if logger:
                logger.info(f"Loading {data_pt} with {len(data)} samples")
            for item in data:
                feats = item['feats']
                label = item['lable'][:,2]
                x += feats.cpu().numpy().tolist()
                y += label.cpu().numpy().tolist()
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(-1, 1)
    return x, y

def load_json_data(data_path, logger=None):
    x = []
    x_graph = []
    y = []
    for case_name in os.listdir(data_path):
        json_path = os.path.join(data_path, case_name, 'data_2.json')

        with open(json_path, 'r') as f:
            data = json.load(f)
            for cut_id, cut_struct in data['cell_struct'].items():
                if cut_struct['supergateIndex'] == -1:
                    continue
                if logger:
                    logger.info(f"Loading cut {cut_id} from {json_path}")
                feats = cut_struct['feats']['feats']
                subg = cut_struct['feats']['subgraph']
                label = cut_struct['abc_delay']
                x.append(feats)
                x_graph.append(subg)
                y.append(label)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(-1, 1)
    return x, x_graph, y

def mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    m = np.median(x)
    return np.median(np.abs(x - m)) + 1e-12

def filter_small_clusters_and_outliers(
    Z, y, cluster_ids, centers,
    min_abs=300,
    min_rel_to_median=0.2,
    dist_mad_k=7.0,
    y_mad_k=7.0,
    outlier_mode="or",  # "or" or "and"
):
    """
    返回:
      keep_mask: (N,) bool  True 表示保留该样本
      valid_clusters: list[int] 保留的簇 id
    """
    Z = np.asarray(Z)
    y = np.asarray(y).ravel()
    cluster_ids = np.asarray(cluster_ids)

    K = centers.shape[0]
    counts = np.bincount(cluster_ids, minlength=K)
    med_cnt = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
    min_rel = int(np.ceil(min_rel_to_median * med_cnt))
    min_keep = max(min_abs, min_rel)

    valid = np.where(counts >= min_keep)[0].tolist()

    keep_mask = np.zeros(len(y), dtype=bool)

    for k in valid:
        idx = np.where(cluster_ids == k)[0]
        if len(idx) == 0:
            continue

        Zk = Z[idx]
        yk = y[idx]
        ck = centers[k]

        # (1) 距离异常
        dist = np.linalg.norm(Zk - ck, axis=1)
        dist_thr = np.median(dist) + dist_mad_k * mad(dist)
        bad_dist = dist > dist_thr

        # (2) delay 异常
        y_med = np.median(yk)
        y_thr = y_mad_k * mad(yk)
        bad_y = np.abs(yk - y_med) > y_thr

        if outlier_mode == "and":
            bad = bad_dist & bad_y
        else:
            bad = bad_dist | bad_y

        keep_mask[idx[~bad]] = True

    return keep_mask, valid, counts, min_keep
