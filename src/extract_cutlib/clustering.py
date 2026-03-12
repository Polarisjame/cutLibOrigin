from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from subutils.extraction_utils import filter_small_clusters_and_outliers

from .config import ExtractCutLibConfig
from .preprocess import fit_pls_projection, standardize_features


@dataclass
class CleanedData:
    x_cont_clean: np.ndarray
    y_clean: np.ndarray
    x_scaled_clean: np.ndarray
    x_proj_clean: np.ndarray
    mu: np.ndarray
    inv_sigma: np.ndarray
    pls: PLSRegression
    clusters: np.ndarray
    centroids: np.ndarray
    k_final: int
    train_mark: np.ndarray
    keep_mask: np.ndarray


def initial_clustering(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    cluster_ids = kmeans.fit_predict(z)
    return cluster_ids, kmeans.cluster_centers_


def clean_and_recluster(
    x_cont: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cluster_ids0: np.ndarray,
    centers0: np.ndarray,
    min_samples_per_cluster: int,
    latent_dim: int,
    split_inside_cluster: bool,
    train_ratio: float,
    logger: logging.Logger,
) -> CleanedData:
    keep_mask, valid_clusters, counts, min_keep = filter_small_clusters_and_outliers(
        Z=z,
        y=y,
        cluster_ids=cluster_ids0,
        centers=centers0,
        min_abs=min_samples_per_cluster,
        min_rel_to_median=0.2,
        dist_mad_k=2.5,
        y_mad_k=2.5,
        outlier_mode="or",
    )

    logger.info("Cluster counts: %s", counts)
    logger.info("min_keep = %s", min_keep)
    logger.info("valid_clusters = %s", valid_clusters)
    logger.info("kept samples = %d / %d", int(keep_mask.sum()), len(keep_mask))

    x_cont_clean = x_cont[keep_mask]
    y_clean = y[keep_mask]
    x_scaled_clean, mu, inv_sigma = standardize_features(x_cont_clean)
    pls, x_proj_clean = fit_pls_projection(x_scaled_clean, y_clean, latent_dim)

    k_valid = len(valid_clusters)
    kmeans = KMeans(n_clusters=k_valid, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(x_proj_clean)
    centroids = kmeans.cluster_centers_
    train_mark = np.ones(len(y_clean), dtype=bool)

    if split_inside_cluster:
        cluster_to_indices = {i: [] for i in range(k_valid)}
        for idx, cid in enumerate(clusters):
            cluster_to_indices[cid].append(idx)
        for cid in range(k_valid):
            idxs = cluster_to_indices[cid]
            np.random.shuffle(idxs)
            n_train = int(len(idxs) * train_ratio)
            for idx in idxs[n_train:]:
                train_mark[idx] = False

    counts_clean = np.bincount(clusters, minlength=k_valid)
    k_final = int(clusters.max()) + 1
    logger.info("After merge: K' = %d", k_final)
    logger.info("Counts per cluster: %s", counts_clean)

    return CleanedData(
        x_cont_clean=x_cont_clean,
        y_clean=y_clean,
        x_scaled_clean=x_scaled_clean,
        x_proj_clean=x_proj_clean,
        mu=mu,
        inv_sigma=inv_sigma,
        pls=pls,
        clusters=clusters,
        centroids=centroids,
        k_final=k_final,
        train_mark=train_mark.astype(bool),
        keep_mask=keep_mask,
    )


def _k_metric_score(metric: str, z: np.ndarray, labels: np.ndarray) -> float:
    if metric == "silhouette":
        return silhouette_score(z, labels)
    if metric == "db":
        return davies_bouldin_score(z, labels)
    if metric == "ch":
        return calinski_harabasz_score(z, labels)
    raise ValueError(f"Unknown metric: {metric}")


def select_best_k(
    x_unnorm: np.ndarray,
    y_used: np.ndarray,
    x_proj: np.ndarray,
    cfg: ExtractCutLibConfig,
    logger: logging.Logger,
) -> tuple[int, CleanedData]:
    n_samples = len(y_used)
    k_max_default = max(2, n_samples // max(1, cfg.min_cluster))
    k_max = cfg.k_max if cfg.k_max is not None else min(cfg.k, k_max_default)
    k_min = max(2, cfg.k_min)
    if k_max < k_min:
        k_max = k_min

    best: tuple[int, float, CleanedData] | None = None
    for k in range(k_min, k_max + 1):
        cluster_ids0, centers0 = initial_clustering(x_proj, k)
        cleaned = clean_and_recluster(
            x_cont=x_unnorm,
            y=y_used,
            z=x_proj,
            cluster_ids0=cluster_ids0,
            centers0=centers0,
            min_samples_per_cluster=cfg.min_cluster,
            latent_dim=cfg.latent_dim,
            split_inside_cluster=(cfg.split_method == "inside_cluster"),
            train_ratio=cfg.train_ratio,
            logger=logger,
        )

        keep_ratio = len(cleaned.y_clean) / n_samples
        if keep_ratio < cfg.keep_ratio_min or cleaned.k_final < 2:
            logger.info("[AutoK] K=%d skipped (keep_ratio=%.3f, K'=%d)", k, keep_ratio, cleaned.k_final)
            continue
        if len(cleaned.clusters) <= cleaned.k_final:
            logger.info("[AutoK] K=%d skipped (insufficient samples)", k)
            continue

        try:
            score = _k_metric_score(cfg.k_metric, cleaned.x_proj_clean, cleaned.clusters)
        except Exception as exc:  # noqa: BLE001
            logger.info("[AutoK] K=%d metric failed: %s", k, exc)
            continue

        logger.info(
            "[AutoK] K=%d, K'=%d, keep_ratio=%.3f, %s=%.4f",
            k,
            cleaned.k_final,
            keep_ratio,
            cfg.k_metric,
            score,
        )

        if best is None:
            best = (k, score, cleaned)
            continue
        best_k, best_score, _ = best
        better = score > best_score if cfg.k_metric in ("silhouette", "ch") else score < best_score
        if better or (score == best_score and k < best_k):
            best = (k, score, cleaned)

    if best is None:
        raise RuntimeError("Auto K selection failed: no valid K candidates.")
    best_k, best_score, cleaned = best
    logger.info("[AutoK] Selected K=%d, K'=%d, %s=%.4f", best_k, cleaned.k_final, cfg.k_metric, best_score)
    return best_k, cleaned
