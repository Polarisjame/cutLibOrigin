from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from .clustering import CleanedData
from .config import ExtractCutLibConfig
from .modeling import rmse, train_models


def _cluster_y_concentration(y: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    y_1d = np.asarray(y).reshape(-1)
    labels = np.asarray(labels)
    total = len(y_1d)
    if total == 0:
        return np.nan, np.nan

    weighted_std = 0.0
    weighted_mad = 0.0
    for cid in range(k):
        idx = labels == cid
        if not np.any(idx):
            continue
        yk = y_1d[idx]
        n_k = len(yk)
        weighted_std += float(np.std(yk)) * n_k
        med = float(np.median(yk))
        weighted_mad += float(np.mean(np.abs(yk - med))) * n_k
    return weighted_std / total, weighted_mad / total


def _safe_improve(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1e-12:
        return np.nan
    return (before - after) / before * 100.0


class _NoOpLogger:
    def info(self, *_args, **_kwargs) -> None:
        return


def _predict_delay_batch_without_pls(
    x_raw: np.ndarray,
    mu: np.ndarray,
    inv_sigma: np.ndarray,
    centroids: np.ndarray,
    artifacts,
) -> np.ndarray:
    x_scaled = (x_raw - mu) * inv_sigma
    dists = np.sum((x_scaled[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    selected_cluster = np.argmin(dists, axis=1)

    y_pred = np.empty(x_raw.shape[0], dtype=np.float64)
    use_global_mask = artifacts.cluster_use_global[selected_cluster].astype(bool)
    if np.any(use_global_mask):
        y_pred[use_global_mask] = artifacts.global_xgb.predict(x_scaled[use_global_mask])

    unique_local_clusters = np.unique(selected_cluster[~use_global_mask])
    for cid in unique_local_clusters:
        model = artifacts.cluster_models[int(cid)]
        cid_mask = selected_cluster == cid
        if model is None:
            y_pred[cid_mask] = artifacts.global_xgb.predict(x_scaled[cid_mask])
            continue
        y_pred[cid_mask] = model.predict(x_scaled[cid_mask])
    return y_pred


def evaluate_pls_effect(
    cfg: ExtractCutLibConfig,
    cleaned: CleanedData,
    x_cont_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_with_pls: np.ndarray,
    logger,
) -> None:
    if len(cleaned.y_clean) <= cleaned.k_final:
        logger.info("[PLS Eval] skipped: sample size (%d) <= K' (%d).", len(cleaned.y_clean), cleaned.k_final)
        return

    kmeans_no_pls = KMeans(n_clusters=cleaned.k_final, random_state=cfg.random_seed, n_init="auto")
    labels_no_pls = kmeans_no_pls.fit_predict(cleaned.x_scaled_clean)
    labels_pls = cleaned.clusters

    std_no_pls, mad_no_pls = _cluster_y_concentration(cleaned.y_clean, labels_no_pls, cleaned.k_final)
    std_pls, mad_pls = _cluster_y_concentration(cleaned.y_clean, labels_pls, cleaned.k_final)
    logger.info(
        "[PLS Eval] y concentration (lower is better): weighted_std %.6f -> %.6f (improve %.2f%%), "
        "weighted_mad_to_median %.6f -> %.6f (improve %.2f%%)",
        std_no_pls,
        std_pls,
        _safe_improve(std_no_pls, std_pls),
        mad_no_pls,
        mad_pls,
        _safe_improve(mad_no_pls, mad_pls),
    )

    artifacts_no_pls = train_models(
        x_scaled_clean=cleaned.x_scaled_clean,
        y_clean=cleaned.y_clean,
        clusters=labels_no_pls,
        k_final=cleaned.k_final,
        train_mark=cleaned.train_mark,
        min_samples_per_cluster=cfg.min_cluster,
        logger=_NoOpLogger(),
    )
    y_pred_no_pls = _predict_delay_batch_without_pls(
        x_raw=x_cont_test,
        mu=cleaned.mu,
        inv_sigma=cleaned.inv_sigma,
        centroids=kmeans_no_pls.cluster_centers_,
        artifacts=artifacts_no_pls,
    )

    r2_with_pls = r2_score(y_test, y_pred_with_pls)
    rmse_with_pls = rmse(y_test, y_pred_with_pls)
    r2_without_pls = r2_score(y_test, y_pred_no_pls)
    rmse_without_pls = rmse(y_test, y_pred_no_pls)
    logger.info(
        "[PLS Eval] End-to-end fit on same test set: "
        "with_pls(R2=%.4f, RMSE=%.4f) vs without_pls(R2=%.4f, RMSE=%.4f), "
        "delta(R2=%.4f, RMSE=%.4f)",
        r2_with_pls,
        rmse_with_pls,
        r2_without_pls,
        rmse_without_pls,
        r2_with_pls - r2_without_pls,
        rmse_with_pls - rmse_without_pls,
    )
