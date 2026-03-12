from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


GLOBAL_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    reg_lambda=1.0,
    reg_alpha=0.0,
)

LOCAL_XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    reg_lambda=1.0,
    reg_alpha=0.0,
)


@dataclass
class ModelArtifacts:
    global_xgb: Any
    cluster_models: list[Any | None]
    cluster_use_global: np.ndarray


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_models(
    x_scaled_clean: np.ndarray,
    y_clean: np.ndarray,
    clusters: np.ndarray,
    k_final: int,
    train_mark: np.ndarray,
    min_samples_per_cluster: int,
    logger: logging.Logger,
) -> ModelArtifacts:
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

    global_xgb = XGBRegressor(**GLOBAL_XGB_PARAMS)
    global_xgb.fit(x_scaled_clean[train_mark], y_clean[train_mark])

    cluster_models: list[XGBRegressor | None] = [None] * k_final
    cluster_use_global = np.zeros(k_final, dtype=np.int32)

    for k in range(k_final):
        idx = np.where((clusters == k) & train_mark)[0]
        n_k = len(idx)
        logger.info("[Cluster %d] n = %d", k, n_k)
        if n_k < min_samples_per_cluster:
            logger.info("  -> use global model (n < %d)", min_samples_per_cluster)
            cluster_use_global[k] = 1
            continue

        x_k = x_scaled_clean[idx]
        y_k = y_clean[idx]
        model = XGBRegressor(**LOCAL_XGB_PARAMS)
        model.fit(x_k, y_k)
        y_k_pred = model.predict(x_k)
        logger.info(
            "  -> local XGB trained, R2=%.4f, RMSE=%.4f",
            r2_score(y_k, y_k_pred),
            rmse(y_k, y_k_pred),
        )
        cluster_models[k] = model
    return ModelArtifacts(
        global_xgb=global_xgb,
        cluster_models=cluster_models,
        cluster_use_global=cluster_use_global,
    )


def predict_delay_batch(
    x_raw: np.ndarray,
    mu: np.ndarray,
    inv_sigma: np.ndarray,
    pls,
    centroids: np.ndarray,
    artifacts: ModelArtifacts,
) -> np.ndarray:
    x_scaled = (x_raw - mu) * inv_sigma
    x_proj = pls.transform(x_scaled)
    dists = np.sum((x_proj[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    selected_cluster = np.argmin(dists, axis=1)

    y_pred = np.empty(x_raw.shape[0], dtype=np.float64)
    use_global_mask = artifacts.cluster_use_global[selected_cluster].astype(bool)
    if np.any(use_global_mask):
        y_pred[use_global_mask] = artifacts.global_xgb.predict(x_scaled[use_global_mask])

    unique_local_clusters = np.unique(selected_cluster[~use_global_mask])
    for cid in unique_local_clusters:
        model = artifacts.cluster_models[int(cid)]
        if model is None:
            cid_mask = selected_cluster == cid
            y_pred[cid_mask] = artifacts.global_xgb.predict(x_scaled[cid_mask])
            continue
        cid_mask = selected_cluster == cid
        y_pred[cid_mask] = model.predict(x_scaled[cid_mask])
    return y_pred
