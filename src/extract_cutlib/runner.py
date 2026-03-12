from __future__ import annotations

import time
import traceback

from sklearn.metrics import r2_score
import gc

from subutils import extraction_utils as ext_utils

from .clustering import clean_and_recluster, initial_clustering, select_best_k
from .config import ExtractCutLibConfig
from .logger import get_logger
from .modeling import predict_delay_batch, rmse, train_models
from .pls_effect_eval import evaluate_pls_effect
from .preprocess import fit_pls_projection, seed_all, standardize_features


def run(cfg: ExtractCutLibConfig) -> None:
    logger = get_logger()
    inside_logger = None
    try:
        seed_all(cfg.random_seed)

        # load openABC-D
        logger.info("Loading data from %s", cfg.data_path)
        data_path = f"{cfg.dataset_path}/save_json_OpenABC"
        x_openabc, x_graph_openabc, y_openabc = ext_utils.load_json_data(data_path)
        logger.info("Data loaded from openABC-D: %d samples", len(y_openabc))
        # load ACE
        data_path = f"{cfg.dataset_path}/save_json_ACE"
        x_ace, x_graph_ace, y_ace = ext_utils.load_json_data(data_path)
        logger.info("Data loaded from ACE: %d samples", len(y_ace))
        # load iwls2022
        data_path = f"{cfg.dataset_path}/save_json_iwls2022"
        x_iwls, x_graph_iwls, y_iwls = ext_utils.load_json_data(data_path)
        logger.info("Data loaded from iwls2022: %d samples", len(y_iwls))
        # load iwls2024
        data_path = f"{cfg.dataset_path}/save_json_iwls2024"
        x_iwls2024, x_graph_iwls2024, y_iwls2024 = ext_utils.load_json_data(data_path)
        logger.info("Data loaded from iwls2024: %d samples", len(y_iwls2024))

        x_raw = ext_utils.np.concatenate([x_openabc, x_ace, x_iwls, x_iwls2024], axis=0)
        y = ext_utils.np.concatenate([y_openabc, y_ace, y_iwls, y_iwls2024], axis=0)
        del x_openabc, y_openabc, x_ace, y_ace, x_iwls, y_iwls, x_iwls2024, y_iwls2024
        ext_utils.torch.cuda.empty_cache()
        gc.collect()

        y = y.reshape(-1, 1)

        x_cont_full = x_raw[:, :-6]
        perm = ext_utils.np.random.permutation(len(y))
        x_cont_full = x_cont_full[perm]
        y = y[perm]

        if cfg.split_method == "across_cluster":
            train_len = int(len(y) * cfg.train_ratio)
            x_cont_train = x_cont_full[:train_len]
            y_train = y[:train_len]
            x_cont_test = x_cont_full[train_len:]
            y_test = y[train_len:]
            x_unnorm = x_cont_train
            y_used = y_train
        else:
            x_unnorm = x_cont_full
            y_used = y

        x_scaled, _, _ = standardize_features(x_unnorm)
        _, x_proj = fit_pls_projection(x_scaled, y_used, cfg.latent_dim)

        if cfg.auto_k:
            _, cleaned = select_best_k(x_unnorm, y_used, x_proj, cfg, logger)
        else:
            cluster_ids0, centers0 = initial_clustering(x_proj, cfg.k)
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

        artifacts = train_models(
            x_scaled_clean=cleaned.x_scaled_clean,
            y_clean=cleaned.y_clean,
            clusters=cleaned.clusters,
            k_final=cleaned.k_final,
            train_mark=cleaned.train_mark,
            min_samples_per_cluster=cfg.min_cluster,
            logger=logger,
        )

        if cfg.split_method == "inside_cluster":
            x_cont_test = cleaned.x_cont_clean[~cleaned.train_mark]
            y_test = cleaned.y_clean[~cleaned.train_mark]

        start = time.perf_counter()
        y_pred = predict_delay_batch(
            x_raw=x_cont_test,
            mu=cleaned.mu,
            inv_sigma=cleaned.inv_sigma,
            pls=cleaned.pls,
            centroids=cleaned.centroids,
            artifacts=artifacts,
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Samples: %d, Time elapsed: %.2f seconds, Avg time per sample: %.2f us",
            len(y_test),
            elapsed,
            elapsed / len(y_test) * 1e6,
        )
        logger.info(
            "[Overall] R2 = %.4f, RMSE = %.4f",
            r2_score(y_test, y_pred),
            rmse(y_test, y_pred),
        )
        if cfg.eval_pls_effect:
            evaluate_pls_effect(cfg, cleaned, x_cont_test, y_test, y_pred, logger)
    except Exception as exc:  # noqa: BLE001
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno if exc.__traceback__ else -1
        file = traceback.extract_tb(exc.__traceback__)[-1].filename if exc.__traceback__ else "unknown"
        logger.error("An error occurred at line %d in %s: %s", line, file, exc)
