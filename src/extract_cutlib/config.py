from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractCutLibConfig:
    data_path: str
    dataset_path: str
    k: int
    latent_dim: int
    min_cluster: int
    train_ratio: float
    split_method: str
    auto_k: bool
    k_min: int
    k_max: int | None
    k_metric: str
    keep_ratio_min: float
    eval_pls_effect: bool
    random_seed: int = 42
