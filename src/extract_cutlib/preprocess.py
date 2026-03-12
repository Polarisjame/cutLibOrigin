from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-12
    inv_sigma = 1.0 / sigma
    scaled = (features - mu) * inv_sigma
    return scaled, mu, inv_sigma


def fit_pls_projection(x_scaled: np.ndarray, y: np.ndarray, latent_dim: int) -> tuple[PLSRegression, np.ndarray]:
    pls = PLSRegression(n_components=latent_dim)
    x_proj = pls.fit_transform(x_scaled, y)[0]
    return pls, x_proj
