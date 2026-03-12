from __future__ import annotations

import argparse

from .config import ExtractCutLibConfig


def parse_args() -> ExtractCutLibConfig:
    parser = argparse.ArgumentParser(description="Extract cutLib data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--K", type=int, required=True, help="Cluster number (or max when --auto_k)")
    parser.add_argument("--latent_dim", type=int, required=True, help="PLS latent dimensions")
    parser.add_argument("--min_cluster", type=int, required=True, help="Minimum samples per cluster")
    parser.add_argument("--train_ratio", type=float, required=True, help="Training ratio")
    parser.add_argument(
        "--split_method",
        type=str,
        choices=["inside_cluster", "across_cluster"],
        default="inside_cluster",
        help="Data split strategy",
    )
    parser.add_argument("--auto_k", action="store_true", help="Auto-select K")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum K for auto selection")
    parser.add_argument("--k_max", type=int, default=None, help="Maximum K for auto selection")
    parser.add_argument(
        "--k_metric",
        type=str,
        choices=["silhouette", "db", "ch"],
        default="db",
        help="Metric for auto K selection",
    )
    parser.add_argument(
        "--keep_ratio_min",
        type=float,
        default=0.5,
        help="Minimum keep ratio after cleaning in auto-K",
    )
    parser.add_argument(
        "--eval_pls_effect",
        action="store_true",
        help="Evaluate whether PLS makes within-cluster y distribution more concentrated",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return ExtractCutLibConfig(
        data_path=args.data_path,
        dataset_path=args.dataset_path,
        k=args.K,
        latent_dim=args.latent_dim,
        min_cluster=args.min_cluster,
        train_ratio=args.train_ratio,
        split_method=args.split_method,
        auto_k=args.auto_k,
        k_min=args.k_min,
        k_max=args.k_max,
        k_metric=args.k_metric,
        keep_ratio_min=args.keep_ratio_min,
        eval_pls_effect=args.eval_pls_effect,
        random_seed=args.random_seed,
    )
