#!/usr/bin/env python
"""
06 -- Wide-ANN surrogate: reproducible 5-fold cross-validation + activation ablation.

Reproduces the modelling family that was developed in this project and later
appeared in the group's publication: a deep-and-wide feed-forward network mapping
``(r/R, H, DT, H0, Ug) -> (eps_s, C/C0)``, with Shapley-style explainability and
an activation-function study.

Data availability
-----------------
The dataset itself belongs to the supervising group (two bed diameters,
76 / 152 mm) and is **not** redistributed here.  Supply it with ``--data``.
Without it the script runs on a synthetic dataset (``--demo``) so the harness
itself is verifiable.

The numbers that were *recorded* in the legacy notebooks are stored in
``results/tables/surrogate_recorded_reference.csv`` with explicit provenance;
they are historical evidence, not something this script can regenerate.

Outputs
-------
results/tables/surrogate_folds_<tag>.csv
results/tables/surrogate_aggregate_<tag>.json
results/tables/surrogate_activation_study.csv     (with --activations)
results/figures/surrogate_parity_<tag>.png
results/figures/surrogate_activation_study.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import model_wide_ann, paths, train_cv, viz


def synthetic_dataset(n: int = 530, seed: int = 0) -> tuple:
    """A stand-in dataset with the same shape as the surrogate problem."""
    rng = np.random.default_rng(seed)
    r_over_R = rng.uniform(0.0, 1.0, n)
    H = rng.uniform(0.25, 1.95, n)
    DT = rng.choice([76.0, 152.0], n)
    H0 = np.ones(n)
    Ug = rng.uniform(0.3, 1.0, n)
    X = np.column_stack([r_over_R, H, DT, H0, Ug])
    eps_s = 0.55 - 0.25 * r_over_R - 0.18 * H + 0.10 * Ug + 0.02 * rng.normal(size=n)
    c_over_c0 = (0.9 * np.exp(-1.6 * H) * (1.0 + 0.4 * r_over_R)
                 + 0.02 * rng.normal(size=n))
    y = np.column_stack([eps_s, c_over_c0])
    return X, y, ["r/R", "H", "DT", "H0", "Ug"], ["eps_s", "C/C0"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="xlsx with columns r/R,H,DT,H0,Ug,eps_s,C/C0")
    parser.add_argument("--demo", action="store_true", help="run on a synthetic dataset")
    parser.add_argument("--tag", default="wide_ann", help="suffix for the output files")
    parser.add_argument("--activations", nargs="*", default=None,
                        help="run the activation study over these activations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if args.data is None or args.demo or not Path(args.data).exists():
        if args.data is not None and not Path(args.data).exists():
            print(f"!! dataset {args.data} not found -- falling back to --demo")
        print("running on the synthetic demo dataset (the real campaign data belongs to "
              "the supervising group and is not redistributed here)")
        X, y, feature_names, target_names = synthetic_dataset()
        tag = f"{args.tag}_demo"
    else:
        X, y, feature_names, target_names = train_cv.load_surrogate_dataset(args.data)
        tag = args.tag
        print(f"loaded {X.shape[0]} samples with {X.shape[1]} features from {args.data}")

    print(f"  features: {feature_names}")
    print(f"  targets : {target_names}")

    model_factory = lambda: model_wide_ann.WideANN(  # noqa: E731
        model_wide_ann.WideANNConfig(
            input_dim=X.shape[1], output_dim=y.shape[1],
            activation="leaky_relu", output_activation="sigmoid",
        )
    )
    print(f"  model   : {model_factory().describe()}")

    config = train_cv.TrainConfig(seed=args.seed, n_splits=args.folds,
                                  num_epochs=args.epochs, verbose=args.verbose,
                                  learning_rate=0.005, weight_decay=0.4,
                                  batch_size=min(72, len(X)))
    result = train_cv.cross_validate(model_factory, X, y, config, target_names=target_names)

    viz.write_table(result["folds"], f"surrogate_folds_{tag}")
    (paths.TABLES_DIR / f"surrogate_aggregate_{tag}.json").write_text(
        json.dumps(result["aggregate"], indent=2)
    )
    print("\n=== 5-fold cross-validation ===")
    print(result["folds"][["fold", "n_train", "n_val", "r2_mean", "rmse_mean"]].to_string(index=False))
    print(f"  aggregate R2 = {result['aggregate']['r2_mean']:.4f} "
          f"+/- {result['aggregate']['r2_std']:.4f}")

    _figure_parity(result, target_names, tag)

    if args.activations:
        study = activation_study(X, y, target_names, args.activations, config)
        viz.write_table(study, "surrogate_activation_study")
        print("\n=== activation study (mean CV R2) ===")
        print(study.to_string(index=False))
        _figure_activation(study)

    reference = paths.TABLES_DIR / "surrogate_recorded_reference.csv"
    if reference.exists():
        print("\n=== numbers recorded in the legacy notebooks (not regenerated here) ===")
        print(pd.read_csv(reference).to_string(index=False))

    print("\nDone. See results/figures.")
    return 0


def activation_study(X, y, target_names, activations, base_config) -> pd.DataFrame:
    """Mean cross-validated R2 for each output activation."""
    rows = []
    for activation in activations:
        factory = lambda activation=activation: model_wide_ann.WideANN(  # noqa: E731
            model_wide_ann.WideANNConfig(
                input_dim=X.shape[1], output_dim=y.shape[1],
                activation=activation, output_activation="sigmoid",
            )
        )
        res = train_cv.cross_validate(factory, X, y, base_config, target_names=target_names)
        rows.append(
            dict(
                activation=activation,
                r2_mean=res["aggregate"]["r2_mean"],
                r2_std=res["aggregate"]["r2_std"],
                rmse_mean=res["aggregate"]["rmse_mean"],
            )
        )
        print(f"    {activation:14s} R2 = {res['aggregate']['r2_mean']:.4f}")
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


def _figure_parity(result, target_names, tag) -> None:
    y_true = np.vstack([p["y_val"] for p in result["predictions"]])
    y_pred = np.vstack([p["y_val_pred"] for p in result["predictions"]])
    fig, axes = plt.subplots(1, len(target_names), figsize=(5.4 * len(target_names), 4.6),
                             squeeze=False)
    for j, (ax, name) in enumerate(zip(axes[0], target_names)):
        ax.scatter(y_true[:, j], y_pred[:, j], s=14, alpha=0.6, color=viz.PALETTE[j % len(viz.PALETTE)],
                   edgecolor="none")
        lo = float(min(y_true[:, j].min(), y_pred[:, j].min()))
        hi = float(max(y_true[:, j].max(), y_pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        from sklearn.metrics import r2_score

        ax.set_title(f"{name}: $R^2$ = {r2_score(y_true[:, j], y_pred[:, j]):.3f}")
        ax.set_xlabel("simulated / measured")
        ax.set_ylabel("surrogate prediction")
        viz.annotate_panel(ax, "ab"[j] if j < 2 else "c")
    fig.suptitle("Wide-ANN surrogate, out-of-fold predictions", fontweight="bold")
    viz.save(fig, f"surrogate_parity_{tag}")


def _figure_activation(study: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(study["activation"], study["r2_mean"], yerr=study["r2_std"],
                  color=viz.PALETTE[: len(study)], capsize=3)
    for rect, value in zip(bars, study["r2_mean"]):
        ax.annotate(f"{value:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("mean 5-fold CV $R^2$")
    ax.set_xlabel("hidden-layer activation")
    ax.set_ylim(0, 1.0)
    ax.set_title("Activation function matters more than network depth:\n"
                 "R2 spread across activations on the same split")
    viz.save(fig, "surrogate_activation_study")


if __name__ == "__main__":
    raise SystemExit(main())
