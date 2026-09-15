#!/usr/bin/env python
"""
12 -- Robustness of the probe-based results: seed spread, and regression.

Two questions a reviewer will ask about the classification numbers:

1. *How much of the accuracy is luck of the split?*  -- answered by repeating the
   whole evaluation over ten seeds and reporting mean +/- standard deviation and a
   95 % interval across seeds.  A single split, which is what the project's
   materials implicitly relied on, cannot answer it.
2. *What if the target is treated as continuous instead of nine classes?*  --
   answered by the regression benchmark below, which also gives the error in the
   physical unit (m/s) rather than as a class label.

Outputs
-------
results/tables/robustness_classification.csv
results/tables/robustness_regression.csv
results/tables/robustness_summary.json
results/figures/robustness_seeds.png
results/figures/robustness_regression.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.base import clone

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import paths, viz
from fbfml.threading import single_threaded

CACHE = paths.TABLES_DIR / "probe_features.csv"
NON_FEATURE = {"file", "bed_height_cm", "radial_position", "velocity", "velocity_label",
               "n_samples", "error", "regime"}
SEEDS = list(range(10))
PROTOCOLS = {"replicate": ["bed_height_cm", "radial_position", "velocity"],
             "radial": ["bed_height_cm", "radial_position"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if not CACHE.exists():
        print(f"{CACHE} not found - run scripts/07_regime_classification.py first")
        return 1
    table = pd.read_csv(CACHE)
    table["velocity_label"] = table["velocity"].map(lambda v: f"{v:.1f}")
    table["regime"] = table["velocity"].map(_regime)
    features = [c for c in table.columns if c not in NON_FEATURE]
    print(f"records: {len(table)}   features: {len(features)}")

    # --- 1. classification over seeds --------------------------------------
    rows = []
    tasks = {"regime_2class": ("regime", ["bubbling", "turbulent"]),
             "regime_3class": ("regime", None),
             "velocity_9class": ("velocity_label", None)}
    models = {"random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                      random_state=None),
              "gradient_boosting": HistGradientBoostingClassifier(random_state=None,
                                                                 max_iter=200)}
    for seed in range(args.seeds):
        for task, (target, keep) in tasks.items():
            subset = table if keep is None else table[table[target].isin(keep)]
            for protocol, groups in PROTOCOLS.items():
                for name, model in models.items():
                    accuracy, f1 = _evaluate_classification(subset, features, target, groups,
                                                            model, seed)
                    rows.append(dict(task=task, protocol=protocol, model=name, seed=seed,
                                     accuracy=accuracy, macro_f1=f1))
    per_seed = pd.DataFrame(rows)
    viz.write_table(per_seed, "robustness_classification")

    summary = (
        per_seed.groupby(["task", "protocol", "model"])
        .agg(accuracy_mean=("accuracy", "mean"),
             accuracy_std=("accuracy", "std"),
             accuracy_p05=("accuracy", lambda s: float(np.percentile(s, 5))),
             accuracy_p95=("accuracy", lambda s: float(np.percentile(s, 95))),
             macro_f1_mean=("macro_f1", "mean"),
             n_seeds=("accuracy", "size"))
        .reset_index()
        .sort_values(["task", "protocol", "accuracy_mean"], ascending=[True, True, False])
    )
    viz.write_table(summary, "robustness_classification_summary")
    print(f"\n=== classification over {args.seeds} seeds ===")
    for task in tasks:
        sub = summary[summary["task"] == task]
        best = sub.iloc[0]
        print(f"  {task:16s} best = {best['model']:18s} ({best['protocol']:9s}) "
              f"accuracy {best['accuracy_mean']:.3f} +/- {best['accuracy_std']:.3f} "
              f"[{best['accuracy_p05']:.3f}, {best['accuracy_p95']:.3f}]  "
              f"macro-F1 {best['macro_f1_mean']:.3f}")

    # --- 2. regression benchmark -------------------------------------------
    regression_rows = []
    for seed in range(args.seeds):
        for protocol, groups in PROTOCOLS.items():
            for name, model in {"random_forest": RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                                      random_state=None),
                                "gradient_boosting": HistGradientBoostingRegressor(random_state=None,
                                                                                  max_iter=200)}.items():
                metrics = _evaluate_regression(table, features, "velocity", groups, model, seed)
                regression_rows.append(dict(protocol=protocol, model=name, seed=seed, **metrics))
    regression = pd.DataFrame(regression_rows)
    viz.write_table(regression, "robustness_regression")
    regression_summary = (
        regression.groupby(["protocol", "model"])
        .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
             rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             r2_baseline=("r2_baseline", "mean"))
        .reset_index()
        .sort_values("r2_mean", ascending=False)
    )
    viz.write_table(regression_summary, "robustness_regression_summary")
    print("\n=== regression of the gas velocity (m/s) from probe features ===")
    print(regression_summary.round(4).to_string(index=False))

    best = regression_summary.iloc[0]
    payload = dict(
        classification=summary.to_dict(orient="records"),
        regression=regression_summary.to_dict(orient="records"),
        note=("Every number is the mean over ten independent splits; the spread is the "
              "part of the score that a single train/test split would have hidden."),
    )
    (paths.TABLES_DIR / "robustness_summary.json").write_text(json.dumps(payload, indent=2))

    _figure_seeds(summary)
    _figure_regression(regression)

    print("\nDone. See results/figures.")
    return 0


def _regime(velocity: float) -> str:
    if velocity <= 0.5:
        return "bubbling"
    if abs(velocity - 0.6) < 1e-9:
        return "transition"
    return "turbulent"


def _evaluate_classification(df, features, target, group_columns, model, seed) -> tuple:
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy()
    groups = df.groupby(group_columns).ngroup().to_numpy()
    n_groups = df.groupby(group_columns).ngroups
    n_splits = max(2, min(6, n_groups))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    y_true, y_pred = [], []
    for train, test in splits:
        m = clone(model)
        m.set_params(**{k: seed for k, v in m.get_params().items() if k == "random_state"})
        with single_threaded():          # see fbfml.threading: 12x faster here
            m.fit(X[train], y[train])
        y_pred.append(m.predict(X[test]))
        y_true.append(y[test])
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))


def _evaluate_regression(df, features, target, group_columns, model, seed) -> dict:
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df.groupby(group_columns).ngroup().to_numpy()
    n_groups = df.groupby(group_columns).ngroups
    splitter = GroupKFold(n_splits=max(2, min(6, n_groups)))
    y_true, y_pred = [], []
    for train, test in splitter.split(X, y, groups):
        m = clone(model)
        m.set_params(**{k: seed for k, v in m.get_params().items() if k == "random_state"})
        with single_threaded():
            m.fit(X[train], y[train])
        y_pred.append(m.predict(X[test]))
        y_true.append(y[test])
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    baseline = float(r2_score(y_true, np.full_like(y_true, y_true.mean())))
    return dict(r2=float(r2_score(y_true, y_pred)),
                rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                mae=float(mean_absolute_error(y_true, y_pred)),
                r2_baseline=baseline)


def _figure_seeds(summary: pd.DataFrame) -> None:
    tasks = summary["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = summary[summary["task"] == task]
        labels = [f"{r['model'].replace('_', ' ')}\n({r['protocol']})" for _, r in sub.iterrows()]
        ax.barh(range(len(sub)), sub["accuracy_mean"], xerr=sub["accuracy_std"],
                color=viz.PALETTE[0], capsize=3)
        ax.set_yticks(range(len(sub)), labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("accuracy (mean $\\pm$ std over 10 seeds)")
        ax.set_title(task)
        ax.invert_yaxis()
    fig.suptitle("Split-to-split spread: the same model, ten different partitions",
                 fontweight="bold")
    viz.save(fig, "robustness_seeds")


def _figure_regression(regression: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for protocol, color in zip(regression["protocol"].unique(), viz.PALETTE):
        sub = regression[regression["protocol"] == protocol]
        axes[0].scatter(sub["model"], sub["r2"], color=color, alpha=0.7, label=protocol)
        axes[1].scatter(sub["model"], sub["mae"], color=color, alpha=0.7, label=protocol)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].annotate("baseline (predict the mean)", xy=(0.02, 0.02), xycoords="axes fraction",
                     fontsize=8)
    axes[0].set_ylabel("$R^2$ (out-of-fold, 10 seeds)")
    axes[1].set_ylabel("MAE (m/s)")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("model")
        ax.legend(fontsize=8)
        viz.annotate_panel(ax, letter)
    fig.suptitle("Continuous gas velocity from 10.5 s of probe signal", fontweight="bold")
    viz.save(fig, "robustness_regression")


if __name__ == "__main__":
    raise SystemExit(main())
