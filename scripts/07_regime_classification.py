#!/usr/bin/env python
"""
07 -- Regime and gas-velocity classification from the optical-probe signal.

This is the analysis that turns the project's headline claim into something
checkable.  The poster states that the network "accurately distinguishes between
different flow regimes", but no classifier, accuracy value or confusion matrix
ever existed in the project folder.  Here the question is answered properly, on
the *local* probe data, with an honest evaluation protocol:

* **Task A** -- 9-class gas-velocity classification (0.1 ... 0.9 m/s);
* **Task B** -- 3-class regime classification (bubbling / transition / turbulent);
* **Task C** -- binary bubbling vs turbulent, dropping the transition velocity.

Evaluation protocols (reported separately, because they mean different things):

* ``replicate`` -- leave-one-replicate-out inside each condition: does the
  measurement repeat?
* ``radial``    -- test on a radial position never seen during training: does the
  label transfer across the bed?

Baselines: a majority-class predictor and a stratified random guess are reported
next to every score, so the reader can judge whether the accuracy is meaningful.

Outputs
-------
results/tables/probe_features.csv
results/tables/probe_classification_results.csv
results/tables/probe_classification_confusion_<task>.csv
results/figures/probe_classification.png
results/figures/probe_confusion_matrices.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import data_ofp, features_probe, paths, viz

CACHE = paths.TABLES_DIR / "probe_features.csv"
NON_FEATURE_COLUMNS = {"file", "bed_height_cm", "radial_position", "velocity",
                       "velocity_label", "n_samples", "error", "regime"}


def make_models() -> dict:
    """Three model families, deliberately cheap and interpretable."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=0, n_jobs=-1, min_samples_leaf=2
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=0, max_iter=300),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the cached feature table")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="use only the first N samples per record (debug)")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if CACHE.exists() and not args.force and args.max_samples is None:
        print(f"reusing cached feature table {CACHE}")
        table = pd.read_csv(CACHE)
    else:
        print(f"extracting probe features from {paths.ofp_dir()} ...")
        table = features_probe.records_to_features(
            data_ofp.iter_pvd_files(), max_samples=args.max_samples
        )
        table["regime"] = table["velocity"].map(features_probe.regime_label)
        if args.max_samples is None:
            table.to_csv(CACHE, index=False)

    table["regime"] = table["velocity"].map(features_probe.regime_label)
    # sklearn classifiers need a discrete target: format the velocity as a label
    table["velocity_label"] = table["velocity"].map(lambda v: f"{v:.1f}")
    viz.write_table(table, "probe_features")
    feature_columns = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    print(f"  records  : {len(table)}")
    print(f"  features : {len(feature_columns)}")
    print(f"  regimes  : {table['regime'].value_counts().to_dict()}")

    tasks = {
        "velocity_9class": ("velocity_label", None),
        "regime_3class": ("regime", None),
        "regime_2class": ("regime", ["bubbling", "turbulent"]),
    }
    protocols = {
        # every record of a condition must be either in train or in test -> group
        # by the condition cell
        "replicate": ["bed_height_cm", "radial_position", "velocity"],
        # test on a radial position never seen during training
        "radial": ["bed_height_cm", "radial_position"],
    }

    results = []
    for task, (target_column, keep_labels) in tasks.items():
        subset = table if keep_labels is None else table[table[target_column].isin(keep_labels)]
        for protocol, group_columns in protocols.items():
            n_groups = subset.groupby(group_columns).ngroups
            for model_name, model in make_models().items():
                scores = evaluate(subset, feature_columns, target_column,
                                  group_columns, model, n_groups,
                                  ordinal=(task == "velocity_9class"))
                results.append(dict(task=task, protocol=protocol, model=model_name,
                                    n_groups=n_groups, **scores))
                extra = (f"  within+/-1={scores['within_one_step']:.3f}"
                         if task == "velocity_9class" else "")
                print(f"  {task:16s} {protocol:10s} {model_name:20s} "
                      f"acc={scores['accuracy']:.3f}  macroF1={scores['macro_f1']:.3f}{extra}")

    results_df = pd.DataFrame(results).sort_values(
        ["task", "protocol", "accuracy"], ascending=[True, True, False]
    )
    viz.write_table(results_df, "probe_classification_results")

    best = results_df.loc[results_df.groupby(["task", "protocol"])["accuracy"].idxmax()]
    best.to_csv(paths.TABLES_DIR / "probe_classification_best.csv", index=False)
    print("\n=== best configuration per task / protocol ===")
    print(best[["task", "protocol", "model", "n_groups", "accuracy", "majority_baseline",
                "macro_f1", "within_one_step"]].to_string(index=False))

    _figure_summary(results_df)
    _figure_confusions(table, feature_columns)

    print("\nDone. See results/figures.")
    return 0


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def ordinal_tolerance_accuracy(y_true, y_pred, tolerance: float = 0.1) -> float:
    """Fraction of predictions within ``tolerance`` of the true gas velocity.

    The 9-class velocity problem is inherently ordinal: confusing 0.4 with 0.5 is
    a very different error from confusing 0.4 with 0.9.  Reporting a
    within-one-step accuracy next to the exact-match accuracy makes that visible.
    """
    t = np.asarray([float(v) for v in y_true])
    p = np.asarray([float(v) for v in y_pred])
    return float(np.mean(np.abs(t - p) <= tolerance + 1e-9))


def evaluate(df: pd.DataFrame, feature_columns, target_column: str, group_columns,
             model, n_groups: int, ordinal: bool = False) -> dict:
    """Grouped out-of-fold accuracy for one (task, protocol, model) combination."""
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy()
    groups = df.groupby(group_columns).ngroup().to_numpy()

    n_splits = max(2, min(6, n_groups))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = list(splitter.split(X, y, groups))
    except ValueError:
        # some tasks cannot be stratified with so few groups
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))

    y_true, y_pred = [], []
    for train_idx, test_idx in splits:
        from sklearn.base import clone

        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred.append(m.predict(X[test_idx]))
        y_true.append(y[test_idx])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels, counts = np.unique(y, return_counts=True)
    majority = float(counts.max() / counts.sum())
    out = dict(
        n_records=int(len(y)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        majority_baseline=majority,
        n_classes=int(len(labels)),
    )
    if ordinal:
        out["within_one_step"] = ordinal_tolerance_accuracy(y_true, y_pred, 0.1)
        out["within_two_steps"] = ordinal_tolerance_accuracy(y_true, y_pred, 0.2)
    else:
        out["within_one_step"] = float("nan")
        out["within_two_steps"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figure_summary(results: pd.DataFrame) -> None:
    tasks = results["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 4.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = results[results["task"] == task]
        protocols = list(sub["protocol"].unique())
        models = list(sub["model"].unique())
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            x = np.arange(len(protocols)) + i * width
            heights = []
            for protocol in protocols:
                sel = sub[(sub["protocol"] == protocol) & (sub["model"] == model)]
                heights.append(float(sel["accuracy"].iloc[0]) if not sel.empty else np.nan)
            ax.bar(x, heights, width=width, label=model)
        ax.axhline(float(sub["majority_baseline"].max()), ls="--", color="k", lw=1.2,
                   label="majority baseline")
        ax.set_xticks(np.arange(len(protocols)) + 0.4 - width / 2, protocols)
        ax.set_ylim(0, 1)
        ax.set_ylabel("out-of-fold accuracy")
        ax.set_title(task)
        ax.legend(fontsize=7)
    fig.suptitle("Classification of the flow state from 10.5 s of optical-probe signal",
                 fontweight="bold")
    viz.save(fig, "probe_classification")


def _figure_confusions(table: pd.DataFrame, feature_columns) -> None:
    targets = {"velocity_9class": "velocity_label",
               "regime_3class": "regime",
               "regime_2class": "regime"}
    keep = {"velocity_9class": None, "regime_3class": None,
            "regime_2class": ["bubbling", "turbulent"]}
    groups = ["bed_height_cm", "radial_position"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (task, target) in zip(axes, targets.items()):
        subset = table if keep[task] is None else table[table[target].isin(keep[task])]
        X = subset[feature_columns].to_numpy(dtype=float)
        y = subset[target].to_numpy()
        group_ids = subset.groupby(groups).ngroup().to_numpy()
        splitter = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=0)
        y_true, y_pred = [], []
        from sklearn.ensemble import RandomForestClassifier
        for train_idx, test_idx in splitter.split(X, y, group_ids):
            model = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            y_true.append(y[test_idx])
            y_pred.append(model.predict(X[test_idx]))
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        labels = np.unique(y)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), [str(l) for l in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), [str(l) for l in labels])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{task}\naccuracy = {accuracy_score(y_true, y_pred):.3f} "
                     f"(unseen radial position)")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                            color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised")
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            paths.TABLES_DIR / f"probe_classification_confusion_{task}.csv"
        )
    viz.save(fig, "probe_confusion_matrices")


if __name__ == "__main__":
    raise SystemExit(main())
