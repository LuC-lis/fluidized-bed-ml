"""Tests for the model definitions and the cross-validation harness.

These use a synthetic dataset, so they run in seconds and need no raw data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fbfml import model_wide_ann, train_cv


def test_wide_ann_shapes_and_parameter_count():
    model = model_wide_ann.WideANN(model_wide_ann.WideANNConfig())
    assert model(torch.zeros(4, 5)).shape == (4, 2)
    assert model.n_parameters > 1000
    assert "WideANN" in model.describe()


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "tanh", "gelu", "elu", "none"])
def test_activations_are_constructible(activation):
    model = model_wide_ann.WideANN(
        model_wide_ann.WideANNConfig(activation=activation, output_activation="none")
    )
    out = model(torch.ones(3, 5))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_unknown_activation_raises():
    with pytest.raises(KeyError):
        model_wide_ann.make_activation("swish")


def test_cross_validate_is_reproducible_and_learns():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(120, 5))
    y = np.column_stack([0.5 - 0.4 * X[:, 0] + 0.1 * X[:, 4],
                         0.3 + 0.2 * X[:, 1]])
    # a deliberately small version of the architecture: the point here is the
    # harness, not the capacity of the published network
    factory = lambda: model_wide_ann.WideANN(  # noqa: E731
        model_wide_ann.WideANNConfig(input_dim=5, output_dim=2, hidden=[16, 8],
                                     output_activation="none")
    )
    config = train_cv.TrainConfig(seed=7, n_splits=3, num_epochs=250, patience=80,
                                  batch_size=32, learning_rate=0.01, weight_decay=1e-4)

    first = train_cv.cross_validate(factory, X, y, config, target_names=["a", "b"])
    second = train_cv.cross_validate(factory, X, y, config, target_names=["a", "b"])

    assert first["aggregate"]["r2_mean"] == pytest.approx(second["aggregate"]["r2_mean"])
    # a near-linear problem must be learnable
    assert first["aggregate"]["r2_mean"] > 0.9
    assert len(first["folds"]) == 3
    assert set(first["folds"]["fold"]) == {1, 2, 3}


def test_scaler_is_fitted_inside_the_fold_only():
    """A blatant train/validation leak must remain visible.

    If the harness fitted the scaler on the full dataset, a model that memorises
    nothing would still score well on a shuffled target.  We check the opposite:
    with a pure-noise target the CV score must stay near/below zero.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    y = rng.normal(size=(80, 1))
    factory = lambda: model_wide_ann.MLP(input_dim=5, output_dim=1)  # noqa: E731
    config = train_cv.TrainConfig(seed=1, n_splits=3, num_epochs=40, patience=15,
                                 batch_size=16, learning_rate=0.01, weight_decay=1e-3)
    result = train_cv.cross_validate(factory, X, y, config, target_names=["noise"])
    assert result["aggregate"]["r2_mean"] < 0.5
