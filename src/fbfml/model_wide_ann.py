"""
Neural-network surrogates used for the fluidized-bed models.

Two architectures are provided:

``WideANN``
    The deep-and-wide fully-connected network used for the published
    ``(r/R, H, DT, H0, Ug) -> (eps_s, C/C0)`` surrogate.  Widths 10-15-20-25-30
    in the encoder, three 30-unit blocks in the middle, mirrored decoder
    (30-25-20-15-2).  The output activation is configurable, because the
    activation study in ``XC_AC preformances.ipynb`` showed it matters a lot
    (LeakyReLU R2 = 0.945 vs ReLU 0.814 vs Tanh 0.671 on the same 5-fold split).

``MLP``
    A conventional 4-hidden-layer network, used for the vibration-feature
    -> operating-condition models.

Both keep a ``raw_output`` mode (no output squashing) because the original
notebooks contained a real bug worth remembering: a ``Sigmoid`` output layer was
combined with **un-standardised** targets whose range was 600-900 (temperature),
which produced negative R2 (down to -37) until the targets were standardised and
the output squashing removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": None,
}

#: Widths of the published wide architecture.
WIDE_HIDDEN: List[int] = [10, 15, 20, 25, 30, 30, 30, 30, 25, 20, 15]


def make_activation(name: str) -> Optional[nn.Module]:
    """Instantiate an activation by name (``"none"`` -> ``None``)."""
    if name not in ACTIVATIONS:
        raise KeyError(f"unknown activation {name!r}; known: {sorted(ACTIVATIONS)}")
    factory = ACTIVATIONS[name]
    return None if factory is None else factory()


@dataclass
class WideANNConfig:
    """Configuration of :class:`WideANN`."""

    input_dim: int = 5
    output_dim: int = 2
    hidden: Sequence[int] = field(default_factory=lambda: list(WIDE_HIDDEN))
    activation: str = "leaky_relu"
    output_activation: str = "sigmoid"
    dropout: float = 0.0
    batch_norm: bool = False


class WideANN(nn.Module):
    """Deep-and-wide feed-forward network (the published surrogate shape)."""

    def __init__(self, config: WideANNConfig) -> None:
        super().__init__()
        self.config = config
        layers: List[nn.Module] = []
        prev = config.input_dim
        for width in config.hidden:
            layers.append(nn.Linear(prev, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            act = make_activation(config.activation)
            if act is not None:
                layers.append(act)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev = width
        layers.append(nn.Linear(prev, config.output_dim))
        out_act = make_activation(config.output_activation)
        if out_act is not None:
            layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)

    # -- introspection ------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (
            f"WideANN(in={self.config.input_dim}, hidden={list(self.config.hidden)}, "
            f"out={self.config.output_dim}, act={self.config.activation}, "
            f"out_act={self.config.output_activation}, params={self.n_parameters})"
        )


class MLP(nn.Module):
    """A plain 4-hidden-layer perceptron (vibration -> operating condition)."""

    def __init__(self, input_dim: int, output_dim: int = 2,
                 hidden: Sequence[int] = (64, 32, 16, 8),
                 activation: str = "leaky_relu",
                 output_activation: str = "none",
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            act = make_activation(activation)
            if act is not None:
                layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        out_act = make_activation(output_activation)
        if out_act is not None:
            layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block used in one of the later experiments."""

    def __init__(self, dim: int, activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = make_activation(activation) or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + self.dropout(self.act(self.bn(self.linear(x))))


class ResidualMLP(nn.Module):
    """Wide trunk with residual blocks (the `XC_regime_最好的` architecture)."""

    def __init__(self, input_dim: int, output_dim: int = 2, width: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout=dropout) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.output_layer(self.blocks(self.input_layer(x)))


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory by name: ``"wide"``, ``"mlp"`` or ``"residual"``."""
    name = name.lower()
    if name in ("wide", "wide_ann"):
        return WideANN(WideANNConfig(**kwargs))
    if name == "mlp":
        return MLP(**kwargs)
    if name in ("residual", "resmlp"):
        return ResidualMLP(**kwargs)
    raise KeyError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
