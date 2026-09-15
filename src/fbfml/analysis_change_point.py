"""
Change-point (breakpoint) estimation on a small, noisy 1-D curve.

Used to ask a single question of three different instruments: *at which gas
velocity does the bed stop behaving like a bubbling bed?*  Each instrument gives
one scalar curve against ``U_g`` (a pressure, a bubble-passage frequency, a
vibration intensity), and the same two-segment fit is applied to all of them, so
the comparison is apples-to-apples.

The estimator is a grid search over the breakpoint, which is the most robust
choice when only ~9 velocities are available and the segments are short.
Uncertainty comes from a residual bootstrap; with this few points the intervals
are wide, and reporting them wide is the honest outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BreakpointFit:
    """Result of a two-segment fit."""

    x: np.ndarray
    y: np.ndarray
    x_break: float
    slope_low: float
    intercept_low: float
    slope_high: float
    intercept_high: float
    r2: float
    sse: float
    profile: np.ndarray  # SSE as a function of the breakpoint
    grid: np.ndarray

    @property
    def jump(self) -> float:
        """Value of the fitted curves at the breakpoint (0 for a continuous fit)."""
        return float(self.slope_low * self.x_break + self.intercept_low)


def _segment_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares line through ``(x, y)``; returns ``(slope, intercept, sse)``."""
    if x.size < 2:
        mean = float(y.mean()) if y.size else 0.0
        return 0.0, mean, float(np.sum((y - mean) ** 2))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sum(residual ** 2))


def fit_breakpoint(
    x: Sequence[float],
    y: Sequence[float],
    min_segment: int = 3,
    n_grid: int = 200,
    continuous: bool = True,
) -> BreakpointFit:
    """Fit two straight lines joined at a breakpoint chosen by grid search.

    Parameters
    ----------
    x, y
        The curve (same length). ``x`` must be sortable and monotone.
    min_segment
        Minimum number of points on each side of the breakpoint.
    n_grid
        Resolution of the breakpoint search between the two interior data points.
    continuous
        If ``True`` the two lines are constrained to meet at the breakpoint (a
        continuous piecewise-linear fit, the appropriate model when a physical
        quantity changes slope rather than jumping).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 2 * min_segment:
        raise ValueError(f"need at least {2 * min_segment} points, got {x.size}")

    lo, hi = x[min_segment - 1], x[-min_segment]
    grid = np.linspace(lo, hi, n_grid)
    profile = np.full(grid.size, np.nan)

    for i, xb in enumerate(grid):
        low = x <= xb
        high = x > xb
        if low.sum() < min_segment or high.sum() < min_segment:
            continue
        s_low, i_low, sse_low = _segment_fit(x[low], y[low])
        s_high, i_high, sse_high = _segment_fit(x[high], y[high])
        if continuous:
            # constrain both lines to pass through (xb, yb_est) where yb_est is
            # the value that minimises the total residual
            a = np.array([np.sum((x[low] - xb) ** 2), np.sum((x[high] - xb) ** 2)])
            b = np.array([np.sum((x[low] - xb) * y[low]), np.sum((x[high] - xb) * y[high])])
            yb = float(b.sum() / a.sum()) if a.sum() > 0 else float(y.mean())
            s_low = float(b[0] / a[0]) if a[0] > 0 else 0.0
            s_high = float(b[1] / a[1]) if a[1] > 0 else 0.0
            i_low = yb - s_low * xb
            i_high = yb - s_high * xb
            sse_low = float(np.sum((y[low] - (s_low * x[low] + i_low)) ** 2))
            sse_high = float(np.sum((y[high] - (s_high * x[high] + i_high)) ** 2))
        profile[i] = sse_low + sse_high

    best = int(np.nanargmin(profile))
    xb = float(grid[best])
    low = x <= xb
    high = x > xb
    if continuous:
        a = np.array([np.sum((x[low] - xb) ** 2), np.sum((x[high] - xb) ** 2)])
        b = np.array([np.sum((x[low] - xb) * y[low]), np.sum((x[high] - xb) * y[high])])
        yb = float(b.sum() / a.sum()) if a.sum() > 0 else float(y.mean())
        s_low = float(b[0] / a[0]) if a[0] > 0 else 0.0
        s_high = float(b[1] / a[1]) if a[1] > 0 else 0.0
        i_low = yb - s_low * xb
        i_high = yb - s_high * xb
    else:
        s_low, i_low, _ = _segment_fit(x[low], y[low])
        s_high, i_high, _ = _segment_fit(x[high], y[high])

    pred = np.where(x <= xb, s_low * x + i_low, s_high * x + i_high)
    sse = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return BreakpointFit(
        x=x, y=y, x_break=xb,
        slope_low=s_low, intercept_low=i_low,
        slope_high=s_high, intercept_high=i_high,
        r2=float(1 - sse / ss_tot) if ss_tot > 0 else float("nan"),
        sse=sse, profile=profile, grid=grid,
    )


def bootstrap_breakpoint_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 500,
    seed: int = 0,
    min_segment: int = 3,
) -> Dict[str, float]:
    """Residual-bootstrap confidence interval for the breakpoint location.

    Returns ``{'x_break', 'ci_low', 'ci_high', 'n_boot', 'share_at_bounds'}``.
    ``share_at_bounds`` is the fraction of bootstrap replicates that put the
    breakpoint at the edge of the searchable interval -- a high value means the
    data do not actually locate the breakpoint, which the report must state.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = fit_breakpoint(x, y, min_segment=min_segment)
    residual = y - np.where(
        x <= base.x_break,
        base.slope_low * x + base.intercept_low,
        base.slope_high * x + base.intercept_high,
    )
    rng = np.random.default_rng(seed)
    draws = []
    edges = 0
    lo_bound = float(np.sort(x)[min_segment - 1])
    hi_bound = float(np.sort(x)[-min_segment])
    for _ in range(n_boot):
        y_star = np.where(
            x <= base.x_break,
            base.slope_low * x + base.intercept_low,
            base.slope_high * x + base.intercept_high,
        ) + rng.choice(residual, size=residual.size, replace=True)
        try:
            fit = fit_breakpoint(x, y_star, min_segment=min_segment)
        except ValueError:
            continue
        draws.append(fit.x_break)
        if fit.x_break <= lo_bound + 1e-9 or fit.x_break >= hi_bound - 1e-9:
            edges += 1
    draws = np.asarray(draws)
    if draws.size == 0:  # pragma: no cover - defensive
        return dict(x_break=base.x_break, ci_low=float("nan"), ci_high=float("nan"),
                    n_boot=0, share_at_bounds=float("nan"))
    return dict(
        x_break=float(base.x_break),
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
        n_boot=int(draws.size),
        share_at_bounds=float(edges / draws.size),
    )
