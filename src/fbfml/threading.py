"""
Thread-pool control.

Small tabular problems (a few hundred rows, a few dozen features) are *slower*
with many OpenMP threads than with one, because thread contention dominates the
actual arithmetic.  Measured on this repository's data:

===========================  ==========  ==========
estimator                    threads     fit time
===========================  ==========  ==========
``HistGradientBoosting``     12 (default) 36 s
``HistGradientBoosting``     1            3 s
``RandomForest`` (400)       ``n_jobs=-1`` 0.7 s
===========================  ==========  ==========

So the gradient-boosting fits are executed inside :func:`single_threaded`, which
is a 12x speed-up with identical results.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def single_threaded() -> Iterator[None]:
    """Run the enclosed block with BLAS/OpenMP limited to a single thread."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        yield
        return
    with threadpool_limits(limits=1):
        yield
