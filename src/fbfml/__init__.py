"""
fbfml -- Gas-Solid Fluidized Bed: signal processing + machine-learning toolkit.

Scope
-----
Utilities to turn the raw laboratory recordings of a 3-D gas-solid fluidized
bed into (a) reproducible physical quantities and (b) validated feature
tables, plus the neural-network surrogates trained on top of them.

Layers (kept deliberately separate, one responsibility per module):

    paths.py              where the raw data lives (env-var overridable)
    data_pressure.py      differential-pressure logs  (.xlsx, 500 Hz, 6 taps)
    data_vibration.py     tri-axial accelerometer logs (.mat/.bin, 1666 Hz)
    data_ofp.py           optical-fibre-probe records (.pvd, 50 kHz) -> local solids fraction
    features_wavelet.py   db4 12-level DWT energy features (bit-exact validated)
    features_stats.py     descriptive statistics used as model inputs
    analysis_*.py         scientific questions answered from the above
    model_wide_ann.py     the wide feed-forward ANN surrogate
    train_cv.py           seeded, reproducible k-fold evaluation harness
    viz.py                plotting helpers / house style

Nothing in this package reads a hard-coded absolute path: all data access goes
through :mod:`fbfml.paths`.
"""

__version__ = "1.0.0"

from . import paths  # noqa: F401

__all__ = ["paths", "__version__"]
