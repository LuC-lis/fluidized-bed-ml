"""
Single source of truth for filesystem locations.

The original notebooks hard-coded paths such as
``/Users/luchi/Desktop/工作簿1.xlsx``, ``F:\\XC_OFP&2D camera\\...`` and
``C:\\Users\\cbi20\\...``, which made them unrunnable anywhere else.
Everything now resolves from one, overridable root.

Resolution order for the raw-data root
--------------------------------------
1. the ``FBFML_DATA_ROOT`` environment variable, if set;
2. ``<repo>/data`` if it exists (the recommended layout for a clone, see README);
3. the parent directory of the repository -- the layout used on the machine
   where this project was written, with the original folder names.

Use :func:`describe` to print what was resolved and whether it exists.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repository layout -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LEGACY_DIR = REPO_ROOT / "legacy"

# --- raw-data layout -------------------------------------------------------
#: folder name -> canonical dataset key.  Used for both of the supported
#: on-disk layouts (``<repo>/data/<name>`` and ``<data_root>/<name>``).
DATASET_DIRNAMES = {
    "pressure": "Pressure_Energy",
    "vibration": "20250609 LMF",
    "ofp": "20250609 LC",
    "tabular": "training & validation",
    "notebooks_legacy": "神经网络",
    "posters": "PPT & Posters",
    "publication": "Result Article",
}


def data_root() -> Path:
    """Return the directory that contains the raw experimental datasets."""
    env = os.environ.get("FBFML_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    local = REPO_ROOT / "data"
    if local.is_dir():
        return local
    return REPO_ROOT.parent


def dataset_path(key: str) -> Path:
    """Absolute path of a raw dataset folder.

    Raises
    ------
    KeyError
        if ``key`` is not a known dataset.
    """
    if key not in DATASET_DIRNAMES:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(DATASET_DIRNAMES)}")
    return data_root() / DATASET_DIRNAMES[key]


def pressure_dir() -> Path:
    return dataset_path("pressure")


def vibration_dir() -> Path:
    return dataset_path("vibration")


def ofp_dir() -> Path:
    return dataset_path("ofp")


def tabular_dir() -> Path:
    return dataset_path("tabular")


def ensure_results_dirs() -> None:
    """Create ``results/figures`` and ``results/tables`` if missing."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable report of the resolved locations (for logs / README)."""
    root = data_root()
    lines = [f"repo root : {REPO_ROOT}", f"data root : {root}"]
    for key, dirname in DATASET_DIRNAMES.items():
        p = root / dirname
        lines.append(f"  {key:18s} {'OK ' if p.exists() else 'MISSING'} {p}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(describe())
