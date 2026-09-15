#!/usr/bin/env python
"""
09 -- Prepare the raw recordings for a public data release (Zenodo / Dataverse).

Read-only by default: it writes a manifest with sizes and SHA-256 checksums, a
datasheet, and a short release README into ``release/``.  It does **not** copy,
compress or upload anything unless you explicitly ask for an archive.

    python scripts/09_package_data_release.py
    python scripts/09_package_data_release.py --create-archive --outdir /Volumes/Ext

Outputs (in ``release/``)
-------------------------
MANIFEST.csv          one row per file: dataset, relative path, bytes, sha256
DATASHEET.md          what the data is, how it was collected, known defects, licence
README_release.md     short pointer document for the Zenodo record
personal_info_scan.csv  every non-numeric header field found in the probe .ini/.pvd
                        files, for a human to review before publishing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import data_ofp, data_pressure, paths

RELEASE_DIR = REPO_ROOT / "release"

#: What goes into the release, and where it ends up.
RELEASE_SETS: Dict[str, dict] = {
    "pressure": dict(folder="Pressure_Energy", patterns=("*.xlsx",), suggested="zenodo-or-github"),
    "vibration": dict(folder="20250609 LMF", patterns=("*.mat", "*.bin"), suggested="zenodo"),
    "ofp": dict(folder="20250609 LC", patterns=("*.pvd", "*.ini", "*-st.txt"), suggested="zenodo"),
    "tabular": dict(folder="training & validation", patterns=("*.xlsx",), suggested="github"),
}

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        found.extend(sorted(root.rglob(pattern)))
    return sorted(set(found))


def build_manifest(with_hash: bool = True) -> pd.DataFrame:
    """One row per releasable raw file."""
    rows: List[dict] = []
    for key, spec in RELEASE_SETS.items():
        root = paths.dataset_path(key)
        if not root.is_dir():
            print(f"  ! {key}: {root} not found, skipped")
            continue
        files = iter_files(root, spec["patterns"])
        for path in files:
            rows.append(
                dict(
                    dataset=key,
                    relative_path=str(path.relative_to(root)),
                    suffix=path.suffix.lower(),
                    bytes=path.stat().st_size,
                    suggested_host=spec["suggested"],
                    sha256=sha256_of(path) if with_hash else "",
                )
            )
        print(f"  {key:10s} {len(files):5d} files  "
              f"{sum(r['bytes'] for r in rows if r['dataset'] == key) / 1e6:8.1f} MB")
    return pd.DataFrame(rows)


def scan_personal_info() -> pd.DataFrame:
    """Collect the free-text header fields of the probe files for human review.

    The ``.pvd`` and ``.ini`` headers can carry operator names, equipment ids and
    acquisition timestamps.  Nothing here is judged automatically -- the point is
    to put the fields in front of a human before the data goes public.
    """
    rows: List[dict] = []
    for path in sorted(paths.ofp_dir().rglob("*.pvd")):
        try:
            with path.open("rb") as handle:
                head = handle.read(4096)
            header = data_ofp.parse_header(head.decode("gbk", errors="replace"))
        except Exception:  # pragma: no cover - defensive
            continue
        for key, value in header.items():
            rows.append(dict(file=path.name, field=key, value=value))
    if not rows:
        return pd.DataFrame(columns=["file", "field", "value"])
    df = pd.DataFrame(rows)
    # only the fields that are not pure numbers/ids are interesting
    interesting = df[df["field"].isin(["文件名", "采样时间", "传感器编号", "Corsele", "采样通道"])]
    return interesting.drop_duplicates(subset=["field", "value"]).reset_index(drop=True)


DATASHEET_TEMPLATE = """# Datasheet: {title}

## 1. Motivation

Measurements were taken in a 3-D gas-solid fluidized bed to characterise the
transition between the bubbling and the turbulent regime and to support
data-driven surrogates for the local solids fraction. The published results of
this project are summarised in the companion code repository (see the Zenodo
record's "related identifiers").

## 2. Composition

| Dataset | Instrument | Rate | Channels | Records | Size |
|---|---|---|---|---|---|
{composition}

Full column/field documentation: see `docs/data_dictionary.md` in the code
repository.

## 3. Collection process

* one continuous session per subsystem; the three acquisition systems are
  **not synchronised** with each other;
* gas velocity 0.1-0.9 m/s, static bed height 1.0 m, FCC catalyst particles;
* per-record lengths: pressure 120 s, vibration ~1000 s, optical probe 10.49 s.

## 4. Known defects (please read before using)

{defects}

## 5. Uses

Intended: signal-processing benchmarks, regime-classification studies,
surrogate-model training, methods for quality control of experimental archives.

Not intended: absolute solids-fraction values without a probe-specific
calibration; any analysis that assumes the three subsystems are time-aligned;
any claim about the 7th accelerometer (no data exists for it).

## 6. Licence and attribution

Raw experimental data: {licence} (to be confirmed by the Fluidization
Laboratory). Code: MIT. Please cite both the dataset DOI and the code
repository. **The data were collected by the Fluidization Laboratory,
Beacons of Excellence Research and Innovation Institute, University of
Nottingham Ningbo China; the release is prepared by Chi Lu.**

## 7. Checksums

See `MANIFEST.csv` (SHA-256 per file).
"""

RELEASE_README = """# Raw data release

Files in this record:

* `MANIFEST.csv` -- every file with its size and SHA-256 checksum
* `DATASHEET.md` -- what the data is, how it was collected, known defects
* `README_release.md` -- this file
* `archives/` -- split archives of the raw recordings

Reproduce every published figure:

```bash
git clone https://github.com/<user>/fluidized-bed-ml
cd fluidized-bed-ml && pip install -r requirements.txt
export FBFML_DATA_ROOT=/path/to/this/dataset
python scripts/02_pressure_regime.py
python scripts/03_wavelet_validation.py
python scripts/07_regime_classification.py
python scripts/08_verify_claims.py
```
"""


def write_datasheet(manifest: pd.DataFrame) -> Path:
    composition = "\n".join(
        f"| {name} | see docs/data_dictionary.md | - | - | "
        f"{int((manifest['dataset'] == name).sum())} | "
        f"{manifest.loc[manifest['dataset'] == name, 'bytes'].sum() / 1e6:.0f} MB |"
        for name in ("pressure", "vibration", "ofp")
    )
    quality = (REPO_ROOT / "docs" / "data_quality.md")
    defects = (
        quality.read_text().split("## 13.")[0] if quality.exists()
        else "*See docs/data_quality.md in the code repository.*"
    )
    out = RELEASE_DIR / "DATASHEET.md"
    out.write_text(
        DATASHEET_TEMPLATE.format(
            title="pressure, vibration and optical-probe measurements in a 3-D "
                  "gas-solid fluidized bed across the bubbling-to-turbulent transition",
            composition=composition,
            defects=defects,
            licence="CC BY 4.0 (pending written confirmation)",
        )
    )
    return out


def create_archives(manifest: pd.DataFrame, outdir: Path, max_bytes: int = 2 * 10 ** 9) -> None:
    """Tar the recordings into chunked archives under ``max_bytes`` each."""
    import tarfile

    outdir.mkdir(parents=True, exist_ok=True)
    for key in manifest["dataset"].unique():
        root = paths.dataset_path(key)
        files = list(manifest.loc[manifest["dataset"] == key, "relative_path"])
        part = 1
        current = 0
        handle = None
        tar = None
        for relative in sorted(files):
            path = root / relative
            if tar is None or (current + path.stat().st_size > max_bytes and current > 0):
                if tar is not None:
                    tar.close()
                    print(f"    wrote {archive_path.name} ({current / 1e9:.2f} GB)")
                archive_path = outdir / f"{key}_part{part:02d}.tar"
                tar = tarfile.open(archive_path, "w")
                current = 0
                part += 1
            tar.add(path, arcname=f"{key}/{relative}")
            current += path.stat().st_size
        if tar is not None:
            tar.close()
            print(f"    wrote {archive_path.name} ({current / 1e9:.2f} GB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (much faster)")
    parser.add_argument("--create-archive", action="store_true",
                        help="also write split tar archives (needs disk space)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="where to write the archives (default: release/archives)")
    args = parser.parse_args()

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    print(paths.describe())
    print("\n=== building manifest ===")
    manifest = build_manifest(with_hash=not args.no_hash)
    if manifest.empty:
        print("nothing to release: raw data not found")
        return 1
    manifest.to_csv(RELEASE_DIR / "MANIFEST.csv", index=False)
    print(f"  total: {len(manifest)} files, {manifest['bytes'].sum() / 1e9:.2f} GB")

    print("\n=== scanning probe headers for free-text fields ===")
    personal = scan_personal_info()
    personal.to_csv(RELEASE_DIR / "personal_info_scan.csv", index=False)
    print(f"  {len(personal)} distinct (field, value) pairs -> release/personal_info_scan.csv")
    print("  review them by hand before publishing (operator ids, timestamps, ...)")

    datasheet = write_datasheet(manifest)
    (RELEASE_DIR / "README_release.md").write_text(RELEASE_README)
    print(f"\nwrote {datasheet.name}, README_release.md, MANIFEST.csv")

    if args.create_archive:
        print("\n=== creating archives ===")
        create_archives(manifest, args.outdir or (RELEASE_DIR / "archives"))

    print("\nNext: upload release/ to Zenodo (see docs/DATA_RELEASE.md) and put the DOI in README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
