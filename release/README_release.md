# Raw data release

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
