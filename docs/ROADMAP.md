# Roadmap

Open items, in the order they are worth doing. Everything already delivered is in
`docs/results.md`.

## 1. Probe calibration constants *(highest value)*

`docs/data_quality.md` §4: the two constant pairs recoverable from the analysis
scripts do not belong to this probe/gain setting, so the pipeline reports the
calibration-free probe voltage and blocks the absolute solids fraction.
Obtaining the correct `(a, b)` for this campaign immediately unlocks the absolute
`εs` figures — `scripts/04` accepts them without any code change, and
`calibration_plausibility()` will start passing.

**Action:** request the probe calibration record for the 2025-06 campaign.

## 2. A third measurement height for the probe

The bubble-passage analysis (`scripts/11`) resolves two heights, 60 and 100 cm,
and their trends have opposite signs. A third height would turn the crossing at
0.33 m/s into a resolved height profile instead of an event marker.

**Action:** measure at 80 cm if the rig is still available, or model the missing
height from the existing two.

## 3. Confirm the pressure unit

`scripts/10` shows that the readings imply a bed-averaged bulk density of
662 kg/m³ *if* they are in kPa, which is inside the expected range for an FCC bed.
Confirmation from the transducer documentation would convert every relative
pressure statement into an absolute one.

## 4. Public data release

The raw recordings (3.8 GB) can be released. Everything needed is prepared:

* `scripts/09_package_data_release.py` — manifest with SHA-256 checksums, a
  datasheet, a release README, and a scan of the free-text probe header fields;
* `docs/DATA_RELEASE.md` — what goes to GitHub versus Zenodo, and the procedure.

Items 1 and 3 above should be settled first, so that the datasheet states the
calibration and the unit correctly.

## 5. Reference dataset for the surrogate

`scripts/06` implements the wide-ANN surrogate and the activation study but runs
on synthetic data in this repository, because the 530-row dataset belongs to the
laboratory. Adding it (or a documented subset) would make the surrogate results
reproducible end to end.

## 6. Continuous integration for the data-dependent stages

CI currently runs the unit tests, a byte-compile check and the synthetic surrogate
smoke test. With a small redistributable fixture (a few probe records and one
pressure log) the wavelet validation and the classification smoke test could run in
CI as well.
