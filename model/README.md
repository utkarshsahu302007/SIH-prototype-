# Model

LightGBM multiclass classifier that labels NASA FIRMS VIIRS hotspot detections
into six operational categories.

## How to Run

```bash
# Prerequisites: complete phases 1–3 first, then:

# 1. Train (outputs model_YYYY-MM-DD.pkl in model/artifacts/)
python model/train.py

# 2. Predict → writes to DB predictions table
python model/predict.py

# 2b. Predict → stdout only (no DB needed, useful for quick checks)
python model/predict.py --no-db

# Use a specific model artifact:
python model/predict.py --model model/artifacts/model_2024-01-15.pkl
```

---

## Output Classes

| Class | Description |
|---|---|
| `industrial_persistent` | Continuously active industrial source — gas flares, furnaces, power plants |
| `industrial_event` | Fire or explosion at an industrial facility (FRP spike + proximity) |
| `agricultural_burning` | Crop-residue burning — seasonal, transient, on cropland |
| `forest_fire` | Spreading wildfire — tree cover + growing hotspot cluster |
| `mining` | Activity at a quarry or open-cast mine |
| `other_false_positive` | Sunglint, hot road surfaces, cloud edges, unknown — flagged for review |

---

## Features Used

### Spatial (Phase 2 — `hotspot_features` table)
| Feature | Source |
|---|---|
| `distance_to_nearest_facility` | Metres to nearest OSM industrial facility centroid |
| `nearest_facility_type` | `quarry / power_plant / refinery / steel_works / industrial_works / industrial` |
| `land_cover_class` | ESA WorldCover 2021 class at hotspot location |

### Temporal (Phase 3 — `temporal_features.parquet`)
| Feature | Description |
|---|---|
| `count_30d` / `count_90d` / `count_365d` | Detections in same ~375m cell in past N days |
| `persistence_ratio` | Fraction of dataset days the cell was active (0–1) |
| `frp_mean` / `frp_std` | Cell-level FRP statistics over all time |
| `frp_zscore` | How far this hotspot's FRP deviates from its cell baseline |
| `cluster_size` | Hotspot count in the same ~11km cell on this day |
| `cluster_growth` | Day-over-day change in `cluster_size` |

### Raw VIIRS
| Feature | Description |
|---|---|
| `bright_ti4` / `bright_ti5` | Brightness temperatures (K), I-4 and I-5 bands |
| `frp` | Fire radiative power (MW) |
| `confidence` | `l` / `n` / `h` (low / nominal / high) |
| `month` / `hour` | Acquisition timing (seasonality + day-night signal) |
| `daynight` / `satellite` | Observation metadata |

---

## Metrics

> **Fill in after running `python model/train.py`** — paste the terminal output below.

```
                        precision    recall  f1-score   support

  agricultural_burning      0.xx      0.xx      0.xx      xxxx
       forest_fire           0.xx      0.xx      0.xx      xxxx
  industrial_event           0.xx      0.xx      0.xx       xxx
industrial_persistent        0.xx      0.xx      0.xx      xxxx
            mining           0.xx      0.xx      0.xx       xxx
other_false_positive         0.xx      0.xx      0.xx      xxxx
```

---

## Known Weaknesses

### Class imbalance — `industrial_event`
Industrial events (fires, explosions at facilities) are rare by definition —
a flare site active every day is `industrial_persistent`, not `industrial_event`.
Even with `class_weight='balanced'`, recall for `industrial_event` will likely be
low. **Treat model predictions of `industrial_event` with additional scrutiny
in the demo.** If F1 < 0.3, consider lowering the FRP z-score threshold in
`weak_labels.py` to generate more training examples for this class.

### Noisy training signal
Weak labels from Phase 3 are heuristic. A gas flare in a cropland grid cell
that fires once during October could be assigned `agricultural_burning` by the
rules, even though it is actually `industrial_persistent`. The model inherits
these label errors. Ground-truth annotations for even 200–300 hotspots would
significantly improve reliability.

### Sparse spatial/temporal features
If OSM industrial data is incomplete (e.g., informal brick kilns not in OSM)
or WorldCover tiles weren't downloaded for a region, `distance_to_nearest_facility`
and `land_cover_class` will be NaN. LightGBM handles NaN natively via surrogate
splits, but classification quality degrades in data-sparse areas.

### Spatial split ≠ temporal split
The spatial split prevents within-location leakage, but temporal autocorrelation
remains: a model trained on Jan–Mar will see Dec–Feb patterns during evaluation
if the data spans a year. For production, use a strict temporal cutoff split.

### Early stopping on the test set
`lgb.early_stopping` currently uses the test split as its validation set, which
makes test metrics slightly optimistic. For a rigorous evaluation, hold out a
separate validation split for early stopping.
