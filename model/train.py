"""
Train a LightGBM multiclass classifier on the weak-labeled hotspot dataset.

Spatial train/test split: rows from the same 1°×1° grid cell (≈111 km/side)
go entirely into train OR test — never both — preventing spatial leakage.
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = Path("data/processed/labeled_dataset.csv")
ARTIFACTS_DIR = Path("model/artifacts")
TARGET_COL = "weak_label"

# ── Feature definitions ───────────────────────────────────────────────────────
# These must match what build_dataset.py + temporal_features.py produce.
# Update here if you add features — no other file needs changing.

NUMERIC_FEATURES = [
    "bright_ti4",                       # VIIRS band I-4 brightness temp (K)
    "bright_ti5",                       # VIIRS band I-5 brightness temp (K)
    "frp",                              # fire radiative power (MW)
    "distance_to_nearest_facility",     # metres (Phase 2 spatial join)
    "count_30d",                        # detections in same ~375m cell, last 30 d
    "count_90d",
    "count_365d",
    "persistence_ratio",                # fraction of dataset days cell was active
    "frp_mean",                         # cell-level FRP mean
    "frp_std",                          # cell-level FRP std
    "frp_zscore",                       # how far this FRP deviates from cell baseline
    "cluster_size",                     # hotspots in same ~11km cell, same day
    "cluster_growth",                   # day-over-day cluster change (spreading fire)
    "month",                            # 1–12 (seasonality signal)
    "hour",                             # 0–23 (derived from acq_time HHMM)
]
CATEGORICAL_FEATURES = [
    "nearest_facility_type",            # quarry / power_plant / refinery / …
    "land_cover_class",                 # tree_cover / cropland / built_up / …
    "confidence",                       # l / n / h
    "daynight",                         # D / N
    "satellite",                        # N (Suomi-NPP) / 1 (NOAA-20)
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Data loading and preparation ──────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    """Load labeled CSV and derive hour from HHMM acq_time string."""
    df = pd.read_csv(path, low_memory=False)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    # month may already exist from weak_labels.py; re-derive to be safe
    df["month"] = df["acq_date"].dt.month
    # acq_time is "HHMM" string (may lack leading zero: "614" = 06:14)
    df["hour"] = (
        df["acq_time"].astype(str).str.zfill(4).str[:2]
        .pipe(pd.to_numeric, errors="coerce")
    )
    log.info(
        f"Loaded {len(df):,} rows | classes: "
        f"{sorted(df[TARGET_COL].dropna().unique())}"
    )
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and type-cast feature columns for LightGBM."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"Feature columns not found in dataset (will be NaN): {missing}")

    X = df.reindex(columns=FEATURE_COLS)
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


# ── Spatial train/test split ──────────────────────────────────────────────────

def spatial_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign each 1°×1° grid region entirely to train or test.

    Why spatial and not random: randomly splitting rows of spatial data causes
    leakage because neighbouring detections share the same persistent-source
    signal (persistence_ratio, count_Nd, frp_mean). A random 80/20 split would
    give the model the answers to its own test questions.
    """
    region = (
        df["latitude"].apply(np.floor).astype(int).astype(str)
        + "_"
        + df["longitude"].apply(np.floor).astype(int).astype(str)
    )
    regions = region.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(regions)

    n_test = max(1, int(len(regions) * test_fraction))
    test_set = set(regions[:n_test])

    train_mask = ~region.isin(test_set)
    test_mask = region.isin(test_set)

    log.info(
        f"Spatial split → train: {train_mask.sum():,} rows "
        f"({len(regions) - n_test} regions) | "
        f"test: {test_mask.sum():,} rows ({n_test} regions)"
    )

    # Warn if any class is absent from test (can happen with small datasets)
    train_classes = set(df.loc[train_mask, TARGET_COL].dropna().unique())
    test_classes = set(df.loc[test_mask, TARGET_COL].dropna().unique())
    absent = train_classes - test_classes
    if absent:
        log.warning(
            f"Classes absent from test split: {absent}. "
            "Metrics for these classes will be zero. "
            "Collect more data or widen the bbox."
        )

    return df[train_mask].copy(), df[test_mask].copy()


# ── Model training ────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_classes: int,
) -> lgb.LGBMClassifier:
    """
    Fit LightGBM with early stopping on the test split.

    Note: using the test set for early stopping makes test metrics slightly
    optimistic — acceptable for a POC, not for a production model.
    """
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",   # corrects for dominant other_false_positive class
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    log.info(f"Best iteration: {clf.best_iteration_}  (of {clf.n_estimators})")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> None:
    """Print per-class report and a labelled confusion matrix."""
    sep = "=" * 68

    print(f"\n{sep}")
    print("PER-CLASS METRICS")
    print(sep)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(le.classes_))),
            target_names=le.classes_,
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print(sep)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_))))
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print()


def print_feature_importance(clf: lgb.LGBMClassifier, feature_names: list[str]) -> None:
    """Print top-10 features by gain."""
    importance = pd.Series(
        clf.booster_.feature_importance(importance_type="gain"),
        index=feature_names,
    ).sort_values(ascending=False)

    print("TOP 10 FEATURES BY GAIN")
    print("=" * 68)
    for feat, score in importance.head(10).items():
        bar = "█" * int(score / importance.iloc[0] * 30)
        print(f"  {feat:<40s} {bar}")
    print()


# ── Model persistence ─────────────────────────────────────────────────────────

def save_artifact(
    clf: lgb.LGBMClassifier,
    le: LabelEncoder,
    feature_cols: list[str],
) -> Path:
    """Dump model + encoder + feature list to a datestamped pkl."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"model_{date.today().isoformat()}.pkl"
    joblib.dump({"model": clf, "label_encoder": le, "feature_cols": feature_cols}, out_path)
    log.info(f"Saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM multiclass classifier with spatial train/test split."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Labeled CSV from weak_labels.py (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of spatial regions held out for evaluation (default: 0.2)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        log.error(
            f"Dataset not found: {args.dataset}. "
            "Run the features pipeline first:\n"
            "  python features/temporal_features.py\n"
            "  python features/build_dataset.py\n"
            "  python features/weak_labels.py"
        )
        raise SystemExit(1)

    df = load_dataset(args.dataset)
    df = df.dropna(subset=[TARGET_COL])
    log.info(f"Rows with valid labels: {len(df):,}")

    train_df, test_df = spatial_split(df, test_fraction=args.test_fraction)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_test = le.transform(test_df[TARGET_COL])

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    log.info(f"Features used: {list(X_train.columns)}")
    log.info(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    clf = train(X_train, y_train, X_test, y_test, n_classes=len(le.classes_))

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, le)
    print_feature_importance(clf, list(X_train.columns))

    save_artifact(clf, le, list(X_train.columns))


if __name__ == "__main__":
    main()
