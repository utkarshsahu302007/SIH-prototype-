"""
Load trained model and write predictions to the DB predictions table.

Reads features from the labeled_dataset CSV (all features pre-computed),
runs model inference, and upserts results into the predictions table.
Safe to re-run: stale predictions for the same hotspots are deleted first.
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("model/artifacts")
DATASET_PATH = Path("data/processed/labeled_dataset.csv")

CATEGORICAL_FEATURES = [
    "nearest_facility_type",
    "land_cover_class",
    "confidence",
    "daynight",
    "satellite",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def find_latest_model(artifacts_dir: Path) -> Path:
    """Return the most recently dated model pkl (sorted by filename)."""
    pkls = sorted(artifacts_dir.glob("model_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No model pkl found in {artifacts_dir}. "
            "Run: python model/train.py"
        )
    latest = pkls[-1]
    log.info(f"Using model: {latest.name}")
    return latest


def load_model(path: Path) -> tuple:
    """Load (clf, label_encoder, feature_cols) from a joblib artifact."""
    artifact = joblib.load(path)
    return artifact["model"], artifact["label_encoder"], artifact["feature_cols"]


# ── Feature preparation (mirrors train.py) ────────────────────────────────────

def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Select and type-cast feature columns, deriving hour/month if needed."""
    df = df.copy()

    if "month" not in df.columns and "acq_date" in df.columns:
        df["month"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.month

    if "hour" not in df.columns and "acq_time" in df.columns:
        df["hour"] = (
            df["acq_time"].astype(str).str.zfill(4).str[:2]
            .pipe(pd.to_numeric, errors="coerce")
        )

    X = df.reindex(columns=feature_cols)
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(clf, le, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_class_strings, max_probability_scores)."""
    proba = clf.predict_proba(X)
    predicted_idx = proba.argmax(axis=1)
    return le.inverse_transform(predicted_idx), proba.max(axis=1)


# ── DB write ──────────────────────────────────────────────────────────────────

def write_predictions(
    conn,
    hotspot_ids: list[int],
    predicted_classes: np.ndarray,
    confidences: np.ndarray,
    model_version: str,
) -> None:
    """
    Delete existing predictions for these hotspot_ids, then bulk-insert new ones.
    Using delete+insert rather than ON CONFLICT avoids needing a UNIQUE constraint
    on the predictions table (which wasn't in the original schema.sql).
    """
    rows = [
        (int(hid), str(cls), float(conf), model_version)
        for hid, cls, conf in zip(hotspot_ids, predicted_classes, confidences)
    ]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM predictions WHERE hotspot_id = ANY(%s)",
            (hotspot_ids,),
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO predictions (hotspot_id, predicted_class, confidence, model_version)
            VALUES %s
            """,
            rows,
            page_size=500,
        )
    conn.commit()
    log.info(f"Wrote {len(rows):,} predictions to DB (model: {model_version})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run model inference on hotspot features and write results "
            "to the DB predictions table."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a specific model pkl (default: latest in model/artifacts/)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_PATH,
        help=f"Features CSV or Parquet to predict on (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Print predictions to stdout instead of writing to the DB "
             "(useful when PostGIS isn't set up)",
    )
    args = parser.parse_args()

    model_path = args.model or find_latest_model(ARTIFACTS_DIR)
    clf, le, feature_cols = load_model(model_path)
    model_version = model_path.stem  # e.g. "model_2024-01-15"

    if not args.input.exists():
        log.error(
            f"Input not found: {args.input}. "
            "Run the features pipeline first:\n"
            "  python features/build_dataset.py\n"
            "  python features/weak_labels.py"
        )
        sys.exit(1)

    log.info(f"Loading features from {args.input} ...")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, low_memory=False)

    if "id" not in df.columns:
        log.error("Input must have an 'id' column matching hotspots.id in the DB.")
        sys.exit(1)

    log.info(f"Running inference on {len(df):,} hotspots...")
    X = prepare_features(df, feature_cols)
    predicted_classes, confidences = run_inference(clf, le, X)

    # Print distribution regardless of --no-db
    counts = pd.Series(predicted_classes).value_counts()
    log.info("Prediction distribution:")
    for label, n in counts.items():
        pct = 100 * n / len(predicted_classes)
        log.info(f"  {label:<30s} {n:>7,}  ({pct:.1f}%)")

    if args.no_db:
        out = df[["id"]].copy()
        out["predicted_class"] = predicted_classes
        out["confidence"] = confidences.round(4)
        print(out.to_string(index=False))
        return

    conn = get_conn()
    try:
        write_predictions(
            conn,
            df["id"].tolist(),
            predicted_classes,
            confidences,
            model_version,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
