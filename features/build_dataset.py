"""
Join all features into one flat dataset ready for model training.

Inputs:
  - hotspots table          (base VIIRS columns)
  - hotspot_features table  (spatial join from features/spatial_join.py)
  - data/processed/temporal_features.parquet  (from features/temporal_features.py)

Output:
  - data/processed/dataset.parquet  (one row per hotspot, all features flat)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEMPORAL_PATH = Path("data/processed/temporal_features.parquet")
OUT_PATH = Path("data/processed/dataset.parquet")


def load_hotspots(engine) -> pd.DataFrame:
    """Load raw VIIRS columns from hotspots table."""
    sql = """
        SELECT id, latitude, longitude, bright_ti4, bright_ti5, frp,
               confidence, acq_date, acq_time, satellite, daynight
        FROM hotspots
    """
    df = pd.read_sql(sql, engine)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    log.info(f"Loaded {len(df):,} hotspots")
    return df


def load_spatial_features(engine) -> pd.DataFrame:
    """Load Phase 2 spatial join results from hotspot_features table."""
    sql = """
        SELECT hotspot_id,
               nearest_facility_id,
               nearest_facility_type,
               distance_to_nearest_facility,
               land_cover_code,
               land_cover_class
        FROM hotspot_features
    """
    df = pd.read_sql(sql, engine)
    log.info(f"Loaded {len(df):,} spatial feature rows")
    return df


def load_temporal_features(path: Path) -> pd.DataFrame:
    """Load Parquet file produced by temporal_features.py."""
    if not path.exists():
        log.error(
            f"Temporal features not found at {path}. "
            "Run: python features/temporal_features.py"
        )
        sys.exit(1)
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} temporal feature rows from {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge hotspot base columns, spatial features, and temporal features "
        "into a single flat Parquet file for model training."
    )
    parser.add_argument(
        "--temporal",
        type=Path,
        default=TEMPORAL_PATH,
        help=f"Path to temporal features Parquet (default: {TEMPORAL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help=f"Output Parquet path (default: {OUT_PATH})",
    )
    args = parser.parse_args()

    engine = get_engine()
    hotspots = load_hotspots(engine)
    spatial = load_spatial_features(engine)
    temporal = load_temporal_features(args.temporal)
    engine.dispose()

    # Left-join so hotspots without spatial/temporal features still appear
    # (their feature columns will be NaN — the model handles this via LightGBM's
    #  native missing-value support).
    df = hotspots.merge(spatial, left_on="id", right_on="hotspot_id", how="left")
    df = df.merge(temporal, left_on="id", right_on="hotspot_id", how="left")

    # Drop the redundant join-key copies
    df = df.drop(columns=[c for c in ["hotspot_id_x", "hotspot_id_y"] if c in df.columns])

    log.info(
        f"Dataset: {len(df):,} rows × {len(df.columns)} columns. "
        f"Spatial coverage: {spatial['hotspot_id'].nunique():,}/{len(hotspots):,} hotspots. "
        f"Temporal coverage: {temporal['hotspot_id'].nunique():,}/{len(hotspots):,} hotspots."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    log.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
