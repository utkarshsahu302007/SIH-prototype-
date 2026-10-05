"""
Compute per-grid-cell temporal features for every hotspot.

Features produced (one row per hotspot):
  hotspot_id, grid_cell, count_30d, count_90d, count_365d,
  persistence_ratio, frp_mean, frp_std, frp_zscore,
  cluster_size, cluster_growth
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# VIIRS pixel footprint ≈ 375 m. 375 / 111_000 ≈ 0.00338°
GRID_RES_DEG = 0.00338
# Coarser grid for cluster-size counting: 0.1° ≈ 11 km
CLUSTER_GRID_DEG = 0.1

OUT_PATH = Path("data/processed/temporal_features.parquet")


# ── Grid cell helpers ─────────────────────────────────────────────────────────

def add_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add fine grid_cell and coarse large_cell columns."""
    df = df.copy()
    df["lat_grid"] = (df["latitude"] / GRID_RES_DEG).round().astype(int)
    df["lon_grid"] = (df["longitude"] / GRID_RES_DEG).round().astype(int)
    df["grid_cell"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    df["lat_coarse"] = (df["latitude"] / CLUSTER_GRID_DEG).apply(np.floor).astype(int)
    df["lon_coarse"] = (df["longitude"] / CLUSTER_GRID_DEG).apply(np.floor).astype(int)
    df["large_cell"] = df["lat_coarse"].astype(str) + "_" + df["lon_coarse"].astype(str)
    return df


# ── Rolling window counts (vectorised within each grid cell) ──────────────────

def compute_rolling_counts(
    df: pd.DataFrame, windows: list[int] = [30, 90, 365]
) -> pd.DataFrame:
    """
    For each hotspot, count how many detections occurred in the same ~375m cell
    in the N days ending on (and including) the hotspot's own date.

    Uses a per-cell O(n²) numpy broadcast — fast because cells are small.
    """
    result = {f"count_{w}d": np.zeros(len(df), dtype=np.int32) for w in windows}

    for _, group in df.groupby("grid_cell"):
        idx = group.index.to_numpy()
        dates = group["acq_date"].values.astype("datetime64[D]")

        # delta[i, j] = dates[i] - dates[j] in whole days (positive = i is later)
        delta = (dates[:, None] - dates[None, :]) / np.timedelta64(1, "D")

        for w in windows:
            in_window = (delta >= 0) & (delta <= w)
            result[f"count_{w}d"][idx] = in_window.sum(axis=1).astype(np.int32)

    return pd.DataFrame(result, index=df.index)


# ── Per-cell aggregate features ───────────────────────────────────────────────

def compute_persistence(df: pd.DataFrame) -> pd.Series:
    """
    Fraction of total dataset days that each ~375m cell had at least one detection.
    Ranges 0–1; high values indicate a persistent source (flare, furnace, plant).
    """
    span = (df["acq_date"].max() - df["acq_date"].min()).days + 1
    span = max(span, 1)
    days_active = df.groupby("grid_cell")["acq_date"].transform("nunique")
    return (days_active / span).rename("persistence_ratio")


def compute_frp_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell FRP mean and std; per-hotspot z-score (spike indicator)."""
    frp_mean = df.groupby("grid_cell")["frp"].transform("mean").rename("frp_mean")
    frp_std  = (
        df.groupby("grid_cell")["frp"].transform("std")
        .fillna(0.0)
        .clip(lower=0.1)   # avoid division by zero for single-detection cells
        .rename("frp_std")
    )
    frp_zscore = ((df["frp"] - frp_mean) / frp_std).rename("frp_zscore")
    return pd.concat([frp_mean, frp_std, frp_zscore], axis=1)


# ── Cluster size and day-over-day growth ──────────────────────────────────────

def compute_cluster_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count hotspots per ~11km cell per day, and compute the day-over-day change.
    Positive cluster_growth indicates a spreading fire event.
    """
    daily = (
        df.groupby(["large_cell", "acq_date"])
        .size()
        .reset_index(name="cluster_size")
        .sort_values(["large_cell", "acq_date"])
    )
    daily["cluster_growth"] = (
        daily.groupby("large_cell")["cluster_size"]
        .diff()
        .fillna(0)
        .astype(int)
    )
    return df.merge(
        daily[["large_cell", "acq_date", "cluster_size", "cluster_growth"]],
        on=["large_cell", "acq_date"],
        how="left",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full temporal feature pipeline on a hotspot DataFrame."""
    log.info(f"Computing temporal features for {len(df):,} hotspots...")

    df = add_grid_cells(df)
    df = compute_cluster_growth(df)

    counts = compute_rolling_counts(df)
    persistence = compute_persistence(df)
    frp_stats = compute_frp_stats(df)

    out = pd.concat(
        [
            df[["id", "grid_cell"]].rename(columns={"id": "hotspot_id"}),
            counts,
            persistence,
            frp_stats,
            df[["cluster_size", "cluster_growth"]],
        ],
        axis=1,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute temporal features for all hotspots and save as Parquet."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help=f"Output Parquet path (default: {OUT_PATH})",
    )
    args = parser.parse_args()

    engine = get_engine()
    log.info("Reading hotspots from DB...")
    df = pd.read_sql(
        "SELECT id, latitude, longitude, frp, acq_date FROM hotspots",
        engine,
    )
    engine.dispose()

    if df.empty:
        log.error("No hotspots in DB. Run db/load_data.py first.")
        sys.exit(1)

    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date

    features = build_temporal_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    log.info(f"Saved {len(features):,} rows → {args.output}")


if __name__ == "__main__":
    main()
