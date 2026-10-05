"""
Apply heuristic rules to assign a provisional (weak) label to each hotspot.

Rules are evaluated in priority order via np.select — highest-priority condition wins.
Each condition is a plain boolean Series; no classes, no rule engine.

Priority:
  1. industrial_event        — spike near a known facility
  2. industrial_persistent   — persistent source near a facility
  3. mining                  — near a quarry
  4. agricultural_burning    — cropland, seasonal, transient
  5. forest_fire             — tree cover, spatially spreading
  6. other_false_positive    — everything else (default)

Outputs data/processed/labeled_dataset.csv with a 'weak_label' column appended.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = Path("data/processed/dataset.parquet")
OUT_PATH = Path("data/processed/labeled_dataset.csv")

# Tunable thresholds — adjust during model development based on label distribution
FACILITY_DIST_M = 500       # metres to nearest facility to be "near" it
FRP_SPIKE_ZSCORE = 2.0      # z-score above cell baseline → spike
PERSIST_HIGH = 0.6          # fraction of days active → persistent source
PERSIST_LOW = 0.1           # fraction of days active → transient event
CLUSTER_GROWTH_MIN = 5      # hotspots added in large cell day-over-day → spreading fire

# Months when crop-residue burning peaks in India (kharif harvest + rabi harvest)
AGRICULTURAL_MONTHS = {10, 11, 3, 4, 5}


def assign_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a 'weak_label' column added. Original df is not mutated."""
    df = df.copy()

    # ── Derived columns ───────────────────────────────────────────────────────

    df["month"] = pd.to_datetime(df["acq_date"]).dt.month

    # Fill NaN distances with a large sentinel so distance comparisons stay valid
    dist = df["distance_to_nearest_facility"].fillna(np.inf)

    # ── Individual conditions (plain boolean Series) ──────────────────────────

    near_facility = dist < FACILITY_DIST_M

    # Rule 1 — industrial event: nearby facility + FRP spike vs cell baseline
    cond_event = near_facility & df["frp_zscore"].fillna(0).gt(FRP_SPIKE_ZSCORE)

    # Rule 2 — industrial persistent: nearby facility + high persistence
    cond_persistent = near_facility & df["persistence_ratio"].fillna(0).gt(PERSIST_HIGH)

    # Rule 3 — mining: quarry specifically (subset of near_facility)
    cond_mining = (
        df["nearest_facility_type"].eq("quarry") & near_facility
    )

    # Rule 4 — agricultural burning: cropland + transient + seasonal window
    cond_agricultural = (
        df["land_cover_class"].eq("cropland")
        & df["persistence_ratio"].fillna(1).lt(PERSIST_LOW)
        & df["month"].isin(AGRICULTURAL_MONTHS)
    )

    # Rule 5 — forest fire: tree cover + day-over-day cluster growth
    cond_forest = (
        df["land_cover_class"].eq("tree_cover")
        & df["cluster_growth"].fillna(0).gt(CLUSTER_GROWTH_MIN)
    )

    # ── Apply in priority order — first True condition wins ───────────────────
    conditions = [
        cond_event,
        cond_persistent,
        cond_mining,
        cond_agricultural,
        cond_forest,
    ]
    choices = [
        "industrial_event",
        "industrial_persistent",
        "mining",
        "agricultural_burning",
        "forest_fire",
    ]

    df["weak_label"] = np.select(conditions, choices, default="other_false_positive")
    return df


def log_distribution(df: pd.DataFrame) -> None:
    """Print label counts and percentages."""
    counts = df["weak_label"].value_counts()
    total = len(df)
    log.info("Weak label distribution:")
    for label, n in counts.items():
        log.info(f"  {label:<30s} {n:>7,}  ({100 * n / total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply rule-based weak labels to the flat dataset Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_PATH,
        help=f"Input dataset Parquet (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help=f"Output CSV path (default: {OUT_PATH})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(
            f"Dataset not found at {args.input}. "
            "Run: python features/build_dataset.py"
        )
        raise SystemExit(1)

    df = pd.read_parquet(args.input)
    log.info(f"Loaded {len(df):,} hotspots from {args.input}")

    labeled = assign_weak_labels(df)
    log_distribution(labeled)

    other_count = (labeled["weak_label"] == "other_false_positive").sum()
    if other_count / len(labeled) > 0.7:
        log.warning(
            f"{100 * other_count / len(labeled):.0f}% of hotspots are 'other_false_positive'. "
            "This is expected if spatial/temporal features are sparse (e.g., limited data). "
            "Review threshold constants at the top of this file before training."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)
    log.info(f"Saved labeled dataset → {args.output}  ({len(labeled):,} rows)")


if __name__ == "__main__":
    main()
