"""Load FIRMS CSV files and OSM GeoJSON into the PostgreSQL/PostGIS database."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2.extras

# Allow running from project root: python db/load_data.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FIRMS_DIR = Path("data/raw/firms")
OSM_DIR = Path("data/raw/osm")

# VIIRS CSV column → DB column.  Columns not listed here are dropped.
VIIRS_COL_MAP = {
    "latitude":   "latitude",
    "longitude":  "longitude",
    "bright_ti4": "bright_ti4",
    "bright_ti5": "bright_ti5",
    "frp":        "frp",
    "confidence": "confidence",
    "acq_date":   "acq_date",
    "acq_time":   "acq_time",
    "satellite":  "satellite",
    "daynight":   "daynight",
}

# Tags that identify a facility type, checked in priority order.
FACILITY_TYPE_RULES: list[tuple[str, str, str]] = [
    ("power",      "plant",       "power_plant"),
    ("industrial", "refinery",    "refinery"),
    ("industrial", "steel_works", "steel_works"),
    ("man_made",   "works",       "industrial_works"),
    ("landuse",    "quarry",      "quarry"),
    ("landuse",    "industrial",  "industrial"),
]


# ── FIRMS helpers ─────────────────────────────────────────────────────────────

def read_firms_csv(csv_path: Path) -> pd.DataFrame:
    """Read one FIRMS VIIRS CSV, normalise columns, add source_file tag."""
    df = pd.read_csv(csv_path, dtype={"acq_time": str})
    # Rename only the columns we care about; silently drop the rest.
    df = df.rename(columns=VIIRS_COL_MAP)
    keep = list(VIIRS_COL_MAP.values())
    df = df[[c for c in keep if c in df.columns]]
    df["source_file"] = csv_path.name
    return df


def load_firms(conn, firms_dir: Path) -> int:
    """Load all FIRMS CSVs in firms_dir into the hotspots table; return row count inserted."""
    csv_files = sorted(firms_dir.glob("*.csv"))
    if not csv_files:
        log.warning(f"No CSV files found in {firms_dir}")
        return 0

    frames = [read_firms_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    # Drop rows where lat/lon is missing (shouldn't happen, but be safe).
    df = df.dropna(subset=["latitude", "longitude"])
    log.info(f"Loaded {len(df)} hotspots from {len(csv_files)} CSV(s)")

    cols = [c for c in df.columns if c != "latitude" and c != "longitude"]
    col_list = ", ".join(["latitude", "longitude", "geom"] + cols)
    placeholders = ", ".join(["%s", "%s", "ST_SetSRID(ST_MakePoint(%s, %s), 4326)"] + ["%s"] * len(cols))

    rows = [
        (
            row["latitude"],
            row["longitude"],
            row["longitude"],  # MakePoint(lon, lat)
            row["latitude"],
            *[row.get(c) for c in cols],
        )
        for _, row in df.iterrows()
    ]

    sql = f"INSERT INTO hotspots ({col_list}) VALUES %s"
    template = f"({placeholders})"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, template=template, page_size=500)
    conn.commit()
    log.info(f"Inserted {len(rows)} rows into hotspots")
    return len(rows)


# ── OSM helpers ──────────────────────────────────────────────────────────────

def derive_facility_type(props: dict) -> str:
    """Determine facility_type from OSM tags by checking rules in priority order."""
    for tag_key, tag_val, ftype in FACILITY_TYPE_RULES:
        if props.get(tag_key) == tag_val:
            return ftype
    return "other"


def load_osm(conn, osm_dir: Path, osm_file: Optional[Path]) -> int:
    """Load an OSM GeoJSON file into the facilities table; return row count inserted."""
    if osm_file:
        path = osm_file
    else:
        candidates = sorted(osm_dir.glob("*.geojson"))
        if not candidates:
            log.warning(f"No GeoJSON found in {osm_dir}. Run fetch_osm_facilities.py first.")
            return 0
        path = candidates[-1]  # most recently created

    log.info(f"Loading OSM facilities from {path}")
    fc = json.loads(path.read_text(encoding="utf-8"))
    features = fc.get("features", [])
    if not features:
        log.warning("GeoJSON has no features")
        return 0

    rows = []
    for feat in features:
        props = feat.get("properties") or {}
        geom_json = json.dumps(feat["geometry"])
        rows.append((
            props.get("osm_id"),
            props.get("osm_type"),
            derive_facility_type(props),
            props.get("name"),
            geom_json,
        ))

    sql = "INSERT INTO facilities (osm_id, osm_type, facility_type, name, geom) VALUES %s"
    template = "(%s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, template=template, page_size=500)
    conn.commit()
    log.info(f"Inserted {len(rows)} facilities into facilities")
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load FIRMS CSVs and OSM GeoJSON into the hotspots / facilities tables."
    )
    parser.add_argument(
        "--firms-dir",
        type=Path,
        default=FIRMS_DIR,
        help=f"Directory containing FIRMS CSV files (default: {FIRMS_DIR})",
    )
    parser.add_argument(
        "--osm-file",
        type=Path,
        default=None,
        help="Specific OSM GeoJSON file (default: latest file in data/raw/osm/)",
    )
    parser.add_argument(
        "--skip-firms",
        action="store_true",
        help="Skip loading FIRMS data (useful if already loaded)",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip loading OSM data",
    )
    args = parser.parse_args()

    conn = get_conn()
    try:
        if not args.skip_firms:
            load_firms(conn, args.firms_dir)
        if not args.skip_osm:
            load_osm(conn, OSM_DIR, args.osm_file)
    finally:
        conn.close()

    log.info("Done.")


if __name__ == "__main__":
    main()
