"""
Compute spatial features for every hotspot and write them to hotspot_features.

For each hotspot:
  - distance_to_nearest_facility (metres) + that facility's type
  - land_cover_code and land_cover_class from the ESA WorldCover VRT

Uses geopandas.sjoin_nearest (vectorised, R-tree indexed) — no row-by-row loops.
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg2.extras
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn, get_engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# UTM Zone 44N covers most of India — unit is metres, needed for sjoin_nearest distances.
INDIA_UTM_EPSG = 32644

DEFAULT_VRT = Path("data/raw/landcover/landcover_mosaic.vrt")

# ESA WorldCover class codes (same as land_cover_lookup table).
LC_CLASS_NAMES: dict[int, str] = {
    10:  "tree_cover",
    20:  "shrubland",
    30:  "grassland",
    40:  "cropland",
    50:  "built_up",
    60:  "bare_sparse_vegetation",
    70:  "snow_ice",
    80:  "permanent_water",
    90:  "herbaceous_wetland",
    95:  "mangroves",
    100: "moss_lichen",
}


# ── DB read helpers ───────────────────────────────────────────────────────────

def load_hotspots(engine) -> gpd.GeoDataFrame:
    """Read all hotspots from DB as a GeoDataFrame (WGS-84)."""
    sql = "SELECT id, latitude, longitude, geom FROM hotspots"
    gdf = gpd.read_postgis(sql, engine, geom_col="geom")
    log.info(f"Loaded {len(gdf)} hotspots from DB")
    return gdf


def load_facilities(engine) -> gpd.GeoDataFrame:
    """Read facilities from DB; represent polygons as centroids for distance calc."""
    sql = "SELECT id, facility_type, ST_Centroid(geom) AS geom FROM facilities"
    gdf = gpd.read_postgis(sql, engine, geom_col="geom")
    log.info(f"Loaded {len(gdf)} facilities from DB")
    return gdf


# ── Spatial join ──────────────────────────────────────────────────────────────

def join_nearest_facility(
    hotspots: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Find the nearest facility for each hotspot.

    Returns a DataFrame with columns:
        hotspot_id, nearest_facility_id, nearest_facility_type,
        distance_to_nearest_facility (metres)
    """
    if facilities.empty:
        log.warning("No facilities in DB — distance columns will be NULL")
        return pd.DataFrame(
            {
                "hotspot_id": hotspots["id"],
                "nearest_facility_id": pd.NA,
                "nearest_facility_type": pd.NA,
                "distance_to_nearest_facility": pd.NA,
            }
        )

    # Project to metres for meaningful distances.
    hot_proj = hotspots[["id", "geom"]].to_crs(epsg=INDIA_UTM_EPSG)
    fac_proj = facilities[["id", "facility_type", "geom"]].to_crs(epsg=INDIA_UTM_EPSG)

    joined = gpd.sjoin_nearest(
        hot_proj,
        fac_proj.rename(columns={"id": "fac_id"}),
        how="left",
        distance_col="distance_to_nearest_facility",
    )

    result = joined[
        ["id", "fac_id", "facility_type", "distance_to_nearest_facility"]
    ].rename(
        columns={
            "id": "hotspot_id",
            "fac_id": "nearest_facility_id",
            "facility_type": "nearest_facility_type",
        }
    )
    log.info("Facility join complete")
    return result.reset_index(drop=True)


# ── Land cover sampling ───────────────────────────────────────────────────────

def sample_land_cover(hotspots: gpd.GeoDataFrame, vrt_path: Path) -> pd.DataFrame:
    """
    Sample ESA WorldCover raster at each hotspot location.

    Returns a DataFrame with columns: hotspot_id, land_cover_code, land_cover_class.
    Falls back to all-NULL if VRT is not present.
    """
    if not vrt_path.exists():
        log.warning(
            f"Land-cover VRT not found at {vrt_path}. "
            "Run fetch_landcover.py first. Skipping land-cover sampling."
        )
        return pd.DataFrame(
            {
                "hotspot_id": hotspots["id"],
                "land_cover_code": pd.NA,
                "land_cover_class": pd.NA,
            }
        )

    # rasterio.sample expects (lon, lat) pairs in the raster's CRS (WGS-84).
    coords = list(zip(hotspots["longitude"], hotspots["latitude"]))

    with rasterio.open(vrt_path) as src:
        # sample() returns an iterator of 1-D arrays (one per band).
        sampled = np.array(list(src.sample(coords)), dtype=float).flatten()

    # Rasterio returns nodata as the raw nodata value; treat 0 as unknown.
    codes = sampled.astype("Int64")
    codes[codes == 0] = pd.NA

    classes = codes.map(lambda c: LC_CLASS_NAMES.get(c) if pd.notna(c) else None)

    result = pd.DataFrame(
        {
            "hotspot_id": hotspots["id"].values,
            "land_cover_code": codes,
            "land_cover_class": classes,
        }
    )
    log.info("Land-cover sampling complete")
    return result


# ── DB write ──────────────────────────────────────────────────────────────────

def write_features(conn, features: pd.DataFrame) -> None:
    """
    Upsert rows into hotspot_features.

    Uses INSERT … ON CONFLICT DO UPDATE so re-running the script is safe.
    """
    sql = """
        INSERT INTO hotspot_features (
            hotspot_id, nearest_facility_id, nearest_facility_type,
            distance_to_nearest_facility, land_cover_code, land_cover_class
        )
        VALUES %s
        ON CONFLICT (hotspot_id) DO UPDATE SET
            nearest_facility_id          = EXCLUDED.nearest_facility_id,
            nearest_facility_type        = EXCLUDED.nearest_facility_type,
            distance_to_nearest_facility = EXCLUDED.distance_to_nearest_facility,
            land_cover_code              = EXCLUDED.land_cover_code,
            land_cover_class             = EXCLUDED.land_cover_class,
            computed_at                  = NOW()
    """

    def _coerce(v):
        """Convert numpy/pandas NA types to Python None for psycopg2."""
        if pd.isna(v):
            return None
        return v

    rows = [
        (
            int(r["hotspot_id"]),
            _coerce(r.get("nearest_facility_id")),
            _coerce(r.get("nearest_facility_type")),
            _coerce(r.get("distance_to_nearest_facility")),
            _coerce(r.get("land_cover_code")),
            _coerce(r.get("land_cover_class")),
        )
        for _, r in features.iterrows()
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    log.info(f"Upserted {len(rows)} rows into hotspot_features")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute spatial features for all hotspots and store in hotspot_features."
    )
    parser.add_argument(
        "--landcover-vrt",
        type=Path,
        default=DEFAULT_VRT,
        help=f"Path to WorldCover VRT mosaic (default: {DEFAULT_VRT})",
    )
    args = parser.parse_args()

    engine = get_engine()
    conn = get_conn()

    try:
        hotspots = load_hotspots(engine)
        if hotspots.empty:
            log.error("No hotspots in DB. Run db/load_data.py first.")
            sys.exit(1)

        facilities = load_facilities(engine)

        facility_df = join_nearest_facility(hotspots, facilities)
        landcover_df = sample_land_cover(hotspots, args.landcover_vrt)

        # Merge both feature sets on hotspot_id before writing.
        features = facility_df.merge(landcover_df, on="hotspot_id", how="left")

        write_features(conn, features)
    finally:
        conn.close()
        engine.dispose()

    log.info("Spatial join complete. Results in hotspot_features table.")


if __name__ == "__main__":
    main()
