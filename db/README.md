# Database Setup

## Prerequisites

- PostgreSQL 14+ with the PostGIS extension available
- `psycopg2-binary` and `geopandas` installed (see project `requirements.txt`)

```bash
# Create the database (once)
createdb hotspots

# Apply the schema (idempotent — safe to re-run)
psql $DATABASE_URL -f db/schema.sql
```

Set the connection string:

```bash
export DATABASE_URL=postgresql://localhost/hotspots
# With credentials:
export DATABASE_URL=postgresql://user:password@host:5432/hotspots
```

---

## Load raw data into the DB

Run these after Phase 1 ingestion scripts have populated `data/raw/`:

```bash
# Load everything (FIRMS CSVs + latest OSM GeoJSON):
python db/load_data.py

# Load only FIRMS (skip OSM):
python db/load_data.py --skip-osm

# Load a specific OSM file:
python db/load_data.py --osm-file data/raw/osm/osm_industrial_20240101T120000.geojson
```

---

## Compute spatial features

```bash
# After load_data.py, compute nearest-facility + land-cover for every hotspot:
python features/spatial_join.py

# If your WorldCover VRT is in a non-default location:
python features/spatial_join.py --landcover-vrt /path/to/landcover_mosaic.vrt
```

Results land in the `hotspot_features` table. Re-running upserts cleanly.

---

## SQLite fallback (if PostGIS not available)

> PostGIS is required for spatial indexing. If you can't install it locally,
> a practical workaround for the hackathon is to run all spatial operations
> **in-memory with geopandas** (no DB at all) and save the feature-enriched
> DataFrame as a Parquet file. The model training step in Phase 3 can read
> Parquet directly without touching the DB.
>
> Ask the team lead before switching to the Parquet path — it changes Phase 3 inputs.

---

## Table summary

| Table | Populated by |
|---|---|
| `hotspots` | `db/load_data.py` |
| `facilities` | `db/load_data.py` |
| `land_cover_lookup` | `db/schema.sql` (static) |
| `hotspot_features` | `features/spatial_join.py` |
| `predictions` | `model/inference.py` (Phase 3) |
