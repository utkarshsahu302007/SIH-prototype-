# Data Ingestion

Three standalone scripts that pull raw data from public APIs.

## Prerequisites

```bash
pip install requests
# For fetch_landcover.py only:
pip install rasterio          # used in later feature-engineering step
# GDAL must be on PATH for gdalbuildvrt (used to build the VRT mosaic):
# conda install -c conda-forge gdal   OR   brew install gdal
```

Set your NASA FIRMS API key (free, register at https://firms.modaps.eosdis.nasa.gov/api/):
```bash
export FIRMS_MAP_KEY=your_key_here
```

---

## Scripts

### 1. `fetch_firms.py` — NASA FIRMS VIIRS hotspots → CSV
```bash
# Historical data (archive/SP source — default):
python data_ingestion/fetch_firms.py --start 2024-01-01 --end 2024-01-07

# Last 10 days (NRT source):
python data_ingestion/fetch_firms.py --start 2024-01-01 --end 2024-01-07 --source VIIRS_SNPP_NRT

# Custom bounding box (W,S,E,N):
python data_ingestion/fetch_firms.py --start 2024-01-01 --end 2024-01-03 --bbox "72,18,80,24"
```
Saves one file per day to `data/raw/firms/viirs_viirs_snpp_sp_YYYY-MM-DD.csv`.

FIRMS API note: SP (Standard Processing) is the archive source and works for any historical date.
NRT (Near Real-Time) only has data for the last 10 days. Use SP for any date range older than that.

---

### 2. `fetch_osm_facilities.py` — OpenStreetMap industrial facilities → GeoJSON
```bash
# Full India (default):
python data_ingestion/fetch_osm_facilities.py

# Custom bbox (S,W,N,E — Overpass order, opposite of FIRMS!):
python data_ingestion/fetch_osm_facilities.py --bbox "18,72,24,80"

# Custom output filename:
python data_ingestion/fetch_osm_facilities.py --output osm_industrial_india.geojson
```
Saves to `data/raw/osm/osm_industrial_<timestamp>.geojson`.

Tags fetched: `landuse=industrial`, `power=plant`, `man_made=works`,
`industrial=refinery`, `industrial=steel_works`, `landuse=quarry`.

---

### 3. `fetch_landcover.py` — ESA WorldCover 2021 (10m) → GeoTIFF tiles + VRT
```bash
# Dry run first — prints all tile URLs so you can verify naming before downloading:
python data_ingestion/fetch_landcover.py --dry-run

# Test with first 4 tiles only (~400–800 MB):
python data_ingestion/fetch_landcover.py --max-tiles 4

# Full India (~120 tiles, 15–30 GB, may take hours):
python data_ingestion/fetch_landcover.py
```
Downloads tiles to `data/raw/landcover/` and builds `landcover_mosaic.vrt`.

> **Important**: Always run `--dry-run` first to verify the tile naming convention against
> actual files in the S3 bucket. If the tile names are off, adjust `tile_name()` in the script.
> Tiles are named by their SW (lower-left) corner: `N09E066` = tile from 9°N–12°N, 66°E–69°E.

---

## Run Order

Use `run_ingestion.sh` from the project root, or run scripts manually in this order:

1. `fetch_firms.py` — no dependencies
2. `fetch_osm_facilities.py` — no dependencies
3. `fetch_landcover.py` — needs `gdal` on PATH for the VRT step

Raw data lands in `data/raw/`. Files in that directory are `.gitignore`d.
