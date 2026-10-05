#!/usr/bin/env bash
# Run all data ingestion steps in order.
# From project root: bash run_ingestion.sh
#
# Prerequisites:
#   export FIRMS_MAP_KEY=your_key_here
#   pip install requests
#   gdal on PATH (for landcover VRT step)

set -euo pipefail

: "${FIRMS_MAP_KEY:?FIRMS_MAP_KEY must be set. Export it before running this script.}"

START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-01-07}"
BBOX_FIRMS="${BBOX_FIRMS:-68.0,8.0,97.5,37.5}"     # W,S,E,N
BBOX_OSM="${BBOX_OSM:-8.0,68.0,37.5,97.5}"          # S,W,N,E (Overpass order)
MAX_LC_TILES="${MAX_LC_TILES:-4}"                    # set to empty string for full download

echo "========================================"
echo " Data Ingestion"
echo " Dates   : $START_DATE → $END_DATE"
echo " FIRMS   bbox: $BBOX_FIRMS (W,S,E,N)"
echo " OSM     bbox: $BBOX_OSM  (S,W,N,E)"
echo " Landcover tiles: first $MAX_LC_TILES (increase MAX_LC_TILES for more)"
echo "========================================"

echo ""
echo "--- Step 1: FIRMS VIIRS hotspots ---"
python data_ingestion/fetch_firms.py \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --bbox "$BBOX_FIRMS" \
  --source VIIRS_SNPP_SP

echo ""
echo "--- Step 2: OSM industrial facilities ---"
python data_ingestion/fetch_osm_facilities.py --bbox "$BBOX_OSM"

echo ""
echo "--- Step 3: ESA WorldCover land-cover tiles ---"
if [ -n "$MAX_LC_TILES" ]; then
  python data_ingestion/fetch_landcover.py --max-tiles "$MAX_LC_TILES"
else
  python data_ingestion/fetch_landcover.py
fi

echo ""
echo "========================================"
echo " Ingestion complete. Raw data in data/raw/"
echo "========================================"
