"""Pull industrial facility geometries from OpenStreetMap via Overpass API and save as GeoJSON."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass uses S,W,N,E order — opposite of FIRMS W,S,E,N
INDIA_BBOX_OVERPASS = "8.0,68.0,37.5,97.5"  # S,W,N,E

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "osm"

# OSM tags that identify industrial or mining facilities
QUERY_TAGS = [
    '["landuse"="industrial"]',
    '["power"="plant"]',
    '["man_made"="works"]',
    '["industrial"="refinery"]',
    '["industrial"="steel_works"]',
    '["landuse"="quarry"]',
]


def build_query(bbox: str) -> str:
    """Build Overpass QL query fetching nodes, ways, and relations for each industrial tag."""
    blocks = []
    for tag in QUERY_TAGS:
        blocks.append(f"  node{tag}({bbox});")
        blocks.append(f"  way{tag}({bbox});")
        blocks.append(f"  relation{tag}({bbox});")
    body = "\n".join(blocks)
    return f"[out:json][timeout:180];\n(\n{body}\n);\nout geom;"


def fetch_overpass(query: str) -> dict:
    """POST query to Overpass API; one retry after 20 s on failure."""
    log.info("Sending Overpass query (may take 1–3 min for full India bbox)...")
    headers = {"User-Agent": "FIRMS-Hotspot-AI-Prototype/1.0"}
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=210)
    if not resp.ok:
        log.warning(f"Status {resp.status_code} — retrying in 20 s...")
        time.sleep(20)
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=210)
        resp.raise_for_status()
    return resp.json()


def element_to_feature(elem: dict) -> Optional[dict]:
    """Convert one Overpass element to a GeoJSON Feature; return None if not convertible."""
    props = {
        "osm_id": elem["id"],
        "osm_type": elem["type"],
        **elem.get("tags", {}),
    }

    if elem["type"] == "node":
        geometry = {"type": "Point", "coordinates": [elem["lon"], elem["lat"]]}
        return {"type": "Feature", "geometry": geometry, "properties": props}

    if elem["type"] == "way" and "geometry" in elem:
        coords = [[pt["lon"], pt["lat"]] for pt in elem["geometry"]]
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])  # close the ring
        geometry = {"type": "Polygon", "coordinates": [coords]}
        return {"type": "Feature", "geometry": geometry, "properties": props}

    # Relations and geometry-less elements are skipped (rare for these tags)
    return None


def to_geojson(overpass_data: dict) -> dict:
    """Convert full Overpass response to a GeoJSON FeatureCollection."""
    features = [
        f
        for elem in overpass_data.get("elements", [])
        if (f := element_to_feature(elem))
    ]
    skipped = len(overpass_data.get("elements", [])) - len(features)
    log.info(f"Converted {len(features)} features ({skipped} elements skipped/unsupported)")
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OSM industrial facilities via Overpass API and save as GeoJSON."
    )
    parser.add_argument(
        "--bbox",
        default=INDIA_BBOX_OVERPASS,
        help="Bounding box in S,W,N,E order (Overpass format). Default: India",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename inside /data/raw/osm/ (auto-timestamped if omitted)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    query = build_query(args.bbox)
    raw = fetch_overpass(query)
    geojson = to_geojson(raw)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = args.output or f"osm_industrial_{ts}.geojson"
    out_path = DATA_DIR / filename
    out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    log.info(f"Saved: {out_path}  ({len(geojson['features'])} features)")


if __name__ == "__main__":
    main()
