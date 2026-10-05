"""Download ESA WorldCover 10m land-cover tiles for a bbox and build a GDAL VRT mosaic."""

import argparse
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ESA WorldCover v200 (2021) — public S3 bucket, no auth required
WORLDCOVER_S3 = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Tile grid: 3°×3° tiles, named by their SW (lower-left) corner.
# Example: N09E066 covers 9°N–12°N, 66°E–69°E.
# Use --dry-run to print all tile URLs before downloading.
TILE_SIZE_DEG = 3

INDIA_BBOX = (68.0, 8.0, 97.5, 37.5)  # W, S, E, N

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "landcover"


def tile_name(sw_lat: int, sw_lon: int) -> str:
    """Build tile name from its SW (lower-left) corner, e.g. N09E066."""
    lat_pfx = "N" if sw_lat >= 0 else "S"
    lon_pfx = "E" if sw_lon >= 0 else "W"
    return f"{lat_pfx}{abs(sw_lat):02d}{lon_pfx}{abs(sw_lon):03d}"


def tile_url(name: str) -> str:
    """Build the S3 download URL for a named WorldCover tile."""
    return f"{WORLDCOVER_S3}/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[tuple[int, int]]:
    """Return SW-corner (lat, lon) pairs for all 3°×3° tiles overlapping the bbox."""
    lat_start = math.floor(south / TILE_SIZE_DEG) * TILE_SIZE_DEG
    lat_end = math.floor(north / TILE_SIZE_DEG) * TILE_SIZE_DEG
    lon_start = math.floor(west / TILE_SIZE_DEG) * TILE_SIZE_DEG
    lon_end = math.floor(east / TILE_SIZE_DEG) * TILE_SIZE_DEG
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + TILE_SIZE_DEG, TILE_SIZE_DEG)
        for lon in range(lon_start, lon_end + TILE_SIZE_DEG, TILE_SIZE_DEG)
    ]


def download_tile(name: str, out_dir: Path) -> Optional[Path]:
    """Download one tile GeoTIFF; skip if already on disk; one retry on failure."""
    out_path = out_dir / f"ESA_WorldCover_10m_2021_v200_{name}_Map.tif"

    if out_path.exists():
        log.info(f"  Already exists: {out_path.name}")
        return out_path

    url = tile_url(name)
    log.info(f"  Downloading {name} ...")

    def _get() -> requests.Response:
        return requests.get(url, stream=True, timeout=300)

    try:
        resp = _get()
        if resp.status_code == 404:
            log.warning(f"  {name}: 404 — ocean tile or outside WorldCover coverage, skipping")
            return None
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        size_mb = out_path.stat().st_size / 1_000_000
        log.info(f"  Saved {out_path.name} ({size_mb:.1f} MB)")
        return out_path
    except Exception as exc:
        log.warning(f"  First attempt failed ({exc}) — retrying in 15 s...")
        time.sleep(15)
        try:
            resp = _get()
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            return out_path
        except Exception as exc2:
            log.error(f"  Failed to download {name}: {exc2}")
            return None


def build_vrt(tile_paths: list[Path], vrt_path: Path) -> None:
    """Create a GDAL VRT that mosaics all downloaded tiles (requires gdal installed)."""
    list_file = vrt_path.parent / "tile_list.txt"
    list_file.write_text("\n".join(str(p) for p in tile_paths), encoding="utf-8")
    cmd = ["gdalbuildvrt", "-input_file_list", str(list_file), str(vrt_path)]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"gdalbuildvrt failed:\n{result.stderr}")
        raise RuntimeError("VRT build failed — is gdal installed? (pip install gdal or conda install gdal)")
    log.info(f"VRT saved: {vrt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download ESA WorldCover 10m tiles for a bounding box and build a VRT mosaic. "
            "Full India coverage requires ~120 tiles (~15–30 GB). Use --max-tiles 4 for a quick test."
        )
    )
    parser.add_argument(
        "--bbox",
        default=",".join(str(v) for v in INDIA_BBOX),
        help="W,S,E,N bounding box (default: India: 68,8,97.5,37.5)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Limit download to first N tiles (useful for testing). Default: all tiles in bbox",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all tile names and URLs without downloading anything",
    )
    args = parser.parse_args()

    try:
        west, south, east, north = [float(x) for x in args.bbox.split(",")]
    except ValueError:
        log.error("--bbox must be four comma-separated floats: W,S,E,N")
        sys.exit(1)

    all_tiles = tiles_for_bbox(west, south, east, north)
    log.info(f"Bbox requires {len(all_tiles)} tile(s) for full coverage")

    if args.dry_run:
        log.info("Dry run — printing tile list only:")
        for sw_lat, sw_lon in all_tiles:
            name = tile_name(sw_lat, sw_lon)
            print(f"  {name}  {tile_url(name)}")
        log.info(
            "Tip: verify one URL in your browser to confirm the tile naming convention is correct "
            "before running a full download."
        )
        return

    tiles_to_fetch = all_tiles
    if args.max_tiles and len(all_tiles) > args.max_tiles:
        log.warning(
            f"--max-tiles {args.max_tiles}: downloading {args.max_tiles} of {len(all_tiles)} tiles. "
            f"The VRT will NOT cover the full bbox."
        )
        tiles_to_fetch = all_tiles[: args.max_tiles]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for sw_lat, sw_lon in tiles_to_fetch:
        name = tile_name(sw_lat, sw_lon)
        path = download_tile(name, DATA_DIR)
        if path:
            downloaded.append(path)

    if not downloaded:
        log.error("No tiles downloaded successfully. Check --bbox and network access.")
        sys.exit(1)

    vrt_path = DATA_DIR / "landcover_mosaic.vrt"
    build_vrt(downloaded, vrt_path)
    log.info(
        f"Done — {len(downloaded)}/{len(tiles_to_fetch)} tiles downloaded. "
        f"VRT: {vrt_path}"
    )


if __name__ == "__main__":
    main()
