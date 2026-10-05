"""Pull VIIRS hotspot data from NASA FIRMS API and save one CSV per day."""

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# FIRMS Area API — SP = archive (historical), NRT = last 10 days
# URL pattern for SP: /api/area/csv/{key}/{source}/{W,S,E,N}/{start_date}/{day_range}
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
INDIA_BBOX = "68.0,8.0,97.5,37.5"  # W,S,E,N — FIRMS uses this order

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "firms"


def build_url(map_key: str, source: str, bbox: str, query_date: date) -> str:
    """Build FIRMS API URL for a single day. day_range=1 means exactly that date."""
    return f"{FIRMS_BASE_URL}/{map_key}/{source}/{bbox}/1/{query_date.isoformat()}"


def fetch_day(map_key: str, source: str, bbox: str, query_date: date) -> str:
    """Fetch VIIRS CSV for one day; one retry on non-2xx response."""
    url = build_url(map_key, source, bbox, query_date)
    log.info(f"GET {url}")
    resp = requests.get(url, timeout=60)
    if not resp.ok:
        log.warning(f"Status {resp.status_code} — retrying in 10 s...")
        time.sleep(10)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    return resp.text


def save_csv(text: str, query_date: date, source: str) -> Path:
    """Write CSV text to /data/raw/firms/ and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"viirs_{source.lower()}_{query_date.isoformat()}.csv"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NASA FIRMS VIIRS hotspot data as daily CSV files."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--bbox",
        default=INDIA_BBOX,
        help="Bounding box in W,S,E,N order (default: India)",
    )
    parser.add_argument(
        "--source",
        default="VIIRS_SNPP_SP",
        choices=[
            "VIIRS_SNPP_SP",
            "VIIRS_SNPP_NRT",
            "VIIRS_NOAA20_SP",
            "VIIRS_NOAA20_NRT",
        ],
        help=(
            "FIRMS data source. SP = archive (works for historical dates). "
            "NRT = last 10 days only. Default: VIIRS_SNPP_SP"
        ),
    )
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        print("\nERROR: FIRMS_MAP_KEY is missing!")
        print("1. Get a free API key instantly at: https://firms.modaps.eosdis.nasa.gov/api/map_key")
        print("2. Copy .env.example to .env:  cp .env.example .env")
        print("3. Paste your key into the .env file under FIRMS_MAP_KEY=\"your_key_here\"\n")
        sys.exit(1)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    if start_date > end_date:
        log.error("--start must be on or before --end")
        sys.exit(1)

    saved, failed = 0, 0
    current = start_date

    while current <= end_date:
        try:
            csv_text = fetch_day(map_key, args.source, args.bbox, current)
            # First line is header; subsequent lines are records
            record_count = max(0, csv_text.count("\n") - 1)
            out_path = save_csv(csv_text, current, args.source)
            log.info(f"  Saved: {out_path.name}  ({record_count} hotspots)")
            saved += 1
        except Exception as exc:
            log.error(f"  Failed for {current}: {exc}")
            failed += 1
        current += timedelta(days=1)

    log.info(f"Done — {saved} days saved, {failed} days failed.")


if __name__ == "__main__":
    main()
