"""FastAPI backend for FIRMS hotspot classification POC."""

import json
import logging
from datetime import date
from typing import Any, Generator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import psycopg2.extras

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="FIRMS Hotspot Classifier API",
    description="Query classified thermal hotspots, industrial facilities, and fire events.",
    version="1.0.0",
)

# Enable CORS for local static HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Generator:
    """Yield a psycopg2 database connection and close it after request completion."""
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def parse_bbox(bbox_str: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """Parse 'min_lon,min_lat,max_lon,max_lat' (W,S,E,N) string into float tuple."""
    if not bbox_str:
        return None
    try:
        parts = [float(p.strip()) for p in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError()
        min_lon, min_lat, max_lon, max_lat = parts
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError()
        return min_lon, min_lat, max_lon, max_lat
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat' (W,S,E,N)",
        )




@app.get("/hotspots")
def get_hotspots(
    bbox: Optional[str] = Query(None, description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'"),
    start_date: Optional[date] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="End date YYYY-MM-DD (inclusive)"),
    predicted_class: Optional[str] = Query(None, alias="class", description="Filter by predicted class"),
    limit: int = Query(2000, ge=1, le=10000, description="Max features to return"),
    offset: int = Query(0, ge=0, description="Query offset"),
    conn=Depends(get_db),
) -> dict[str, Any]:
    """Return classified hotspots as a GeoJSON FeatureCollection."""
    clauses = ["1=1"]
    params: list[Any] = []

    bbox_tuple = parse_bbox(bbox)
    if bbox_tuple:
        min_lon, min_lat, max_lon, max_lat = bbox_tuple
        clauses.append("h.geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)")
        params.extend([min_lon, min_lat, max_lon, max_lat])

    if start_date:
        clauses.append("h.acq_date >= %s")
        params.append(start_date)

    if end_date:
        clauses.append("h.acq_date <= %s")
        params.append(end_date)

    if predicted_class:
        clauses.append("p.predicted_class = %s")
        params.append(predicted_class)

    where_sql = " AND ".join(clauses)
    params.extend([limit, offset])

    sql = f"""
        SELECT 
            h.id,
            h.latitude,
            h.longitude,
            h.bright_ti4,
            h.bright_ti5,
            h.frp,
            h.confidence,
            h.acq_date,
            h.acq_time,
            h.satellite,
            h.daynight,
            ST_AsGeoJSON(h.geom) AS geojson_geom,
            p.predicted_class,
            p.confidence AS prediction_confidence,
            hf.nearest_facility_type,
            hf.distance_to_nearest_facility,
            hf.land_cover_class
        FROM hotspots h
        LEFT JOIN predictions p ON p.hotspot_id = h.id
        LEFT JOIN hotspot_features hf ON hf.hotspot_id = h.id
        WHERE {where_sql}
        ORDER BY h.acq_date DESC, h.id DESC
        LIMIT %s OFFSET %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    features = []
    for r in rows:
        geom = json.loads(r["geojson_geom"]) if r["geojson_geom"] else {
            "type": "Point",
            "coordinates": [r["longitude"], r["latitude"]],
        }
        props = {
            "id": r["id"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "bright_ti4": r["bright_ti4"],
            "bright_ti5": r["bright_ti5"],
            "frp": r["frp"],
            "confidence": r["confidence"],
            "acq_date": r["acq_date"].isoformat() if r["acq_date"] else None,
            "acq_time": r["acq_time"],
            "satellite": r["satellite"],
            "daynight": r["daynight"],
            "predicted_class": r["predicted_class"],
            "prediction_confidence": r["prediction_confidence"],
            "nearest_facility_type": r["nearest_facility_type"],
            "distance_to_nearest_facility": r["distance_to_nearest_facility"],
            "land_cover_class": r["land_cover_class"],
        }
        features.append({
            "type": "Feature",
            "id": r["id"],
            "geometry": geom,
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/facilities")
def get_facilities(
    bbox: Optional[str] = Query(None, description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'"),
    facility_type: Optional[str] = Query(None, description="Filter by facility type"),
    limit: int = Query(1000, ge=1, le=5000, description="Max facilities to return"),
    offset: int = Query(0, ge=0, description="Query offset"),
    conn=Depends(get_db),
) -> dict[str, Any]:
    """Return industrial facilities as a GeoJSON FeatureCollection."""
    clauses = ["1=1"]
    params: list[Any] = []

    bbox_tuple = parse_bbox(bbox)
    if bbox_tuple:
        min_lon, min_lat, max_lon, max_lat = bbox_tuple
        clauses.append("f.geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)")
        params.extend([min_lon, min_lat, max_lon, max_lat])

    if facility_type:
        clauses.append("f.facility_type = %s")
        params.append(facility_type)

    where_sql = " AND ".join(clauses)
    params.extend([limit, offset])

    sql = f"""
        SELECT 
            f.id,
            f.osm_id,
            f.osm_type,
            f.facility_type,
            f.name,
            ST_AsGeoJSON(f.geom) AS geojson_geom
        FROM facilities f
        WHERE {where_sql}
        LIMIT %s OFFSET %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    features = []
    for r in rows:
        geom = json.loads(r["geojson_geom"]) if r["geojson_geom"] else None
        if not geom:
            continue
        props = {
            "id": r["id"],
            "osm_id": r["osm_id"],
            "osm_type": r["osm_type"],
            "facility_type": r["facility_type"],
            "name": r["name"],
        }
        features.append({
            "type": "Feature",
            "id": r["id"],
            "geometry": geom,
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/hotspots/{hotspot_id}")
def get_hotspot_detail(
    hotspot_id: int,
    conn=Depends(get_db),
) -> dict[str, Any]:
    """Return full detail for one hotspot: raw VIIRS data, spatial features, and prediction."""
    sql = """
        SELECT 
            h.id,
            h.latitude,
            h.longitude,
            h.bright_ti4,
            h.bright_ti5,
            h.frp,
            h.confidence,
            h.acq_date,
            h.acq_time,
            h.satellite,
            h.daynight,
            h.source_file,
            ST_AsGeoJSON(h.geom) AS geojson_geom,
            hf.nearest_facility_id,
            hf.nearest_facility_type,
            hf.distance_to_nearest_facility,
            hf.land_cover_code,
            hf.land_cover_class,
            hf.computed_at AS features_computed_at,
            p.predicted_class,
            p.confidence AS prediction_confidence,
            p.model_version,
            p.created_at AS prediction_created_at
        FROM hotspots h
        LEFT JOIN hotspot_features hf ON hf.hotspot_id = h.id
        LEFT JOIN predictions p ON p.hotspot_id = h.id
        WHERE h.id = %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, [hotspot_id])
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Hotspot ID {hotspot_id} not found")

    geom = json.loads(row["geojson_geom"]) if row["geojson_geom"] else {
        "type": "Point",
        "coordinates": [row["longitude"], row["latitude"]],
    }

    return {
        "id": row["id"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "bright_ti4": row["bright_ti4"],
        "bright_ti5": row["bright_ti5"],
        "frp": row["frp"],
        "confidence": row["confidence"],
        "acq_date": row["acq_date"].isoformat() if row["acq_date"] else None,
        "acq_time": row["acq_time"],
        "satellite": row["satellite"],
        "daynight": row["daynight"],
        "source_file": row["source_file"],
        "geometry": geom,
        "features": {
            "nearest_facility_id": row["nearest_facility_id"],
            "nearest_facility_type": row["nearest_facility_type"],
            "distance_to_nearest_facility": row["distance_to_nearest_facility"],
            "land_cover_code": row["land_cover_code"],
            "land_cover_class": row["land_cover_class"],
            "computed_at": row["features_computed_at"].isoformat() if row["features_computed_at"] else None,
        },
        "prediction": {
            "predicted_class": row["predicted_class"],
            "confidence": row["prediction_confidence"],
            "model_version": row["model_version"],
            "created_at": row["prediction_created_at"].isoformat() if row["prediction_created_at"] else None,
        },
    }


@app.get("/events")
def get_events(
    limit: int = Query(100, ge=1, le=1000, description="Max events to return"),
    offset: int = Query(0, ge=0, description="Query offset"),
    conn=Depends(get_db),
) -> list[dict[str, Any]]:
    """Return hotspots classified as industrial_event, most recent first."""
    sql = """
        SELECT 
            h.id,
            h.latitude,
            h.longitude,
            h.bright_ti4,
            h.bright_ti5,
            h.frp,
            h.confidence,
            h.acq_date,
            h.acq_time,
            h.satellite,
            h.daynight,
            ST_AsGeoJSON(h.geom) AS geojson_geom,
            p.predicted_class,
            p.confidence AS prediction_confidence,
            p.model_version,
            hf.nearest_facility_id,
            hf.nearest_facility_type,
            hf.distance_to_nearest_facility,
            hf.land_cover_class,
            f.name AS facility_name
        FROM hotspots h
        JOIN predictions p ON p.hotspot_id = h.id
        LEFT JOIN hotspot_features hf ON hf.hotspot_id = h.id
        LEFT JOIN facilities f ON f.id = hf.nearest_facility_id
        WHERE p.predicted_class = 'industrial_event'
        ORDER BY h.acq_date DESC, h.acq_time DESC, h.id DESC
        LIMIT %s OFFSET %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, [limit, offset])
        rows = cur.fetchall()

    events = []
    for r in rows:
        geom = json.loads(r["geojson_geom"]) if r["geojson_geom"] else {
            "type": "Point",
            "coordinates": [r["longitude"], r["latitude"]],
        }
        events.append({
            "id": r["id"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "bright_ti4": r["bright_ti4"],
            "bright_ti5": r["bright_ti5"],
            "frp": r["frp"],
            "confidence": r["confidence"],
            "acq_date": r["acq_date"].isoformat() if r["acq_date"] else None,
            "acq_time": r["acq_time"],
            "satellite": r["satellite"],
            "daynight": r["daynight"],
            "predicted_class": r["predicted_class"],
            "prediction_confidence": r["prediction_confidence"],
            "model_version": r["model_version"],
            "nearest_facility_id": r["nearest_facility_id"],
            "nearest_facility_type": r["nearest_facility_type"],
            "facility_name": r["facility_name"],
            "distance_to_nearest_facility": r["distance_to_nearest_facility"],
            "land_cover_class": r["land_cover_class"],
            "geometry": geom,
        })

    return events


app.mount("/", StaticFiles(directory=str(Path(__file__).parent.parent / "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
