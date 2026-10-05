# API Server

FastAPI backend connecting directly to PostGIS using psycopg2.

## Starting the Server

Make sure your database connection is exported:
```bash
export DATABASE_URL=postgresql://localhost/hotspots
# or with user/password:
# export DATABASE_URL=postgresql://user:password@host:5432/hotspots
```

Start the API:
```bash
uvicorn api.main:app --reload --port 8000
# or:
python api/main.py
```

Interactive OpenAPI docs will be available at:
`http://localhost:8000/docs`

---

## Endpoints

### 1. `GET /hotspots`
Returns classified hotspots as a GeoJSON FeatureCollection.

**Query Parameters:**
- `bbox`: Bounding box in `min_lon,min_lat,max_lon,max_lat` (W,S,E,N) format, e.g. `72.0,18.0,80.0,24.0`
- `start_date`: Filter by acquisition date on or after `YYYY-MM-DD`
- `end_date`: Filter by acquisition date on or before `YYYY-MM-DD`
- `class`: Filter by predicted classification label (e.g. `industrial_persistent`, `industrial_event`, `agricultural_burning`, `forest_fire`, `mining`, `other_false_positive`)
- `limit`: Default `2000`, max `10000`
- `offset`: Default `0`

**Example:**
```bash
curl "http://localhost:8000/hotspots?class=industrial_event&limit=50"
```

---

### 2. `GET /facilities`
Returns OpenStreetMap industrial and mining facilities as a GeoJSON FeatureCollection.

**Query Parameters:**
- `bbox`: Bounding box in `min_lon,min_lat,max_lon,max_lat`
- `facility_type`: Filter by facility type (`power_plant`, `refinery`, `steel_works`, `industrial_works`, `quarry`, `industrial`)
- `limit`: Default `1000`, max `5000`
- `offset`: Default `0`

**Example:**
```bash
curl "http://localhost:8000/facilities?bbox=72.0,18.0,80.0,24.0"
```

---

### 3. `GET /hotspots/{id}`
Returns full detail for a single hotspot: raw VIIRS data, spatial join features, and ML prediction with confidence score.

**Example:**
```bash
curl "http://localhost:8000/hotspots/42"
```

---

### 4. `GET /events`
Returns all hotspots predicted as `industrial_event` in descending chronological order (most recent first).

**Query Parameters:**
- `limit`: Default `100`, max `1000`
- `offset`: Default `0`

**Example:**
```bash
curl "http://localhost:8000/events?limit=20"
```
