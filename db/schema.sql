-- Apply with: psql $DATABASE_URL -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Raw VIIRS hotspot data from NASA FIRMS ─────────────────────────────────
-- Column names match the FIRMS VIIRS CSV exactly (bright_ti4/ti5, not brightness).
CREATE TABLE IF NOT EXISTS hotspots (
    id          SERIAL PRIMARY KEY,
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    bright_ti4  DOUBLE PRECISION,           -- brightness temp I-4 band (~4 µm), Kelvin
    bright_ti5  DOUBLE PRECISION,           -- brightness temp I-5 band (~11 µm), Kelvin
    frp         DOUBLE PRECISION,           -- fire radiative power, MW
    confidence  VARCHAR(10),                -- l / n / h  (low / nominal / high)
    acq_date    DATE,
    acq_time    VARCHAR(6),                 -- HHMM string
    satellite   VARCHAR(10),               -- N = Suomi-NPP, 1 = NOAA-20
    daynight    CHAR(1),                   -- D or N
    source_file VARCHAR(255),              -- CSV filename this row came from
    geom        geometry(Point, 4326)
);

CREATE INDEX IF NOT EXISTS hotspots_geom_idx  ON hotspots USING GIST (geom);
CREATE INDEX IF NOT EXISTS hotspots_date_idx  ON hotspots (acq_date);

-- ── OSM industrial facilities ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facilities (
    id              SERIAL PRIMARY KEY,
    osm_id          BIGINT,
    osm_type        VARCHAR(20),            -- node / way / relation
    facility_type   VARCHAR(50),            -- derived from tags; see load_data.py
    name            VARCHAR(255),
    geom            geometry(Geometry, 4326)
);

CREATE INDEX IF NOT EXISTS facilities_geom_idx ON facilities USING GIST (geom);

-- ── ESA WorldCover class code → human-readable name ────────────────────────
CREATE TABLE IF NOT EXISTS land_cover_lookup (
    class_code  INTEGER PRIMARY KEY,
    class_name  VARCHAR(50) NOT NULL
);

INSERT INTO land_cover_lookup (class_code, class_name) VALUES
    (10,  'tree_cover'),
    (20,  'shrubland'),
    (30,  'grassland'),
    (40,  'cropland'),
    (50,  'built_up'),
    (60,  'bare_sparse_vegetation'),
    (70,  'snow_ice'),
    (80,  'permanent_water'),
    (90,  'herbaceous_wetland'),
    (95,  'mangroves'),
    (100, 'moss_lichen')
ON CONFLICT (class_code) DO NOTHING;

-- ── Spatial join results (one row per hotspot, computed by spatial_join.py) ─
CREATE TABLE IF NOT EXISTS hotspot_features (
    hotspot_id                      INTEGER PRIMARY KEY REFERENCES hotspots (id),
    nearest_facility_id             INTEGER REFERENCES facilities (id),
    nearest_facility_type           VARCHAR(50),
    distance_to_nearest_facility    DOUBLE PRECISION,   -- metres, in UTM projection
    land_cover_code                 INTEGER,
    land_cover_class                VARCHAR(50),
    computed_at                     TIMESTAMPTZ DEFAULT NOW()
);

-- ── ML predictions (populated by model/inference.py in a later phase) ──────
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    hotspot_id      INTEGER REFERENCES hotspots (id),
    predicted_class VARCHAR(50),
    confidence      DOUBLE PRECISION,
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS predictions_hotspot_idx ON predictions (hotspot_id);
