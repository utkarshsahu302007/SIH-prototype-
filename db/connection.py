"""Return a psycopg2 connection or SQLAlchemy engine from DATABASE_URL env var."""

import os

import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Default: local DB named 'hotspots', no password.
# Override with: export DATABASE_URL=postgresql://user:pass@host:5432/dbname
_DEFAULT_URL = "postgresql://localhost/hotspots"


def _url() -> str:
    raw = os.environ.get("DATABASE_URL", _DEFAULT_URL)
    # SQLAlchemy 2.x needs the +psycopg2 dialect tag explicitly.
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+psycopg2://", 1)
    return raw


def get_conn() -> psycopg2.extensions.connection:
    """Open a raw psycopg2 connection (for DDL, bulk inserts, and writes)."""
    raw_url = os.environ.get("DATABASE_URL", _DEFAULT_URL)
    return psycopg2.connect(raw_url)


def get_engine():
    """Return a SQLAlchemy engine (needed by geopandas read_postgis)."""
    return create_engine(_url())
