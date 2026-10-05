# FIRMS Hotspot Classifier

AI system to classify NASA FIRMS thermal hotspot detections into categories (industrial, agricultural, forest fire, etc.) using PostGIS and LightGBM.

## 🚀 Collaborator Quickstart

Follow these steps exactly to get the project running on your local machine.

### 1. Database Setup (macOS)
The app requires PostgreSQL and the PostGIS spatial extension.
```bash
# Install and start PostgreSQL
brew install postgresql postgis
brew services start postgresql

# Create the database and apply the schema
createdb hotspots
psql -d hotspots -f db/schema.sql
```

### 2. Environment Setup
```bash
# Create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

### 3. API Keys Configuration
NASA requires a free map key to pull their satellite data.
1. Get a key instantly here: [NASA FIRMS API Key](https://firms.modaps.eosdis.nasa.gov/api/map_key)
2. Copy the template: `cp .env.example .env`
3. Open `.env` and paste your key into `FIRMS_MAP_KEY`.

### 4. Download Data & Train Model
Run the data pipeline to fetch raw satellite/map data, engineer features, and train the AI model.
*(Note: These scripts default to a small test region in India for quick POC testing)*
```bash
# 1. Download NASA & OpenStreetMap Data
python3 data_ingestion/fetch_firms.py --start 2024-01-01 --end 2024-01-07 --bbox 83.0,21.0,85.0,23.0
python3 data_ingestion/fetch_osm_facilities.py --bbox 21.0,83.0,23.0,85.0
python3 db/load_data.py

# 2. Extract Features
python3 features/spatial_join.py
python3 features/temporal_features.py
python3 features/build_dataset.py

# 3. Label & Train
python3 features/weak_labels.py
python3 model/train.py
python3 model/predict.py
```

### 5. Run the Application!
Start the backend API and frontend dashboard:
```bash
python3 -m uvicorn api.main:app --reload --port 8000
```
Open [http://localhost:8000/](http://localhost:8000/) in your browser.
