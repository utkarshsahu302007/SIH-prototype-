# Frontend Map UI

A standalone single-page web app built with vanilla HTML, CSS, and Leaflet.js. Zero build step, zero npm dependencies.

## How to View

### Option 1: Direct File Opening
Double-click `frontend/index.html` in your file explorer, or open it in any modern browser:
```bash
open frontend/index.html      # macOS
# xdg-open frontend/index.html # Linux
```

### Option 2: Local HTTP Server (Recommended)
From the project root:
```bash
python3 -m http.server 3000 --directory frontend
```
Then visit: `http://localhost:3000`

---

## Features

- **India-Centered Map**: Styled with high-contrast CartoDB dark basemap tiles.
- **Categorical Color Coding**:
  - 🔴 `industrial_event` (Red / pulsating prominence)
  - 🟠 `industrial_persistent` (Orange / gas flares, refineries, power plants)
  - 🟡 `agricultural_burning` (Gold / crop-residue burns)
  - 🟣 `forest_fire` (Purple / wildfires)
  - 🔵 `mining` (Blue / quarries)
  - ⚪ `other_false_positive` (Slate gray / sunglint, cloud edges, road glare)
- **Dynamic Filtering**:
  - Class toggles with live count tallies
  - Date range pickers (From / To)
- **Detailed Popups**: Shows predicted class, ML confidence score, FRP (MW), brightness temp, acquisition date/time, nearest facility type & distance, and land cover class.
- **Industrial Events Feed**: Sidebar panel listing high-priority industrial incidents. Clicking any event smoothly flies the map to that hotspot and opens its detail popup.
- **Facilities Overlay**: Optional toggle to visualize OSM industrial facility geometries and quarries.
- **Configurable API Target**: Top bar lets you point the frontend to any FastAPI base URL (persisted in browser `localStorage`).
