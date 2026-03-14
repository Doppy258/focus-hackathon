# Ahead — Predictive Ambulance Signal Preemption

> "Green wave, 2–3 intersections ahead. Not reactive. Predictive."

## Setup

### 1. Install dependencies

```bash
cd ahead/backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-cache the Manhattan road network

This downloads ~50 MB of OSM data and caches it locally. Only needed once.

```bash
cd ahead/backend
source venv/bin/activate
python3 -c "from route_predictor import load_graph; load_graph()"
```

Expect this to take ~60 seconds on first run.

### 3. Start the server

```bash
cd ahead/backend
source venv/bin/activate
python3 app.py
```

You should see:
```
[app] Loading graph …
[route_predictor] Graph loaded: XXXXX nodes, XXXXX edges
[route_predictor] Generating synthetic training data …
[route_predictor] Training on XXXXX samples …
[route_predictor] Model trained and cached.
[route_predictor] Precomputing edge scores …
[app] Ready.
[app] Starting server on http://localhost:5000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser (no web server needed — it's fully self-contained).

### 5. Run the demo

1. Click **Start Demo**
2. Watch the route appear from Times Square to Bellevue Hospital
3. Observe the green wave washing 2–3 intersections ahead of the ambulance 🚑
4. Watch the **Time Saved** counter tick up at each preempted intersection
5. See the summary modal on arrival (~45 seconds demo time)

---

## How It Works

| Component | What it does |
|-----------|-------------|
| `route_predictor.py` | Downloads Manhattan road network via OSMnx, trains a Random Forest on synthetic ambulance-route data, predicts the optimal route weighted by ML scores + highway preference + time-of-day traffic |
| `app.py` | Flask + Socket.IO server; runs the signal state machine in a background thread, emitting position updates every 500ms |
| `frontend/index.html` | Leaflet map with real-time signal markers, ambulance icon with heading rotation, live stats panel |

### Signal State Machine

```
delta = intersection_index - current_ambulance_intersection_index

delta > 3  →  normal  (red cycling)
delta == 3 →  yellow  (warning: preemption incoming)
delta 1-2  →  green   (preempted — clear passage)
delta == 0 →  at      (ambulance here)
delta < 0  →  passed  (returned to normal)
```

### ML Model

- Random Forest (100 trees) trained on 500 synthetic (origin, dest) route pairs
- Features: highway preference, speed, length, one-way flag, lane count
- Edge score = 0.6 × RF probability + 0.4 × highway preference
- Route weight = travel_time / (0.1 + score × highway_pref)

---

## Files

```
ahead/
├── backend/
│   ├── app.py              Flask + SocketIO server + simulation loop
│   ├── route_predictor.py  ML model: OSMnx graph, RF training, route prediction
│   └── requirements.txt
├── frontend/
│   └── index.html          Leaflet map + Socket.IO (self-contained)
└── README.md
```
