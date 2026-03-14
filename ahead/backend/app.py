"""
app.py — Flask + Socket.IO server for Ahead ambulance signal preemption demo.
Uses threading async_mode for broad compatibility.
"""

import math
import os
import time
import threading

from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO, emit

from route_predictor import (
    load_graph,
    train_model,
    precompute_edge_scores,
    predict_route,
    subsample_intersections,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "ahead-demo-secret"

# Handle CORS so the frontend (opened as file://) can reach Flask endpoints.
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        res = make_response()
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return res

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ---------------------------------------------------------------------------
# Demo scenario
# ---------------------------------------------------------------------------

TIMES_SQUARE = (40.7580, -73.9855)      # origin lat, lon
BELLEVUE_HOSPITAL = (40.7394, -73.9748) # destination lat, lon

# ---------------------------------------------------------------------------
# Global state (single-session demo)
# ---------------------------------------------------------------------------

G = None
edge_scores = None

_sim_running = False
_sim_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Startup: load graph + model
# ---------------------------------------------------------------------------

def startup():
    global G, edge_scores
    with app.app_context():
        print("[app] Loading graph …")
        G = load_graph()
        clf = train_model(G)
        edge_scores = precompute_edge_scores(G, clf)
        print("[app] Ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heading(lat1, lon1, lat2, lon2) -> float:
    """Return compass heading in degrees from point 1 to point 2."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    angle = math.degrees(math.atan2(dlon, dlat))
    return (angle + 360) % 360


def _interpolate_position(timeline: list, elapsed_demo_s: float):
    """
    Linearly interpolate ambulance lat/lon and heading given elapsed demo time.
    Returns (lat, lon, heading_deg, node_index).
    """
    if not timeline:
        return None

    # Find surrounding waypoints
    for i in range(len(timeline) - 1):
        t0 = timeline[i]["arrival_s"]
        t1 = timeline[i + 1]["arrival_s"]
        if t0 <= elapsed_demo_s <= t1:
            frac = (elapsed_demo_s - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            lat = timeline[i]["lat"] + frac * (timeline[i + 1]["lat"] - timeline[i]["lat"])
            lon = timeline[i]["lon"] + frac * (timeline[i + 1]["lon"] - timeline[i]["lon"])
            heading = _heading(
                timeline[i]["lat"], timeline[i]["lon"],
                timeline[i + 1]["lat"], timeline[i + 1]["lon"],
            )
            return lat, lon, heading, i
    # Past end
    last = timeline[-1]
    return last["lat"], last["lon"], 0.0, len(timeline) - 1


def _current_ambulance_intersection_index(intersections, elapsed_demo_s):
    """Return index of the most recent passed intersection."""
    idx = -1
    for i, inter in enumerate(intersections):
        if inter["arrival_s"] <= elapsed_demo_s:
            idx = i
    return idx


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------

def _simulation_thread(timeline: list, intersections: list, total_demo_s: float):
    global _sim_running

    time_saved_s = 0.0
    preempted_set = set()
    start_wall = time.time()

    TICK = 0.5  # seconds between updates

    while True:
        with _sim_lock:
            if not _sim_running:
                break

        elapsed = time.time() - start_wall

        # Interpolate position
        result = _interpolate_position(timeline, elapsed)
        if result is None:
            break
        lat, lon, heading, node_idx = result

        progress_pct = min(100.0, elapsed / total_demo_s * 100)

        # Determine current ambulance intersection index
        amb_inter_idx = _current_ambulance_intersection_index(intersections, elapsed)

        # Build signal states
        signal_states = {}
        for i, inter in enumerate(intersections):
            node_id = str(inter["node_id"])
            delta = i - amb_inter_idx

            if delta > 3:
                state = "normal"
            elif delta == 3:
                state = "yellow"
            elif 1 <= delta <= 2:
                state = "green"
            elif delta == 0:
                state = "at"
            else:  # delta < 0
                state = "passed"

            # Track preemption (green or at state, not yet counted)
            if state in ("green", "at") and node_id not in preempted_set:
                preempted_set.add(node_id)
                time_saved_s += 30.0

            signal_states[node_id] = state

        socketio.emit("position_update", {
            "ambulance": {
                "lat": lat,
                "lon": lon,
                "heading_deg": heading,
            },
            "signal_states": signal_states,
            "time_saved_s": time_saved_s,
            "progress_pct": progress_pct,
        })

        if elapsed >= total_demo_s:
            break

        time.sleep(TICK)

    # Emit completion
    socketio.emit("simulation_complete", {
        "total_time_saved_s": time_saved_s,
        "intersections_preempted": len(preempted_set),
    })

    with _sim_lock:
        _sim_running = False

    print(f"[app] Simulation complete. Time saved: {time_saved_s:.0f}s, Intersections preempted: {len(preempted_set)}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


@app.route("/<path:filename>")
def static_frontend(filename):
    from flask import send_from_directory
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/start", methods=["POST"])
def start_demo():
    global _sim_running

    with _sim_lock:
        if _sim_running:
            return jsonify({"status": "already_running"}), 400
        _sim_running = True

    print("[app] Computing route …")
    timeline = predict_route(
        G, edge_scores,
        TIMES_SQUARE[0], TIMES_SQUARE[1],
        BELLEVUE_HOSPITAL[0], BELLEVUE_HOSPITAL[1],
    )

    intersections = subsample_intersections(timeline)
    total_demo_s = timeline[-1]["arrival_s"] if timeline else 45.0

    # Emit route_ready
    route_coords = [{"lat": t["lat"], "lon": t["lon"]} for t in timeline]
    inter_payload = [
        {
            "lat": inter["lat"],
            "lon": inter["lon"],
            "node_id": str(inter["node_id"]),
            "arrival_s": inter["arrival_s"],
            "index": i,
        }
        for i, inter in enumerate(intersections)
    ]

    socketio.emit("route_ready", {
        "route": route_coords,
        "intersections": inter_payload,
        "estimated_total_time_s": total_demo_s,
    })

    # Launch simulation background thread
    socketio.start_background_task(
        _simulation_thread, timeline, intersections, total_demo_s
    )

    print(f"[app] Route ready: {len(timeline)} waypoints, {len(intersections)} intersections, {total_demo_s:.1f}s demo time")
    return jsonify({
        "status": "started",
        "waypoints": len(timeline),
        "intersections": len(intersections),
        "estimated_total_time_s": total_demo_s,
    })


@app.route("/stop", methods=["POST"])
def stop_demo():
    global _sim_running
    with _sim_lock:
        _sim_running = False
    return jsonify({"status": "stopped"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "graph_loaded": G is not None})


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    print("[app] Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("[app] Client disconnected")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    startup()
    print("[app] Starting server on http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
