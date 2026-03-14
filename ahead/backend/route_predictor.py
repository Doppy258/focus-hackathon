"""
route_predictor.py — ML-powered ambulance route prediction for Ahead demo.

Loads Manhattan road network via OSMnx, trains a RandomForest model on
synthetic data, and predicts the optimal ambulance route with preemption-
aware edge weights.
"""

import os
import pickle
import random
import math
import numpy as np
import networkx as nx

# OSMnx v1.x API
import osmnx as ox

from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_CACHE = os.path.join(os.path.dirname(__file__), "manhattan_graph.pkl")
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "rf_model.pkl")

HIGHWAY_PREF = {
    "motorway": 0.9,
    "trunk": 0.88,
    "primary": 0.85,
    "secondary": 0.75,
    "tertiary": 0.5,
    "residential": 0.2,
    "unclassified": 0.15,
    "service": 0.1,
    "living_street": 0.1,
    "path": 0.05,
    "footway": 0.02,
    "cycleway": 0.02,
    "steps": 0.01,
}

# Demo speed-up: ambulance simulation runs 60x real time
SPEED_MULTIPLIER = 60.0
MAX_INTERSECTIONS = 12

# Time-of-day speed multipliers keyed by (start_hour, end_hour)
TIME_BANDS = [
    (7, 9, 0.55),    # morning rush
    (17, 19, 0.55),  # evening rush
    (11, 14, 0.80),  # midday
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_highway(data: dict) -> str:
    """Return the primary highway type from an edge data dict."""
    hw = data.get("highway", "unclassified")
    if isinstance(hw, list):
        hw = hw[0]
    return hw or "unclassified"


def _highway_pref(hw: str) -> float:
    return HIGHWAY_PREF.get(hw, 0.15)


def _time_of_day_multiplier() -> float:
    """Return speed multiplier based on current wall-clock hour."""
    import datetime
    hour = datetime.datetime.now().hour
    for start, end, mult in TIME_BANDS:
        if start <= hour < end:
            return mult
    return 1.0


# ---------------------------------------------------------------------------
# Graph loading / caching
# ---------------------------------------------------------------------------

def load_graph() -> nx.MultiDiGraph:
    """Load Manhattan road network, using disk cache if available."""
    if os.path.exists(GRAPH_CACHE):
        print("[route_predictor] Loading graph from cache …")
        with open(GRAPH_CACHE, "rb") as f:
            G = pickle.load(f)
        print(f"[route_predictor] Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G

    print("[route_predictor] Downloading Manhattan road network (this may take ~1 min) …")
    G = ox.graph_from_place(
        "Manhattan, New York City, New York, USA",
        network_type="drive",
        retain_all=False,
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(G, f)
    print(f"[route_predictor] Graph cached: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _edge_features(u: int, v: int, data: dict) -> list:
    """Return numeric feature vector for a single edge."""
    hw = _get_highway(data)
    pref = _highway_pref(hw)
    speed = float(data.get("speed_kph", 40.0))
    length = float(data.get("length", 50.0))
    oneway = 1.0 if data.get("oneway", False) else 0.0
    lanes_raw = data.get("lanes", 1)
    if isinstance(lanes_raw, list):
        lanes_raw = lanes_raw[0]
    lanes = float(lanes_raw) if lanes_raw else 1.0
    return [pref, speed, length, oneway, lanes]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(G: nx.MultiDiGraph) -> RandomForestClassifier:
    """Train RandomForest on synthetic ambulance-route data."""
    if os.path.exists(MODEL_CACHE):
        print("[route_predictor] Loading RF model from cache …")
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    print("[route_predictor] Generating synthetic training data …")
    nodes = list(G.nodes())
    random.seed(42)

    X, y = [], []
    sample_pairs = 80

    for _ in range(sample_pairs):
        orig, dest = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, orig, dest, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        path_edges = set(zip(path[:-1], path[1:]))

        # Positive examples: edges on path
        for u, v in path_edges:
            if v in G[u]:
                data = G[u][v][0]
                X.append(_edge_features(u, v, data))
                y.append(1)

        # Negative examples: equal number of random non-path edges
        all_edges = list(G.edges())
        neg_sample = random.sample(all_edges, min(len(path_edges), len(all_edges)))
        for u, v in neg_sample:
            if (u, v) not in path_edges and v in G[u]:
                data = G[u][v][0]
                hw = _get_highway(data)
                label = 1 if hw in ("motorway", "trunk", "primary", "secondary") else 0
                X.append(_edge_features(u, v, data))
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"[route_predictor] Training on {len(X)} samples …")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(clf, f)
    print("[route_predictor] Model trained and cached.")
    return clf


# ---------------------------------------------------------------------------
# Edge scoring
# ---------------------------------------------------------------------------

def precompute_edge_scores(G: nx.MultiDiGraph, clf: RandomForestClassifier) -> dict:
    """
    Return dict {(u,v,key): score} for all edges.
    Score = P(ambulance-preferred) blended with highway preference.
    """
    print("[route_predictor] Precomputing edge scores …")
    edge_scores = {}
    features = []
    edge_keys = []

    for u, v, k, data in G.edges(keys=True, data=True):
        features.append(_edge_features(u, v, data))
        edge_keys.append((u, v, k))

    probs = clf.predict_proba(np.array(features))[:, 1]

    for (u, v, k), prob in zip(edge_keys, probs):
        data = G[u][v][k]
        hw = _get_highway(data)
        hw_pref = _highway_pref(hw)
        # Blend ML score with deterministic highway preference
        edge_scores[(u, v, k)] = 0.6 * prob + 0.4 * hw_pref

    print(f"[route_predictor] Scored {len(edge_scores)} edges.")
    return edge_scores


# ---------------------------------------------------------------------------
# Route prediction
# ---------------------------------------------------------------------------

def predict_route(
    G: nx.MultiDiGraph,
    edge_scores: dict,
    orig_lat: float,
    orig_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> list:
    """
    Predict optimal ambulance route, returning a timeline of nodes.

    Returns list of dicts:
        {node_id, lat, lon, arrival_s, is_intersection}
    """
    tod_mult = _time_of_day_multiplier()

    # Stamp ambulance_weight on every edge
    for u, v, k, data in G.edges(keys=True, data=True):
        score = edge_scores.get((u, v, k), 0.2)
        hw = _get_highway(data)
        hw_pref = _highway_pref(hw)
        travel_time = float(data.get("travel_time", 30.0)) / tod_mult
        # Lower weight = more preferred
        data["ambulance_weight"] = travel_time / (0.1 + score * hw_pref)

    orig_node = ox.distance.nearest_nodes(G, X=orig_lon, Y=orig_lat)
    dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

    path = nx.shortest_path(G, orig_node, dest_node, weight="ambulance_weight")

    # Build timeline
    elapsed = 0.0
    timeline = []
    for i, node_id in enumerate(path):
        node = G.nodes[node_id]
        lat = node["y"]
        lon = node["x"]
        degree = G.degree(node_id)
        is_intersection = degree >= 3

        timeline.append({
            "node_id": node_id,
            "lat": lat,
            "lon": lon,
            "arrival_s": elapsed / SPEED_MULTIPLIER,  # demo time
            "is_intersection": is_intersection,
        })

        if i < len(path) - 1:
            next_node = path[i + 1]
            edge_data = G[node_id][next_node][0]
            tt = float(edge_data.get("travel_time", 5.0)) / tod_mult
            elapsed += tt

    return timeline


# ---------------------------------------------------------------------------
# Subsample intersections
# ---------------------------------------------------------------------------

def subsample_intersections(timeline: list, max_count: int = MAX_INTERSECTIONS) -> list:
    """
    Keep at most max_count intersections from the timeline, evenly spaced.
    Always include the first and last intersection.
    """
    intersections = [t for t in timeline if t["is_intersection"]]
    if len(intersections) <= max_count:
        return intersections

    indices = np.linspace(0, len(intersections) - 1, max_count, dtype=int)
    return [intersections[i] for i in indices]
