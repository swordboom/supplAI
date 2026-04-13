"""
anomaly_detector.py
--------------------
Detects anomalous shipment patterns per city using Isolation Forest.

Features used (all derived from order_large.csv per SOURCE city):
  - total_orders      : how many orders the city ships
  - avg_weight_kg     : average shipment weight
  - weight_cv         : coefficient of variation of weight (irregular = high)
  - danger_ratio      : fraction of orders that are type_3 or type_4
  - avg_lead_days     : average days between available and deadline
  - min_lead_days     : shortest lead time (panic orders = very low)
  - orders_per_edge   : order volume normalised by number of outgoing edges
  - weight_spike_flag : 1 if max weight > 4x median weight

Cities with very few edges (no outbound orders) are given a synthetic
"silent" feature vector that scores as anomalous.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDER_PATH   = PROJECT_ROOT / "datasets" / "order_large.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
ANOMALY_PATH = MODEL_DIR / "anomaly_model.pkl"

ANOMALY_THRESHOLD = -0.05   # below this = anomalous (sklearn IF range: ~-0.5 to 0.5)


# ---------------------------------------------------------------------------
# Build city-level feature matrix from order_large.csv
# ---------------------------------------------------------------------------
def _build_city_features(order_path: Path = ORDER_PATH) -> pd.DataFrame:
    """
    Aggregate order_large.csv into one row per SOURCE city with
    statistically meaningful features.
    """
    dtypes = {
        "Order_ID":       str,
        "Source":         str,
        "Destination":    str,
        "Danger_Type":    str,
        "Weight":         float,
        "Area":           float,
    }
    df = pd.read_csv(order_path, dtype=dtypes, parse_dates=["Available_Time", "Deadline"])

    # Lead time in days
    df["lead_days"] = (df["Deadline"] - df["Available_Time"]).dt.total_seconds() / 86400
    df["lead_days"] = df["lead_days"].clip(lower=0)

    # Danger flag
    df["is_dangerous"] = df["Danger_Type"].isin(["type_3", "type_4"]).astype(int)

    # Weight in kg (already in grams in some versions — normalise by 1000 if huge)
    df["weight_kg"] = df["Weight"].where(df["Weight"] < 1e9, df["Weight"] / 1000)

    grp = df.groupby("Source")

    feat = pd.DataFrame({
        "total_orders":   grp["Order_ID"].count(),
        "avg_weight_kg":  grp["weight_kg"].mean(),
        "std_weight_kg":  grp["weight_kg"].std().fillna(0),
        "max_weight_kg":  grp["weight_kg"].max(),
        "median_weight":  grp["weight_kg"].median(),
        "danger_ratio":   grp["is_dangerous"].mean(),
        "avg_lead_days":  grp["lead_days"].mean(),
        "min_lead_days":  grp["lead_days"].min(),
        "n_destinations": grp["Destination"].nunique(),
    }).reset_index().rename(columns={"Source": "city_id"})

    # Derived features
    feat["weight_cv"] = feat["std_weight_kg"] / (feat["avg_weight_kg"] + 1e-6)
    feat["orders_per_dest"] = feat["total_orders"] / (feat["n_destinations"] + 1)
    feat["weight_spike_flag"] = (feat["max_weight_kg"] > 4 * feat["median_weight"]).astype(float)

    feat = feat.fillna(0)
    return feat


FEATURE_COLS = [
    "total_orders", "avg_weight_kg", "weight_cv",
    "danger_ratio", "avg_lead_days", "min_lead_days",
    "orders_per_dest", "weight_spike_flag",
]


# ---------------------------------------------------------------------------
# Train Isolation Forest on city-level features
# ---------------------------------------------------------------------------
def _train_isolation_forest(
    order_path:  Path = ORDER_PATH,
    model_path:  Path = ANOMALY_PATH,
) -> Dict[str, Any]:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    print("  [anomaly] Building city-level features from order_large.csv …")
    feat_df = _build_city_features(order_path)

    X_raw = feat_df[FEATURE_COLS].values.astype(np.float32)

    # RobustScaler handles outliers better than StandardScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    print(f"  [anomaly] Training Isolation Forest on {len(X)} cities …")
    model = IsolationForest(
        n_estimators  = 300,
        contamination = 0.10,   # expect ~10% anomalous cities
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X)

    scores = model.decision_function(X)
    artifact = {
        "model":        model,
        "scaler":       scaler,
        "features":     FEATURE_COLS,
        "feat_df":      feat_df,         # cached city features (city_id → features)
        "score_mean":   float(np.mean(scores)),
        "score_std":    float(np.std(scores)),
        "score_p10":    float(np.percentile(scores, 10)),
        "score_p5":     float(np.percentile(scores, 5)),
        "train_rows":   len(X),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"  [anomaly] Model saved to {model_path}")
    print(f"  [anomaly] Score stats: mean={artifact['score_mean']:.4f}  "
          f"std={artifact['score_std']:.4f}  p5={artifact['score_p5']:.4f}")
    return artifact


def load_or_train_anomaly(
    order_path:    Path = ORDER_PATH,
    model_path:    Path = ANOMALY_PATH,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    if model_path.exists() and not force_retrain:
        print(f"  [anomaly] Loading cached model from {model_path}")
        return joblib.load(model_path)
    return _train_isolation_forest(order_path, model_path)


def load_anomaly_model(
    model_path: Path = ANOMALY_PATH,
) -> Dict[str, Any]:
    """
    Load an already-trained anomaly artifact.
    Raises FileNotFoundError when model is absent.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Anomaly model not found at {model_path}. "
            "Train it first with: python train_models.py"
        )

    artifact = joblib.load(model_path)
    required = {"model", "scaler", "features", "feat_df"}
    missing = sorted(required - set(artifact.keys()))
    if missing:
        raise ValueError(
            f"Invalid anomaly model artifact at {model_path}: missing {', '.join(missing)}"
        )
    return artifact


# ---------------------------------------------------------------------------
# Score all graph nodes
# ---------------------------------------------------------------------------
def score_anomalies(
    artifact:  Dict[str, Any],
    G,
    supply_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Score every node in G using the pre-built city feature table.
    Nodes with no orders get a synthetic 'silent' vector (high anomaly).
    """
    model    = artifact["model"]
    scaler   = artifact["scaler"]
    feat_df  = artifact["feat_df"].set_index("city_id")

    score_mean = artifact["score_mean"]
    score_std  = artifact["score_std"]

    # Worst (most anomalous) score seen, used to fill silent nodes
    silent_vec = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)

    rows = []
    for node in G.nodes():
        node_data = G.nodes[node]

        if node in feat_df.index:
            x_raw = feat_df.loc[node, FEATURE_COLS].values.astype(np.float32).reshape(1, -1)
            x     = scaler.transform(x_raw)
            score = float(model.decision_function(x)[0])
        else:
            # City has no outbound orders → silent → anomalous
            x     = scaler.transform(silent_vec)
            score = float(model.decision_function(x)[0]) - 0.05   # nudge lower

        z = (score - score_mean) / (score_std + 1e-9)

        if score < artifact["score_p5"]:
            level = "High Anomaly"
        elif score < ANOMALY_THRESHOLD:
            level = "Moderate Anomaly"
        else:
            level = "Normal"

        rows.append({
            "node":          node,
            "city_name":     node_data.get("city_name", node),
            "country":       node_data.get("country",  "Unknown"),
            "product":       node_data.get("product_category", "General"),
            "anomaly_score": round(score, 5),
            "anomaly_z":     round(z,     3),
            "anomaly_level": level,
            "is_anomalous":  score < ANOMALY_THRESHOLD,
        })

    df = pd.DataFrame(rows)
    df.sort_values("anomaly_score", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Plotly bar chart
# ---------------------------------------------------------------------------
def anomaly_bar_figure(anomaly_df: pd.DataFrame, top_n: int = 20):
    import plotly.graph_objects as go

    if anomaly_df.empty:
        return go.Figure()

    df = anomaly_df.head(top_n).copy()
    df = df.sort_values("anomaly_score", ascending=True)

    p5_threshold = anomaly_df["anomaly_score"].quantile(0.05)
    colours = [
        "#ef4444" if s < p5_threshold
        else ("#f97316" if s < ANOMALY_THRESHOLD else "#22c55e")
        for s in df["anomaly_score"]
    ]

    fig = go.Figure(go.Bar(
        x = df["anomaly_score"],
        y = df["city_name"] + " (" + df["country"] + ")",
        orientation = "h",
        marker      = dict(color=colours, line=dict(width=0)),
        text        = [f"{s:.4f}" for s in df["anomaly_score"]],
        textposition = "outside",
        textfont    = dict(color="#94a3b8", size=10),
        hovertemplate = "<b>%{y}</b><br>Anomaly Score: %{x:.5f}<extra></extra>",
    ))

    fig.add_vline(
        x=ANOMALY_THRESHOLD, line_dash="dash",
        line_color="#f97316", line_width=1.5,
        annotation_text="Anomaly threshold",
        annotation_font_color="#f97316",
    )

    fig.update_layout(
        title       = dict(text=f"Shipment Anomaly Detection — Top {top_n} Nodes",
                           font=dict(color="#e2e8f0", size=14, family="Inter"), x=0),
        paper_bgcolor = "#0a0e1a",
        plot_bgcolor  = "#0a0e1a",
        xaxis = dict(title="Isolation Forest Score (lower = more anomalous)",
                     color="#94a3b8", gridcolor="#1e293b"),
        yaxis = dict(color="#e2e8f0", tickfont=dict(size=10)),
        font  = dict(color="#e2e8f0", family="Inter"),
        height = 500,
        margin = dict(t=40, b=40, l=10, r=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph

    artifact = load_or_train_anomaly(force_retrain=True)
    G        = build_graph()
    df       = score_anomalies(artifact, G)

    print(f"\nAnomaly scores computed for {len(df)} nodes")
    print(f"Normal nodes    : {(df['anomaly_level'] == 'Normal').sum()}")
    print(f"Moderate anomaly: {(df['anomaly_level'] == 'Moderate Anomaly').sum()}")
    print(f"High anomaly    : {(df['anomaly_level'] == 'High Anomaly').sum()}")
    print(f"\nScore spread: min={df['anomaly_score'].min():.4f}  "
          f"max={df['anomaly_score'].max():.4f}  "
          f"std={df['anomaly_score'].std():.4f}")
    print("\nTop 15 most anomalous:")
    print(df[["city_name","country","anomaly_score","anomaly_level"]].head(15).to_string(index=False))
