"""
shap_explain.py
---------------
ML explainability layer for the delay prediction model using SHAP
(SHapley Additive exPlanations).

Provides:
  compute_shap()          â†’ dict of {node_id: {feature: shap_val}} for top-N nodes
  shap_bar_figure()       â†’ Plotly Figure â€” global mean |SHAP| feature importance
  shap_waterfall_figure() â†’ Plotly Figure â€” per-node waterfall (push/pull per feature)
  shap_to_text()          â†’ plain-English summary string (for Gemini prompt injection)

Works with both XGBoost (TreeExplainer, fast) and RandomForest (TreeExplainer) backends.
Falls back gracefully if SHAP is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Feature display names â€” human-readable labels for the 13 model features
# ---------------------------------------------------------------------------
FEATURE_LABELS: Dict[str, str] = {
    "distance":              "Route Distance",
    "shipment_weight":       "Shipment Weight",
    "SLA":                   "SLA (days)",
    "pickup_metro":          "Pickup Metro",
    "pickup_non_metro":      "Pickup Non-Metro",
    "drop_metro":            "Drop Metro",
    "drop_non_metro":        "Drop Non-Metro",
    "cp_delay_per_quarter":  "Carrier Delay Rate (Q)",
    "cp_ontime_per_quarter": "Carrier On-Time Rate (Q)",
    "cp_delay_per_month":    "Carrier Delay Rate (M)",
    "cp_ontime_per_month":   "Carrier On-Time Rate (M)",
    "holiday_in_between":    "Holiday En Route",
    "is_sunday_in_between":  "Sunday En Route",
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "distance":              "Total route distance in km. Longer routes have more checkpoints, border crossings, and exposure to regional disruptions â€” each adding delay probability.",
    "shipment_weight":       "Cargo weight. Heavier shipments require specialised handling, larger vehicles, and stricter customs checks â€” all increasing the chance of a delay.",
    "SLA":                   "Service Level Agreement â€” the contracted delivery window in days. Tighter SLAs leave zero buffer if anything goes wrong en route.",
    "pickup_metro":          "Whether the origin is a major urban logistics hub. Metro hubs have better infrastructure, 24/7 operations, and faster handoffs â€” reducing delay risk.",
    "pickup_non_metro":      "Whether the origin is a non-metropolitan location. Rural or remote origins have limited carrier options and slower processing â€” increasing delay risk.",
    "drop_metro":            "Whether the destination is a major urban hub. Metro destinations have better receiving infrastructure and faster customs clearance.",
    "drop_non_metro":        "Whether the destination is non-metropolitan. Last-mile delivery to remote areas is the most common cause of delay.",
    "cp_delay_per_quarter":  "Carrier's delay rate over the past 3 months (0 = never late, 1 = always late). A carrier that delayed 30%+ of shipments this quarter is a serious risk signal.",
    "cp_ontime_per_quarter": "Carrier's on-time delivery rate this quarter. High on-time rate means a reliable partner â€” this feature reduces predicted delay risk.",
    "cp_delay_per_month":    "Carrier's delay rate over the past 30 days. Recent performance is more predictive than quarterly â€” sudden spikes here indicate operational issues.",
    "cp_ontime_per_month":   "Carrier's on-time rate this month. If this has dropped compared to the quarterly rate, the carrier may be experiencing current capacity problems.",
    "holiday_in_between":    "Whether a public holiday falls within the shipment window. Holidays cause warehouse closures, skeleton crews, and customs backlogs â€” adding 1-3 days.",
    "is_sunday_in_between":  "Whether a Sunday falls within the shipment window. Sunday logistics operations run at reduced capacity in most countries â€” a known delay multiplier.",
}

# ---------------------------------------------------------------------------
# Build feature matrix for a set of nodes
# ---------------------------------------------------------------------------

def _node_features(
    artifact: Dict[str, Any],
    G,
    nodes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an (N Ã— 13) float32 feature matrix for the supplied node list
    using the same defaults as delay_model.predict_delay_proba.

    Returns (X, valid_nodes) â€” rows in X map 1-to-1 to valid_nodes.
    """
    rows: List[np.ndarray] = []
    valid: List[str] = []

    for node in nodes:
        if node not in G.nodes:
            continue

        in_edges = list(G.in_edges(node, data=True))
        if in_edges:
            dists   = [d.get("distance_m",     500_000) for _, _, d in in_edges]
            weights = [d.get("avg_weight_kg",   200)    for _, _, d in in_edges]
            avg_dist   = float(np.mean(dists))
            avg_weight = float(np.mean(weights)) * 1000   # kg â†’ g
        else:
            tier = G.nodes[node].get("tier", 3)
            avg_dist   = 500_000 - tier * 50_000
            avg_weight = 200_000

        row = np.array([
            avg_dist,
            avg_weight,
            1,      # SLA default
            1,      # pickup_metro
            0,
            1,      # drop_metro
            0,
            0.15,   # cp_delay_per_quarter
            0.85,
            0.15,   # cp_delay_per_month
            0.85,
            0,      # holiday_in_between
            0,      # is_sunday_in_between
        ], dtype=np.float32)

        rows.append(row)
        valid.append(node)

    if not rows:
        return np.empty((0, 13), dtype=np.float32), []

    return np.vstack(rows), valid


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def _force_xgb_cpu_predictor(model: Any) -> None:
    """
    Keep SHAP inference on CPU to avoid XGBoost CUDA/CPU mismatch warnings.
    """
    try:
        module_name = model.__class__.__module__.lower()
    except Exception:
        module_name = ""
    if "xgboost" not in module_name:
        return
    try:
        model.set_params(device="cpu")
    except Exception:
        pass
    try:
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
    except Exception:
        pass

def compute_shap(
    artifact:  Dict[str, Any],
    risk_df:   pd.DataFrame,
    G,
    top_n:     int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Compute SHAP values for the top-N highest-risk nodes.

    Parameters
    ----------
    artifact  : dict from delay_model.load_or_train()
    risk_df   : DataFrame from risk_scoring.score_nodes()
    G         : NetworkX DiGraph
    top_n     : max nodes to explain (keep small for speed)

    Returns
    -------
    dict  {node_id: {feature_name: shap_value_float}}
    Empty dict if SHAP is unavailable or no nodes to explain.
    """
    try:
        import shap
    except ImportError:
        print("  [shap_explain] shap not installed â€” skipping explainability")
        return {}

    if risk_df.empty:
        return {}

    model    = artifact["model"]
    _force_xgb_cpu_predictor(model)
    features = artifact["features"]   # ordered list of 13 feature names

    # Select top-N nodes by risk score
    top_nodes = risk_df.nlargest(top_n, "risk_score")["node"].tolist()

    X, valid_nodes = _node_features(artifact, G, top_nodes)
    if X.shape[0] == 0:
        return {}

    # Use TreeExplainer (works for XGBoost + RandomForest)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # RandomForest returns list[array] (one per class) â€” take class-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    except Exception as e:
        print(f"  [shap_explain] TreeExplainer failed: {e} â€” using KernelExplainer")
        try:
            bg = shap.kmeans(X, min(10, X.shape[0]))
            explainer   = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X, nsamples=50)[:, :, 1]
        except Exception as e2:
            print(f"  [shap_explain] KernelExplainer also failed: {e2}")
            return {}

    # Map back to dict
    result: Dict[str, Dict[str, float]] = {}
    for i, node in enumerate(valid_nodes):
        result[node] = {
            feat: float(shap_values[i, j])
            for j, feat in enumerate(features)
        }

    print(f"  [shap_explain] Computed SHAP for {len(result)} nodes")
    return result


# ---------------------------------------------------------------------------
# Plotly: Global feature importance bar chart
# ---------------------------------------------------------------------------

def shap_bar_figure(
    shap_results: Dict[str, Dict[str, float]],
) -> go.Figure:
    """
    Horizontal bar chart of mean |SHAP| across all explained nodes.
    Styled to match the SupplAI dark theme.
    """
    if not shap_results:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            annotations=[dict(
                text="No SHAP data available", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(color="#64748b", size=14),
            )],
        )
        return fig

    # Aggregate mean |SHAP| per feature
    all_features = list(next(iter(shap_results.values())).keys())
    mean_abs: Dict[str, float] = {}
    for feat in all_features:
        vals = [abs(shap_results[node].get(feat, 0.0)) for node in shap_results]
        mean_abs[feat] = float(np.mean(vals))

    # Sort by importance
    sorted_items = sorted(mean_abs.items(), key=lambda x: x[1])
    feats  = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour gradient: low importance = slate, high = purple-red
    max_v  = max(values) if values else 1.0
    colours = [
        f"rgba({int(99 + 150*(v/max_v))}, {int(102 - 60*(v/max_v))}, {int(241 - 150*(v/max_v))}, 0.85)"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=feats,
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(
            title="Mean |SHAP value| â€” impact on delay probability",
            color="#94a3b8", gridcolor="#1e293b",
            title_font=dict(size=12),
        ),
        yaxis=dict(color="#e2e8f0", tickfont=dict(size=11)),
        font=dict(color="#e2e8f0", family="Inter"),
        height=420,
        margin=dict(t=30, b=40, l=10, r=80),
        title=dict(
            text="Global Feature Importance (mean |SHAP| across all risk nodes)",
            font=dict(color="#e2e8f0", size=14, family="Inter"),
            x=0,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Per-node waterfall chart
# ---------------------------------------------------------------------------

def shap_waterfall_figure(
    node_shap:  Dict[str, float],
    node_name:  str,
    base_value: float = 0.5,
) -> go.Figure:
    """
    Waterfall chart showing how each feature pushes the delay probability
    up (red) or down (green) from the baseline for a specific node.
    """
    if not node_shap:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a")
        return fig

    # Sort by absolute SHAP value (most impactful first)
    sorted_items = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)

    labels = ["Base"] + [FEATURE_LABELS.get(f, f) for f, _ in sorted_items] + ["Final Score"]
    values = [base_value] + [v for _, v in sorted_items]

    # Build cumulative for waterfall
    running = base_value
    measures = ["absolute"]
    texts    = [f"{base_value:.3f}"]
    for _, v in sorted_items:
        running += v
        measures.append("relative")
        texts.append(f"{v:+.4f}")

    measures.append("total")
    texts.append(f"{running:.3f}")

    # Colour: positive SHAP = risk-increasing (red), negative = risk-reducing (green)
    marker_colours = ["#6366f1"]   # base purple
    for _, v in sorted_items:
        marker_colours.append("#ef4444" if v > 0 else "#22c55e")
    marker_colours.append("#f97316")  # final total = orange

    shap_vals_for_fig = [base_value] + [v for _, v in sorted_items]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=labels,
        y=shap_vals_for_fig + [None],  # last point is "total"
        text=texts,
        textposition="outside",
        connector=dict(line=dict(color="#334155", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#f97316")),
        textfont=dict(color="#e2e8f0", size=11),
    ))

    fig.update_layout(
        title=dict(
            text=f"Delay Risk Breakdown â€” What is driving delay probability for <b>{node_name}</b>",
            font=dict(color="#e2e8f0", size=14, family="Inter"),
            x=0,
        ),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(
            color="#94a3b8", tickangle=-35,
            tickfont=dict(size=10),
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title="Delay Probability Contribution",
            color="#94a3b8", gridcolor="#1e293b",
            title_font=dict(size=11),
        ),
        font=dict(color="#e2e8f0", family="Inter"),
        height=440,
        margin=dict(t=50, b=100, l=80, r=30),
        showlegend=False,
        annotations=[
            dict(
                x=0.01, y=1.04, xref="paper", yref="paper",
                text="<b style='color:#ef4444;'>Red</b> = increases delay risk &nbsp;|&nbsp; <b style='color:#22c55e;'>Green</b> = reduces delay risk",
                showarrow=False,
                font=dict(size=11, color="#94a3b8"),
                align="left",
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Plain-English SHAP summary (for Gemini prompt injection)
# ---------------------------------------------------------------------------

def shap_to_text(
    node_shap: Dict[str, float],
    node_name: str,
    top_k: int = 5,
) -> str:
    """
    Produce a clear, business-friendly explanation of why this node's
    shipments are predicted to be delayed or reliable.
    Injected into the Gemini prompt and shown in the dashboard.
    """
    if not node_shap:
        return "No SHAP data available."

    sorted_items = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    risk_drivers    = [(f, v) for f, v in sorted_items if v > 0]
    protect_drivers = [(f, v) for f, v in sorted_items if v <= 0]

    lines = []

    if risk_drivers:
        lines.append("WHAT IS INCREASING DELAY RISK:")
        for feat, val in risk_drivers:
            label = FEATURE_LABELS.get(feat, feat)
            desc  = FEATURE_DESCRIPTIONS.get(feat, "")
            # Extract the first sentence only
            short = desc.split(".")[0]
            lines.append(f"  + {label} (impact: +{val:.3f})")
            lines.append(f"    {short}.")

    if protect_drivers:
        lines.append("")
        lines.append("WHAT IS PROTECTING AGAINST DELAY:")
        for feat, val in protect_drivers:
            label = FEATURE_LABELS.get(feat, feat)
            desc  = FEATURE_DESCRIPTIONS.get(feat, "")
            short = desc.split(".")[0]
            lines.append(f"  - {label} (impact: {val:.3f})")
            lines.append(f"    {short}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from delay_model import load_or_train
    from graph_builder import build_graph

    print("Loading model â€¦")
    artifact = load_or_train()

    print("Building graph â€¦")
    G = build_graph()

    # Mock a tiny risk_df
    nodes = list(G.nodes())[:10]
    risk_df = pd.DataFrame({"node": nodes, "risk_score": [0.9 - i*0.05 for i in range(len(nodes))]})

    print("Computing SHAP â€¦")
    results = compute_shap(artifact, risk_df, G, top_n=5)

    if results:
        top_node = next(iter(results))
        print(f"\nTop node: {top_node}")
        print(shap_to_text(results[top_node], top_node))
    else:
        print("No SHAP results (shap may not be installed).")

