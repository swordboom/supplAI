"""
risk_scoring.py
----------------
Calculates a composite risk score for every affected node in the
supply chain disruption cascade.

Risk Score Formula
------------------
  risk = 0.40 × depth_score       (how far from the disruption source)
       + 0.40 × centrality_score  (how critical the node is to the network)
       + 0.20 × delay_probability (historical delay risk for that route)

All three components are normalised to [0, 1] before combining.

Returns a pandas DataFrame sorted by risk_score (highest first).
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def compute_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute betweenness centrality for all nodes.

    Betweenness centrality measures how often a node sits on the
    shortest path between other nodes — high centrality = bottleneck.

    Uses both in-flow and out-flow (undirected betweenness on DiGraph).

    Returns dict {node: centrality_score (0-1)}
    """
    print("  [risk_scoring] Computing betweenness centrality …")
    # normalised=True already gives values in [0,1]
    centrality = nx.betweenness_centrality(G, normalized=True, weight="weight")
    return centrality


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]. Handle constant arrays gracefully."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def score_nodes(
    G:              nx.DiGraph,
    cascade_result: Dict[str, int],
    centrality:     Dict[str, float],
    delay_artifact: Optional[Dict[str, Any]] = None,
    weights:        Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute the composite risk score for each affected node.

    Parameters
    ----------
    G               : nx.DiGraph — full supply chain graph
    cascade_result  : dict {node: cascade_depth}
    centrality      : dict {node: betweenness_centrality}
    delay_artifact  : dict from delay_model.load_or_train() (optional)
    weights         : override default weights {depth, centrality, delay}

    Returns
    -------
    pd.DataFrame sorted by risk_score descending with columns:
        node, city_name, country, product_category, tier,
        cascade_depth, centrality, delay_prob, risk_score, status
    """
    if weights is None:
        weights = {"depth": 0.40, "centrality": 0.40, "delay": 0.20}

    if not cascade_result:
        return pd.DataFrame()

    rows = []
    for node, depth in cascade_result.items():
        node_data = G.nodes.get(node, {})

        rows.append({
            "node":             node,
            "city_name":        node_data.get("city_name",        node),
            "country":          node_data.get("country",          "Unknown"),
            "region":           node_data.get("region",           "Unknown"),
            "product_category": node_data.get("product_category", "General"),
            "tier":             node_data.get("tier",             3),
            "risk_factor":      node_data.get("risk_factor",      "medium"),
            "lat":              node_data.get("lat",              0.0),
            "lon":              node_data.get("lon",              0.0),
            "cascade_depth":    depth,
            "centrality_raw":   centrality.get(node, 0.0),
        })

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Component 1: Depth score (shallower depth = closer to origin = higher risk)
    # Invert: depth 0 (seed) → highest risk from proximity
    # -----------------------------------------------------------------------
    max_depth = df["cascade_depth"].max()
    if max_depth > 0:
        df["depth_score"] = 1.0 - (df["cascade_depth"] / max_depth)
    else:
        df["depth_score"] = 1.0

    # -----------------------------------------------------------------------
    # Component 2: Centrality score (already normalised)
    # -----------------------------------------------------------------------
    df["centrality_score"] = _normalise(df["centrality_raw"].values)

    # -----------------------------------------------------------------------
    # Component 3: Delay probability from the ML model
    # -----------------------------------------------------------------------
    if delay_artifact is not None:
        from delay_model import estimate_node_delay
        df["delay_prob"] = df["node"].apply(
            lambda n: estimate_node_delay(delay_artifact, G, n)
        )
    else:
        # Fallback: use risk_factor from supply_chain.csv as proxy
        risk_map = {"high": 0.75, "medium": 0.45, "low": 0.20}
        df["delay_prob"] = df["risk_factor"].map(risk_map).fillna(0.45)

    df["delay_score"] = _normalise(df["delay_prob"].values)

    # -----------------------------------------------------------------------
    # Composite risk score
    # -----------------------------------------------------------------------
    df["risk_score"] = (
          weights["depth"]       * df["depth_score"]
        + weights["centrality"]  * df["centrality_score"]
        + weights["delay"]       * df["delay_score"]
    )

    # -----------------------------------------------------------------------
    # Human-readable risk level
    # -----------------------------------------------------------------------
    def _level(score):
        if score >= 0.65:
            return "🔴 Critical"
        elif score >= 0.40:
            return "🟠 High"
        elif score >= 0.20:
            return "🟡 Medium"
        else:
            return "🟢 Low"

    df["risk_level"] = df["risk_score"].apply(_level)

    # -----------------------------------------------------------------------
    # Status label
    # -----------------------------------------------------------------------
    df["status"] = df["cascade_depth"].apply(
        lambda d: "🔴 Direct Impact" if d == 0 else f"⚡ Cascade (depth {d})"
    )

    # Round scores for display
    score_cols = ["depth_score", "centrality_score", "delay_prob", "risk_score"]
    df[score_cols] = df[score_cols].round(4)

    # Sort by risk score descending
    df.sort_values("risk_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph
    from cascade_model import run_cascade

    G = build_graph()
    centrality = compute_centrality(G)
    cascade = run_cascade(G, ["City_1", "City_2"], max_depth=3)
    df = score_nodes(G, cascade, centrality, delay_artifact=None)

    print(f"Risk table ({len(df)} nodes):")
    print(df[["node", "city_name", "country", "risk_score", "risk_level", "status"]].head(10).to_string(index=False))
