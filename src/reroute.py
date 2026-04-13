"""
reroute.py
-----------
Suggests alternate supply routes to avoid disrupted nodes.

Strategy
--------
1. Identify the most critical Source→Destination pairs currently routed
   through disrupted nodes.
2. For each such pair, find the shortest path that avoids ALL disrupted nodes,
   using Dijkstra's algorithm on the distance-weighted graph.
3. Compare the alternate path against the original in terms of:
     - route length (hops)
     - total distance (km)
     - estimated added delay

Returns a list of recommendation dicts ready for the dashboard.
"""

import random
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Any


def validate_upstream_exposure(
    G:             nx.DiGraph,
    node:          str,
    cascade_nodes: set,
    hops:          int = 3,
) -> Dict[str, Any]:
    """
    Check whether a node's upstream supply chain is secretly exposed to the
    disrupted region.

    Walks backward (against edge direction) up to `hops` levels from `node`
    and collects every ancestor.  Returns:
        upstream_total   : int   — total ancestors found
        upstream_exposed : int   — how many are in the cascade
        exposure_ratio   : float — exposed / total  (0 if no upstream found)
        exposed_nodes    : list  — IDs of overlapping nodes
        validation_status: str   — "✅ Clean" | "⚠️ Partial Exposure" | "⛔ Hidden Dependency"
    """
    visited: set = set()
    frontier = {node}

    for _ in range(hops):
        next_frontier: set = set()
        for n in frontier:
            for pred in G.predecessors(n):
                if pred not in visited and pred != node:
                    next_frontier.add(pred)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    upstream_total   = len(visited)
    exposed_nodes    = [n for n in visited if n in cascade_nodes]
    upstream_exposed = len(exposed_nodes)

    if upstream_total == 0:
        ratio = 0.0
    else:
        ratio = upstream_exposed / upstream_total

    if ratio >= 0.50:
        vstatus = "⛔ Hidden Dependency"
    elif ratio >= 0.20:
        vstatus = "⚠️ Partial Exposure"
    else:
        vstatus = "✅ Clean"

    return {
        "upstream_total":    upstream_total,
        "upstream_exposed":  upstream_exposed,
        "exposure_ratio":    round(ratio, 3),
        "exposed_nodes":     exposed_nodes,
        "validation_status": vstatus,
    }


def validate_alternate_route(
    G:             nx.DiGraph,
    alt_path:      List[str],
    cascade_nodes: set,
    supply_df:     Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Validate every intermediate node in an alternate route for hidden upstream
    exposure to the disrupted region.

    Returns a dict with:
        route_validation     : "✅ Clean" | "⚠️ Partial Exposure" | "⛔ Hidden Dependency"
        worst_exposure_ratio : float
        exposed_intermediates: list of dicts  {node, city_name, exposure_ratio, exposed_upstream}
        validation_note      : plain-English explanation
    """
    if not alt_path or len(alt_path) < 3:
        return {
            "route_validation":      "✅ Clean",
            "worst_exposure_ratio":  0.0,
            "exposed_intermediates": [],
            "validation_note":       "Direct connection — no intermediate nodes to validate.",
        }

    intermediates = alt_path[1:-1]  # exclude source & destination
    worst_ratio   = 0.0
    exposed_intermediates: List[Dict] = []

    for node in intermediates:
        result = validate_upstream_exposure(G, node, cascade_nodes, hops=3)
        if result["upstream_exposed"] > 0:
            def _city(n):
                if supply_df is not None and n in supply_df.index:
                    row = supply_df.loc[n]
                    return f"{row['city_name']}, {row['country']}"
                return n

            exposed_intermediates.append({
                "node":            node,
                "city_name":       _city(node),
                "exposure_ratio":  result["exposure_ratio"],
                "exposed_upstream": [_city(n) for n in result["exposed_nodes"][:5]],
                "validation_status": result["validation_status"],
            })
        worst_ratio = max(worst_ratio, result["exposure_ratio"])

    if worst_ratio >= 0.50:
        route_validation = "⛔ Hidden Dependency"
        note = (
            "One or more intermediate nodes source >50% of their inputs from the "
            "disrupted region. This route may fail under the same event."
        )
    elif worst_ratio >= 0.20:
        route_validation = "⚠️ Partial Exposure"
        note = (
            "Some intermediate nodes have partial upstream exposure to the disrupted "
            "region. Monitor closely and prepare a contingency."
        )
    else:
        route_validation = "✅ Clean"
        note = "No significant upstream exposure detected. Route is genuinely independent."

    return {
        "route_validation":      route_validation,
        "worst_exposure_ratio":  round(worst_ratio, 3),
        "exposed_intermediates": exposed_intermediates,
        "validation_note":       note,
    }


def find_alternates(
    G:               nx.DiGraph,
    affected_nodes:  List[str],
    cascade_result:  Dict[str, int],
    supply_df:       Optional[pd.DataFrame] = None,
    top_pairs:       int = 8,
) -> List[Dict[str, Any]]:
    """
    Find alternate routes avoiding all disrupted nodes.

    Parameters
    ----------
    G               : nx.DiGraph   — full supply chain graph
    affected_nodes  : list[str]    — seed nodes from disruption_input
    cascade_result  : dict         — full cascade (including downstream)
    supply_df       : pd.DataFrame — city metadata (optional, for display)
    top_pairs       : int          — max number of rerouting suggestions

    Returns
    -------
    list of dicts, each containing:
        source          : str   — origin city ID
        destination     : str   — destination city ID
        source_name     : str   — human-readable origin
        destination_name: str   — human-readable destination
        disrupted_path  : list  — original path (passes through disrupted nodes)
        alternate_path  : list  — new safe path
        original_dist_km: float
        alternate_dist_km: float
        distance_delta_km: float  — extra km added by rerouting
        detour_pct      : float  — % increase in distance
        status          : str   — "✅ Alternate Found" / "⚠️ No Alternate"
        hops_original   : int
        hops_alternate  : int
    """
    # Only block the SEED nodes (depth=0) as fully unavailable.
    # Cascade nodes (depth>=1) are "at risk" but can still be used for routing.
    seed_nodes    = {n for n, d in cascade_result.items() if d == 0}
    all_disrupted = seed_nodes   # for rerouting purposes

    # Build a "safe" subgraph that excludes directly disrupted seed nodes
    # AND any node flagged on the OFAC SDN sanctions list
    ofac_blocked = {n for n, attr in G.nodes(data=True) if attr.get("ofac_sanctioned", False)}
    banned_nations = seed_nodes.union(ofac_blocked)
    
    safe_nodes = [n for n in G.nodes if n not in banned_nations]
    G_safe     = G.subgraph(safe_nodes).copy()

    results: List[Dict[str, Any]] = []

    # Build (source, disrupted_node, destination) triples from DIRECT edges.
    # Using triples (not pairs) means the "original path" is always the concrete
    # direct route that uses the disrupted node — Dijkstra's detour cannot
    # accidentally bypass it and cause the pair to be dropped.
    candidate_triples: List[tuple] = []
    seen_pairs: set = set()

    for disrupted_node in seed_nodes:
        for pred in G.predecessors(disrupted_node):
            if pred not in seed_nodes:
                for succ in G.successors(disrupted_node):
                    if succ not in seed_nodes:
                        pair = (pred, succ)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidate_triples.append((pred, disrupted_node, succ))

    # Shuffle for variety then cap
    random.shuffle(candidate_triples)
    candidate_triples = candidate_triples[: top_pairs * 4]

    # Evaluate each triple
    for source, via_disrupted, destination in candidate_triples:

        if source not in G or destination not in G:
            continue

        # ---- Original path = the direct route through the disrupted node ----
        def _edge_km(u, v, graph=G):
            return graph[u][v].get("distance_m", 1000) / 1000.0
            
        def _edge_weight(u, v, graph=G):
            return graph[u][v].get("weight", _edge_km(u, v, graph))
            
        def _edge_tariff(u, v, graph=G):
            return graph[u][v].get("tariff_rate", 0.0)

        def _calc_usd_cost(path, is_alternate=False, graph=G):
            if not path or len(path) < 2: return 0.0
            cost = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                dist_km = _edge_km(u, v, graph)
                tariff_pct = _edge_tariff(u, v, graph)
                
                # Realism Fix: Original routes reflect cheap long-term bulk contracts.
                # Alternate routes reflect expensive emergency spot-market logistics.
                base_rate = 2.50 if is_alternate else 0.40
                
                edge_usd = (dist_km * base_rate) + 500.0 + (50000.0 * (tariff_pct / 100.0))
                cost += edge_usd
                
            # Expedited emergency freight penalty
            if is_alternate:
                cost += 25000.0
            return cost

        try:
            orig_dist = _edge_km(source, via_disrupted) + _edge_km(via_disrupted, destination)
            orig_path = [source, via_disrupted, destination]
            orig_cost_usd = _calc_usd_cost(orig_path, is_alternate=False, graph=G)
        except (KeyError, TypeError):
            orig_path = [source, destination]
            orig_dist = float("inf")
            orig_cost_usd = float("inf")

        # ---- Alternate path (avoid all disrupted nodes) ----
        alt_status   = "✅ Alternate Found"
        alt_path     = []
        alt_dist     = float("inf")
        alt_cost_usd = float("inf")
        dist_delta   = 0.0
        detour_pct   = 0.0
        cost_delta_usd = 0.0
        alt_max_tariff = 0.0

        if source in G_safe and destination in G_safe:
            try:
                alt_path = nx.shortest_path(G_safe, source, destination, weight="weight")
                alt_dist = sum(
                    _edge_km(alt_path[i], alt_path[i+1], G_safe)
                    for i in range(len(alt_path) - 1)
                )
                alt_max_tariff = max([0.0] + [
                    _edge_tariff(alt_path[i], alt_path[i+1], G_safe)
                    for i in range(len(alt_path) - 1)
                ])
                alt_cost_usd = _calc_usd_cost(alt_path, is_alternate=True, graph=G_safe)
                
                if orig_dist < float("inf"):
                    dist_delta = alt_dist - orig_dist
                    detour_pct = (dist_delta / orig_dist) * 100 if orig_dist > 0 else 0.0
                    cost_delta_usd = alt_cost_usd - orig_cost_usd
            except nx.NetworkXNoPath:
                # Usually no path implies the only bridges across the gap are sanctioned or disrupted
                alt_status = "⚠️ Route Severed (OFAC / Disruption Block)"
        else:
            is_source_ofac = G.nodes[source].get("ofac_sanctioned")
            is_dest_ofac = G.nodes[destination].get("ofac_sanctioned")
            if is_source_ofac or is_dest_ofac:
                alt_status = "⛔ OFAC SANCTIONED ENDPOINT"
            else:
                alt_status = "⚠️ Disrupted Endpoint"

        # ---- Human-readable names ----
        def _name(node_id):
            if supply_df is not None and node_id in supply_df.index:
                row = supply_df.loc[node_id]
                return f"{row['city_name']}, {row['country']}"
            return node_id

        # ---- Upstream validation (Feature 3) ----
        cascade_nodes = set(cascade_result.keys())
        if alt_path and alt_status == "✅ Alternate Found":
            validation = validate_alternate_route(G, alt_path, cascade_nodes, supply_df)
        else:
            validation = {
                "route_validation":      "N/A",
                "worst_exposure_ratio":  0.0,
                "exposed_intermediates": [],
                "validation_note":       "No alternate route available.",
            }

        results.append({
            "source":              source,
            "destination":         destination,
            "source_name":         _name(source),
            "destination_name":    _name(destination),
            "disrupted_path":      orig_path,
            "alternate_path":      alt_path if alt_path else [],
            "original_dist_km":    round(orig_dist, 1) if orig_dist < float("inf") else None,
            "alternate_dist_km":   round(alt_dist,  1) if alt_dist  < float("inf") else None,
            "distance_delta_km":   round(dist_delta, 1),
            "detour_pct":          round(detour_pct, 1),
            "orig_cost_usd":       round(orig_cost_usd, 2) if orig_cost_usd < float("inf") else None,
            "alt_cost_usd":        round(alt_cost_usd, 2)  if alt_cost_usd  < float("inf") else None,
            "cost_delta_usd":      round(cost_delta_usd, 2),
            "status":              alt_status,
            "hops_original":       len(orig_path) - 1,
            "hops_alternate":      len(alt_path)  - 1 if alt_path else 0,
            "max_tariff_pct":      round(alt_max_tariff, 1),
            # Validation fields
            "route_validation":      validation["route_validation"],
            "worst_exposure_ratio":  validation["worst_exposure_ratio"],
            "exposed_intermediates": validation["exposed_intermediates"],
            "validation_note":       validation["validation_note"],
        })

    # De-duplicate and sort: found alternates first, then by detour %
    results = _deduplicate(results)
    results.sort(key=lambda r: (r["status"] != "✅ Alternate Found", r["detour_pct"]))

    return results[:top_pairs]


def _deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove duplicate (source, destination) pairs, keeping the best result."""
    seen   = {}
    for r in results:
        key = (r["source"], r["destination"])
        if key not in seen or r["status"] == "✅ Alternate Found":
            seen[key] = r
    return list(seen.values())


def format_path(path: List[str], supply_df: Optional[pd.DataFrame] = None) -> str:
    """Convert a list of city IDs into a human-readable route string."""
    if not path:
        return "No path"

    def _label(node_id: str) -> str:
        if supply_df is not None and node_id in supply_df.index:
            row = supply_df.loc[node_id]
            return f"{row['city_name']} ({row['country']})"
        return node_id

    return " → ".join(_label(n) for n in path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph, load_supply_metadata
    from cascade_model import run_cascade

    G        = build_graph()
    supply   = load_supply_metadata()
    cascade  = run_cascade(G, ["City_1", "City_2"], max_depth=3)
    routes   = find_alternates(G, ["City_1", "City_2"], cascade, supply_df=supply)

    print(f"Found {len(routes)} rerouting suggestions:\n")
    for r in routes:
        print(f"  {r['source_name']} → {r['destination_name']}")
        print(f"    Status: {r['status']}")
        if r["alternate_path"]:
            print(f"    Alt path: {format_path(r['alternate_path'], supply)}")
        print(f"    Extra km: +{r['distance_delta_km']} ({r['detour_pct']}%)\n")
