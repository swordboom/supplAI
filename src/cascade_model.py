"""
cascade_model.py
-----------------
Simulates how a supply chain disruption propagates through the network.

Algorithm: Breadth-First Search (BFS) from the set of initially affected
           ("seed") nodes, moving downstream (following directed edges)
           up to a configurable max_depth.

Returns a dict mapping each affected node to its cascade depth (how far
it is from the original disruption source).
"""

import networkx as nx
from collections import deque
from typing import Dict, List, Set


def run_cascade(
    G:           nx.DiGraph,
    seed_nodes:  List[str],
    max_depth:   int = 5,
) -> Dict[str, int]:
    """
    BFS cascade propagation from seed nodes.

    Parameters
    ----------
    G          : nx.DiGraph  — supply chain graph
    seed_nodes : list[str]   — initially disrupted nodes (from disruption_input)
    max_depth  : int         — max downstream hops to follow (default: 5)

    Returns
    -------
    dict[str, int]
        Keys   = all affected node IDs (including seeds)
        Values = cascade depth (seeds = 0, direct downstream = 1, ...)
    """
    # Filter seed nodes to only those that exist in the graph
    valid_seeds: List[str] = [n for n in seed_nodes if n in G.nodes]

    if not valid_seeds:
        return {}

    # BFS queue: (node, depth)
    queue: deque = deque()
    visited: Dict[str, int] = {}   # node → depth

    # Initialise seeds at depth 0
    for node in valid_seeds:
        queue.append((node, 0))
        visited[node] = 0

    while queue:
        current_node, depth = queue.popleft()

        # Stop propagating if we've hit the depth limit
        if depth >= max_depth:
            continue

        # Follow outgoing edges (downstream suppliers/buyers)
        for neighbor in G.successors(current_node):
            if neighbor not in visited:
                new_depth = depth + 1
                visited[neighbor] = new_depth
                queue.append((neighbor, new_depth))

    return visited


def get_cascade_stats(cascade_result: Dict[str, int]) -> dict:
    """
    Compute summary statistics from a cascade result.

    Returns dict with:
        total_affected : int   — number of nodes disrupted
        max_depth      : int   — furthest propagation depth reached
        by_depth       : dict  — {depth: count_of_nodes_at_that_depth}
        seed_count     : int   — number of original seed nodes
    """
    if not cascade_result:
        return {
            "total_affected": 0,
            "max_depth":      0,
            "by_depth":       {},
            "seed_count":     0,
        }

    max_depth = max(cascade_result.values())
    by_depth: Dict[int, int] = {}
    for depth in cascade_result.values():
        by_depth[depth] = by_depth.get(depth, 0) + 1

    return {
        "total_affected": len(cascade_result),
        "max_depth":      max_depth,
        "by_depth":       dict(sorted(by_depth.items())),
        "seed_count":     by_depth.get(0, 0),
    }


def get_cascade_subgraph(
    G:              nx.DiGraph,
    cascade_result: Dict[str, int],
) -> nx.DiGraph:
    """
    Return a subgraph containing only the affected nodes from the cascade.
    Useful for focused visualisation.
    """
    affected_nodes = list(cascade_result.keys())
    return G.subgraph(affected_nodes).copy()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny test graph
    G = nx.DiGraph()
    edges = [
        ("City_1", "City_24"), ("City_1", "City_35"),
        ("City_24", "City_53"), ("City_35", "City_47"),
        ("City_53", "City_63"), ("City_47", "City_17"),
    ]
    G.add_edges_from(edges)

    seed = ["City_1"]
    result = run_cascade(G, seed, max_depth=4)
    stats  = get_cascade_stats(result)

    print("Cascade result:", result)
    print("Stats:", stats)
