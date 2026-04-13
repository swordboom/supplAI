"""
material_flow.py
----------------
Maps each supply chain node's product_category to the specific materials
it produces and ships to downstream nodes.

Key idea: an edge  A → B  carries the material that node A *produces*.
So the material on an edge = source node's product_category mapped to
real-world commodities.
"""

from __future__ import annotations
import networkx as nx
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core mapping: product_category → list of materials the city produces/ships
# ---------------------------------------------------------------------------
CATEGORY_MATERIALS: dict[str, list[str]] = {
    "Electronics":       ["Circuit Boards", "Consumer Electronics", "Electronic Components"],
    "Semiconductors":    ["Microchips", "Memory Chips", "Processors", "Silicon Wafers"],
    "Automotive":        ["Engine Components", "Vehicle Parts", "Transmissions", "EV Batteries"],
    "Textiles":          ["Fabric", "Yarn", "Apparel", "Synthetic Fibres"],
    "Steel":             ["Steel Coils", "Metal Sheets", "Steel Beams", "Alloy Steel"],
    "Aerospace":         ["Aircraft Components", "Avionics", "Turbine Parts", "Composite Materials"],
    "Chemicals":         ["Industrial Chemicals", "Solvents", "Polymers", "Adhesives"],
    "Pharmaceuticals":   ["Active Ingredients", "Medicines", "Medical Supplies", "Vaccines"],
    "E-Commerce":        ["Consumer Goods", "Packaged Retail Products", "Small Parcels"],
    "IT Hardware":       ["Servers", "Networking Equipment", "Computer Hardware", "Storage Drives"],
    "Manufacturing":     ["Industrial Equipment", "Machinery Parts", "Precision Tools"],
    "Distribution":      ["Mixed Cargo", "Retail Packages", "Consumer Goods"],
    "Machinery":         ["Heavy Machinery", "Industrial Tools", "CNC Equipment"],
    "Oil Gas":           ["Crude Oil", "Natural Gas", "Petroleum Products", "LNG"],
    "Logistics":         ["Mixed Freight", "Containerised Cargo", "Bulk Goods"],
    "Port Logistics":    ["Shipping Containers", "Port Cargo", "Bulk Commodities"],
    "Port Distribution": ["Containerised Goods", "Trans-shipment Cargo"],
    "Luxury Goods":      ["Luxury Items", "High-Value Goods", "Branded Products"],
    "Raw Materials":     ["Iron Ore", "Coal", "Copper", "Rare Earth Minerals"],
    "Minerals":          ["Lithium", "Cobalt", "Iron Ore", "Copper Ore"],
    "Agriculture":       ["Grains", "Produce", "Agricultural Commodities", "Fertilisers"],
    "Oil Minerals":      ["Crude Oil", "Mineral Resources", "Petroleum"],
    "Petrochemicals":    ["Plastics", "Synthetic Rubber", "Petroleum Derivatives", "Resins"],
}

# Single representative label per category (used in tight spaces like edge labels)
CATEGORY_SHORT_LABEL: dict[str, str] = {
    "Electronics":       "Electronic Components",
    "Semiconductors":    "Microchips & Wafers",
    "Automotive":        "Vehicle Parts",
    "Textiles":          "Fabric & Apparel",
    "Steel":             "Steel Products",
    "Aerospace":         "Aerospace Components",
    "Chemicals":         "Industrial Chemicals",
    "Pharmaceuticals":   "Pharmaceuticals",
    "E-Commerce":        "Consumer Goods",
    "IT Hardware":       "IT Hardware",
    "Manufacturing":     "Machinery Parts",
    "Distribution":      "Mixed Cargo",
    "Machinery":         "Heavy Machinery",
    "Oil Gas":           "Crude Oil & Gas",
    "Logistics":         "Freight",
    "Port Logistics":    "Port Cargo",
    "Port Distribution": "Containerised Cargo",
    "Luxury Goods":      "Luxury Goods",
    "Raw Materials":     "Raw Materials",
    "Minerals":          "Minerals & Ores",
    "Agriculture":       "Agricultural Goods",
    "Oil Minerals":      "Oil & Minerals",
    "Petrochemicals":    "Petrochemicals",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_edge_material(G: nx.DiGraph, src: str, dst: str) -> str:
    """
    Return a short material label for the edge src → dst.
    Uses the source node's product_category.
    Falls back to the raw material ID stored on the edge if available.
    """
    cat = G.nodes[src].get("product_category", "")
    if cat in CATEGORY_SHORT_LABEL:
        return CATEGORY_SHORT_LABEL[cat]
    # Fallback: use the material field already on the edge
    raw = G.edges[src, dst].get("material", "")
    if raw and raw != "None":
        return raw
    return "General Cargo"


def get_edge_material_detail(G: nx.DiGraph, src: str, dst: str) -> list[str]:
    """Return the full list of materials produced by the source node."""
    cat = G.nodes[src].get("product_category", "")
    return CATEGORY_MATERIALS.get(cat, ["General Cargo"])


def get_disrupted_materials(
    G: nx.DiGraph,
    cascade_result: dict,
) -> pd.DataFrame:
    """
    For every disrupted edge (both endpoints in the cascade), collect:
      - source city, destination city, material label, full material list,
        source product_category, destination product_category

    Returns a DataFrame sorted by material label.
    """
    cascade_nodes = set(cascade_result.keys())
    rows = []

    for src, dst in G.edges():
        if src in cascade_nodes and dst in cascade_nodes:
            src_nd = G.nodes[src]
            dst_nd = G.nodes[dst]
            rows.append({
                "from_node":      src,
                "to_node":        dst,
                "from_city":      src_nd.get("city_name", src),
                "to_city":        dst_nd.get("city_name", dst),
                "from_country":   src_nd.get("country", "?"),
                "to_country":     dst_nd.get("country", "?"),
                "material":       get_edge_material(G, src, dst),
                "material_detail":get_edge_material_detail(G, src, dst),
                "from_category":  src_nd.get("product_category", "?"),
                "to_category":    dst_nd.get("product_category", "?"),
                "from_tier":      src_nd.get("tier", "?"),
                "to_tier":        dst_nd.get("tier", "?"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("material")
    return df


def summarise_materials_at_risk(
    G: nx.DiGraph,
    cascade_result: dict,
) -> pd.DataFrame:
    """
    Aggregate disrupted materials into a summary:
      material | routes_disrupted | countries_affected | sample_route
    """
    df = get_disrupted_materials(G, cascade_result)
    if df.empty:
        return pd.DataFrame()

    summary_rows = []
    for mat, grp in df.groupby("material"):
        countries = set(grp["from_country"]) | set(grp["to_country"])
        sample = f"{grp.iloc[0]['from_city']} → {grp.iloc[0]['to_city']}"
        # Collect all specific sub-materials for this category
        all_items: list[str] = []
        for items in grp["material_detail"]:
            all_items.extend(items)
        sub_mats = ", ".join(sorted(set(all_items)))
        summary_rows.append({
            "Material Flow":        mat,
            "Specific Items":       sub_mats,
            "Disrupted Routes":     len(grp),
            "Countries Affected":   len(countries),
            "Example Route":        sample,
        })

    return (
        pd.DataFrame(summary_rows)
        .sort_values("Disrupted Routes", ascending=False)
        .reset_index(drop=True)
    )


def get_node_material_label(G: nx.DiGraph, node: str) -> str:
    """Return what a node produces, for use in hover text."""
    cat = G.nodes[node].get("product_category", "")
    return CATEGORY_SHORT_LABEL.get(cat, "General Cargo")
