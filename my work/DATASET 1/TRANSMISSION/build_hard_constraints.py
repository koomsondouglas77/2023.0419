"""
build_hard_constraints.py
=============================================================================
Builds Hard Must-Links (ML) and Hard Cannot-Links (CL) for PCCC using the
34 European MapCodes and Synchronous Area topologies.

Author: Douglas Kojo Kumi Koomson (RRE Lab, ETH Zurich / Ashesi University)
=============================================================================
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

# -----------------------------------------------------------------------------
# 1. EUROPEAN SYNCHRONOUS AREAS (Derived from your 34 MapCodes)
# -----------------------------------------------------------------------------
SYNCHRONOUS_AREAS = {
    # Synchronous Area Continental Europe (CE)
    "CONTINENTAL_EUROPE": {
        "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "ES", "FR",
        "GR", "HR", "HU", "IT", "ME", "MK", "NL", "PL", "RO", "RS", "SI", "SK"
    },
    # Nordic Synchronous Area
    "NORDIC": {"FI", "NO", "SE"},
    # Great Britain Island System
    "GREAT_BRITAIN": {"GB"},
    # Baltic Synchronous Area
    "BALTIC": {"EE", "LT"},
    # IPS/UPS & Eastern Interconnections
    "EASTERN_INTERCONNECTION": {"AZ", "BY", "GE", "MD", "RU"}
}

def get_synchronous_zone(code: str) -> str:
    """Returns the synchronous area name for a given ISO MapCode."""
    if not isinstance(code, str):
        return "UNKNOWN"
    clean = code.strip().upper()
    for zone, members in SYNCHRONOUS_AREAS.items():
        if clean in members:
            return zone
    return "UNKNOWN"


def generate_hard_constraints(
    meta_csv="transmission_metadata_lookup.csv",
    metrics_csv="transmission_aggregated_metrics.csv",
    output_json="transmission_hard_constraints.json",
    max_ml_pairs=200,
    max_cl_pairs=200
):
    print("=" * 75)
    print("GENERATING HARD CONSTRAINTS (ML & CL) FOR PCCC")
    print("=" * 75)

    cwd = Path.cwd()

    # Locate metadata and source metrics
    meta_path = cwd / meta_csv if (cwd / meta_csv).exists() else list(cwd.rglob("*metadata*.csv"))[0]
    metrics_path = cwd / metrics_csv if (cwd / metrics_csv).exists() else list(cwd.rglob("*transmission_aggregated*.csv"))[0]

    df_meta = pd.read_csv(meta_path)
    df_metrics = pd.read_csv(metrics_path)

    print(f"[*] Loaded {len(df_meta):,} assets from: {meta_path.name}")
    print(f"[*] Loaded source metrics from: {metrics_path.name}")

    # Ensure index alignment
    df_meta["pccc_idx"] = df_meta["pccc_index"] if "pccc_index" in df_meta.columns else df_meta.index

    # Merge mapcode and corridor info from source metrics into metadata
    for col in ["mapcode", "mapcode1", "mapcode2", "in_area", "out_area", "production_type"]:
        matched = [c for c in df_metrics.columns if c.lower() == col.lower()]
        if matched and col not in df_meta.columns:
            df_meta[col] = df_metrics[matched[0]].values

    # Determine primary country / corridor representation
    if "mapcode" in df_meta.columns:
        df_meta["country"] = df_meta["mapcode"].astype(str).str.upper()
    elif "mapcode1" in df_meta.columns:
        df_meta["country"] = df_meta["mapcode1"].astype(str).str.upper()
    else:
        # Fallback to extract 2-letter country code from asset EIC string (chars 2:4 or prefix)
        id_col = "asset_id" if "asset_id" in df_meta.columns else df_meta.columns[0]
        df_meta["country"] = df_meta[id_col].astype(str).str[:2].str.upper()

    df_meta["sync_zone"] = df_meta["country"].apply(get_synchronous_zone)

    print("\nASSET DISTRIBUTION ACROSS SYNCHRONOUS ZONES:")
    zone_counts = df_meta["sync_zone"].value_counts()
    for z, count in zone_counts.items():
        print(f"  - {z:25s}: {count:5d} assets")

    ml_pairs = []
    cl_pairs = []

    # -------------------------------------------------------------------------
    # A. HARD MUST-LINKS (ML): Parallel Circuits in Same Border Corridor
    # -------------------------------------------------------------------------
    corridor_key = None
    if "mapcode1" in df_meta.columns and "mapcode2" in df_meta.columns:
        df_meta["corridor"] = df_meta.apply(
            lambda r: "-".join(sorted([str(r["mapcode1"]), str(r["mapcode2"])])), axis=1
        )
        corridor_key = "corridor"
    elif "in_area" in df_meta.columns and "out_area" in df_meta.columns:
        df_meta["corridor"] = df_meta.apply(
            lambda r: "-".join(sorted([str(r["in_area"]), str(r["out_area"])])), axis=1
        )
        corridor_key = "corridor"
    else:
        # Use first 10 characters of asset name/EIC as substation identifier
        id_col = "asset_id" if "asset_id" in df_meta.columns else df_meta.columns[0]
        df_meta["substation_stem"] = df_meta[id_col].astype(str).str[:10]
        corridor_key = "substation_stem"

    print(f"\n[*] Constructing Hard Must-Links using grouping key: '{corridor_key}'...")
    for _, grp in df_meta.groupby(corridor_key):
        indices = grp["pccc_idx"].tolist()
        if 1 < len(indices) <= 6:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ml_pairs.append([int(indices[i]), int(indices[j])])
                    if len(ml_pairs) >= max_ml_pairs:
                        break
            if len(ml_pairs) >= max_ml_pairs:
                break

    # -------------------------------------------------------------------------
    # B. HARD CANNOT-LINKS (CL): Asynchronous Grids + Opposite Risk Archetypes
    # -------------------------------------------------------------------------
    print("[*] Constructing Hard Cannot-Links across Asynchronous Synchronous Zones...")
    
    # 1. Asynchronous pairs (Nordic vs Continental Europe, or GB vs Continental)
    ce_indices = df_meta[df_meta["sync_zone"] == "CONTINENTAL_EUROPE"]["pccc_idx"].tolist()
    nordic_indices = df_meta[df_meta["sync_zone"] == "NORDIC"]["pccc_idx"].tolist()
    gb_indices = df_meta[df_meta["sync_zone"] == "GREAT_BRITAIN"]["pccc_idx"].tolist()

    async_other = nordic_indices if nordic_indices else gb_indices

    if ce_indices and async_other:
        for u in ce_indices[:30]:
            for v in async_other[:30]:
                cl_pairs.append([int(u), int(v)])
                if len(cl_pairs) >= (max_cl_pairs // 2):
                    break
            if len(cl_pairs) >= (max_cl_pairs // 2):
                break

    # 2. Opposite Operational Risk Classes (Class 0: Robust vs Class 3: Chronic Chokepoint)
    if "class_label" in df_meta.columns:
        print("[*] Adding Cannot-Links between Class 0 (Robust) and Class 3 (Chokepoints)...")
        c0 = df_meta[df_meta["class_label"] == 0]["pccc_idx"].tolist()
        c3 = df_meta[df_meta["class_label"] == 3]["pccc_idx"].tolist()
        for i in range(min(len(c0), 20)):
            for j in range(min(len(c3), 20)):
                cl_pairs.append([int(c0[i]), int(c3[j])])
                if len(cl_pairs) >= max_cl_pairs:
                    break
            if len(cl_pairs) >= max_cl_pairs:
                break

    # -------------------------------------------------------------------------
    # C. FEASIBILITY AUDIT (Baumann & Hochbaum Section 4.1)
    # Check that no CL edge exists between two nodes in the same ML super-node
    # -------------------------------------------------------------------------
    print("\n[*] Auditing constraint graph feasibility...")
    g_ml = nx.Graph()
    g_ml.add_nodes_from(range(len(df_meta)))
    g_ml.add_edges_from(ml_pairs)

    comp_map = {}
    for comp_id, comp in enumerate(nx.connected_components(g_ml)):
        for node in comp:
            comp_map[node] = comp_id

    feasible_cl = []
    dropped_cl = 0
    for u, v in cl_pairs:
        if comp_map.get(u, u) == comp_map.get(v, v):
            dropped_cl += 1
        else:
            feasible_cl.append([u, v])

    # -------------------------------------------------------------------------
    # D. EXPORT TO JSON FOR PCCC
    # -------------------------------------------------------------------------
    payload = {
        "ml": ml_pairs,
        "cl": feasible_cl,
        "sml": [],
        "scl": [],
        "sml_proba": [],
        "scl_proba": []
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 75)
    print("CONSTRAINTS GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"[*] Hard Must-Links (ML):   {len(ml_pairs):,} pairs")
    print(f"[*] Hard Cannot-Links (CL): {len(feasible_cl):,} pairs")
    print(f"[*] Dropped Conflicting CL: {dropped_cl}")
    print(f"[+] Constraints exported:   '{output_json}'")
    print("=" * 75)


if __name__ == "__main__":
    generate_hard_constraints()