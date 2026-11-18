"""
build_generation_hard_constraints.py
=============================================================================
Builds Hard Must-Links (ML) and Hard Cannot-Links (CL) for the generation fleet.
Target Output: .\my work\DATASET 1\GENERATION\generation_hard_constraints.json
=============================================================================
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx


SYNCHRONOUS_AREAS = {
    "CONTINENTAL_EUROPE": {
        "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "ES", "FR",
        "GR", "HR", "HU", "IT", "ME", "MK", "NL", "PL", "RO", "RS", "SI", "SK"
    },
    "NORDIC": {"FI", "NO", "SE"},
    "GREAT_BRITAIN": {"GB"},
    "BALTIC": {"EE", "LT"},
    "EASTERN": {"AZ", "BY", "GE", "MD", "RU"}
}

def get_zone(code):
    if not isinstance(code, str):
        return "UNKNOWN"
    c = code.strip().upper()
    for z, members in SYNCHRONOUS_AREAS.items():
        if c in members:
            return z
    return "UNKNOWN"


def generate_gen_constraints():
    print("=" * 75)
    print("GENERATING HARD CONSTRAINTS FOR GENERATION FLEET")
    print("=" * 75)

    base_dir = Path("my work") / "DATASET 1" / "GENERATION"
    meta_path = base_dir / "generation_metadata_lookup.csv"
    output_json = base_dir / "generation_hard_constraints.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing '{meta_path}'. Run prepare_generation_pccc.py first.")

    df_meta = pd.read_csv(meta_path)
    print(f"[*] Loaded {len(df_meta):,} generators from: {meta_path}")

    df_meta["pccc_idx"] = df_meta["pccc_index"] if "pccc_index" in df_meta.columns else df_meta.index

    # Identify Country & Synchronous Area
    country_col = [c for c in df_meta.columns if "map" in c.lower()]
    if country_col:
        df_meta["country"] = df_meta[country_col[0]].astype(str).str.upper()
    else:
        id_col = "generator_id" if "generator_id" in df_meta.columns else df_meta.columns[0]
        df_meta["country"] = df_meta[id_col].astype(str).str[:2].str.upper()

    df_meta["sync_zone"] = df_meta["country"].apply(get_zone)

    ml_pairs = []
    cl_pairs = []

    # 1. Hard Must-Links (ML): Units in the same multi-unit power plant
    plant_col = None
    for cand in ["locationname", "productiontype", "production_type"]:
        if cand in df_meta.columns:
            plant_col = cand
            break

    if plant_col:
        print(f"[*] Building Hard Must-Links based on: '{plant_col}'...")
        for _, grp in df_meta.groupby(plant_col):
            indices = grp["pccc_idx"].tolist()
            if 1 < len(indices) <= 5:  # Multi-unit plants
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        ml_pairs.append([int(indices[i]), int(indices[j])])
                        if len(ml_pairs) >= 200:
                            break
                if len(ml_pairs) >= 200:
                    break

    # 2. Hard Cannot-Links (CL): Asynchronous Zones + Opposite Risk Classes
    print("[*] Building Hard Cannot-Links across Asynchronous Zones and Risk Classes...")
    ce_units = df_meta[df_meta["sync_zone"] == "CONTINENTAL_EUROPE"]["pccc_idx"].tolist()
    async_units = df_meta[df_meta["sync_zone"].isin(["NORDIC", "GREAT_BRITAIN"])]["pccc_idx"].tolist()

    if ce_units and async_units:
        for u in ce_units[:25]:
            for v in async_units[:25]:
                cl_pairs.append([int(u), int(v)])
                if len(cl_pairs) >= 100:
                    break
            if len(cl_pairs) >= 100:
                break

    # Separate Class 0 (Baseload) from Class 3 (Chronic Unreliable)
    c0 = df_meta[df_meta["class_label"] == 0]["pccc_idx"].tolist()
    c3 = df_meta[df_meta["class_label"] == 3]["pccc_idx"].tolist()
    for i in range(min(len(c0), 25)):
        for j in range(min(len(c3), 25)):
            cl_pairs.append([int(c0[i]), int(c3[j])])
            if len(cl_pairs) >= 200:
                break
        if len(cl_pairs) >= 200:
            break

    # 3. Feasibility Pre-Check (Baumann & Hochbaum Sec 4.1)
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

    # 4. Save to JSON
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
    print("GENERATION CONSTRAINTS SUMMARY:")
    print("=" * 75)
    print(f"[*] Hard Must-Links (ML):   {len(ml_pairs):,} pairs")
    print(f"[*] Hard Cannot-Links (CL): {len(feasible_cl):,} pairs")
    print(f"[*] Dropped Conflicting CL: {dropped_cl}")
    print(f"[+] Output JSON saved to:   '{output_json}'")
    print("=" * 75)


if __name__ == "__main__":
    generate_gen_constraints()