"""
build_national_hard_constraints.py
=============================================================================
Builds Hard Must-Links (ML) and Cannot-Links (CL) for National Blackout Events:
- ML: Sub-events sharing the exact same substation / regional grid area.
- CL: Minor localized trips (Class 0) vs severe systemic blackouts (Class 3),
      and geographically decoupled island/peninsular areas (Palermo/Cagliari vs Milan/Turin).
=============================================================================
"""

import json
from pathlib import Path
import pandas as pd
import networkx as nx


def generate_national_constraints():
    print("=" * 75)
    print("GENERATING HARD CONSTRAINTS FOR NATIONAL BLACKOUTS")
    print("=" * 75)

    base_dir = Path("my work") / "DATASET 2" / "NATIONAL"
    meta_path = base_dir / "national_metadata_lookup.csv"
    output_json = base_dir / "national_hard_constraints.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing '{meta_path}'. Run prepare_national_pccc.py first.")

    df_meta = pd.read_csv(meta_path)
    print(f"[*] Loaded {len(df_meta):,} events from metadata lookup.")

    df_meta["pccc_idx"] = df_meta["pccc_index"] if "pccc_index" in df_meta.columns else df_meta.index

    ml_pairs = []
    cl_pairs = []

    # 1. Hard Must-Links (ML): Co-located failures in the same substation/zone with matching cause
    zone_col = [c for c in df_meta.columns if any(k in c.lower() for k in ["zone", "region", "area"])]
    cause_col = [c for c in df_meta.columns if any(k in c.lower() for k in ["cause", "origin"])]

    if zone_col and cause_col:
        z = zone_col[0]
        c = cause_col[0]
        print(f"[*] Building Hard Must-Links on shared ({z}, {c})...")
        df_meta["geo_cause_group"] = df_meta[z].astype(str) + "_" + df_meta[c].astype(str)
        for _, grp in df_meta.groupby("geo_cause_group"):
            indices = grp["pccc_idx"].tolist()
            if 1 < len(indices) <= 4:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        ml_pairs.append([int(indices[i]), int(indices[j])])
                        if len(ml_pairs) >= 200:
                            break
                if len(ml_pairs) >= 200:
                    break

    # 2. Hard Cannot-Links (CL)
    # A. Geographic Islanding: Palermo (Sicily) / Cagliari (Sardinia) vs Milan/Turin (Northern Alps)
    if zone_col:
        z = zone_col[0]
        island_idx = df_meta[df_meta[z].astype(str).str.contains("Palermo|Cagliari|Sicilia|Sardegna", case=False, na=False)]["pccc_idx"].tolist()
        north_idx = df_meta[df_meta[z].astype(str).str.contains("Milano|Torino|Milan|Turin", case=False, na=False)]["pccc_idx"].tolist()
        if island_idx and north_idx:
            print("[*] Adding Cannot-Links across non-contiguous grid zones (Islands vs Northern Mainland)...")
            for u in island_idx[:25]:
                for v in north_idx[:25]:
                    cl_pairs.append([int(u), int(v)])
                    if len(cl_pairs) >= 100:
                        break
                if len(cl_pairs) >= 100:
                    break

    # B. Opposite Operational Classes: Class 0 (Minor) vs Class 3 (Severe Systemic)
    if "class_label" in df_meta.columns:
        print("[*] Adding Cannot-Links between Class 0 (Minor) and Class 3 (Severe Systemic)...")
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

    # 4. Save JSON
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
    print("CONSTRAINTS SUMMARY:")
    print("=" * 75)
    print(f"[*] Hard Must-Links (ML):   {len(ml_pairs):,} pairs")
    print(f"[*] Hard Cannot-Links (CL): {len(feasible_cl):,} pairs")
    print(f"[*] Dropped Conflicting CL: {dropped_cl}")
    print(f"[+] Constraints exported:   '{output_json}'")
    print("=" * 75)


if __name__ == "__main__":
    generate_national_constraints()