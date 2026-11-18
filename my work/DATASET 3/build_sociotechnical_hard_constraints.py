"""
build_sociotechnical_hard_constraints.py
=============================================================================
Builds Hard Must-Links (ML) and Hard Cannot-Links (CL) for Dataset 3:
- ML: Regional co-location in the same macro-geographic zone.
- CL: Class 0 (Resilient Urban) vs Class 3 (Critical Hotspots) and
      island boundaries (Sicily/Sardinia vs Northern Alps).
Target Output: .\my work\DATASET 3\sociotechnical_hard_constraints.json
=============================================================================
"""

import json
from pathlib import Path
import pandas as pd
import networkx as nx


def generate_sociotech_constraints():
    print("=" * 75)
    print("GENERATING HARD CONSTRAINTS FOR SOCIO-TECHNICAL DATASET")
    print("=" * 75)

    base_dir = Path("my work") / "DATASET 3"
    meta_path = base_dir / "sociotechnical_metadata_lookup.csv"
    output_json = base_dir / "sociotechnical_hard_constraints.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing '{meta_path}'. Run prepare_sociotechnical_pccc.py first.")

    df_meta = pd.read_csv(meta_path)
    print(f"[*] Loaded {len(df_meta):,} units from metadata lookup.")

    df_meta["pccc_idx"] = df_meta["pccc_index"] if "pccc_index" in df_meta.columns else df_meta.index

    # Identify regional identifier column
    id_col = df_meta.columns[1] if df_meta.columns[0] == "pccc_index" else df_meta.columns[0]
    df_meta["reg_id"] = df_meta[id_col].astype(str).str.upper()

    ml_pairs = []
    cl_pairs = []

    # 1. Hard Must-Links (ML): Regional Macro-Zone Grouping (NUTS1 prefix ITC, ITD, ITE, ITF, ITG)
    df_meta["macro_zone"] = df_meta["reg_id"].str[:3]
    print("[*] Constructing Hard Must-Links on shared NUTS1 macro-zones...")
    for _, grp in df_meta.groupby("macro_zone"):
        indices = grp["pccc_idx"].tolist()
        if 1 < len(indices) <= 4:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ml_pairs.append([int(indices[i]), int(indices[j])])
                    if len(ml_pairs) >= 50:
                        break
            if len(ml_pairs) >= 50:
                break

    # 2. Hard Cannot-Links (CL)
    # A. Opposite Risk Classes: Class 0 (Resilient) vs Class 3 (Hotspots)
    if "class_label" in df_meta.columns:
        print("[*] Adding Cannot-Links between Class 0 (Resilient) and Class 3 (Hotspots)...")
        c0 = df_meta[df_meta["class_label"] == 0]["pccc_idx"].tolist()
        c3 = df_meta[df_meta["class_label"] == 3]["pccc_idx"].tolist()
        for i in range(min(len(c0), 10)):
            for j in range(min(len(c3), 10)):
                cl_pairs.append([int(c0[i]), int(c3[j])])
                if len(cl_pairs) >= 50:
                    break
            if len(cl_pairs) >= 50:
                break

    # B. Geographic Islanding: Sicily / Sardinia (ITG) vs Northwest Alps (ITC)
    island_idx = df_meta[df_meta["reg_id"].str.contains("ITG|SICIL|SARDEG", case=False, na=False)]["pccc_idx"].tolist()
    north_idx = df_meta[df_meta["reg_id"].str.contains("ITC|PIEM|LOMB", case=False, na=False)]["pccc_idx"].tolist()
    if island_idx and north_idx:
        print("[*] Adding Cannot-Links across decoupled island vs Alpine grid zones...")
        for u in island_idx[:10]:
            for v in north_idx[:10]:
                cl_pairs.append([int(u), int(v)])
                if len(cl_pairs) >= 50:
                    break
            if len(cl_pairs) >= 50:
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
    generate_sociotech_constraints()