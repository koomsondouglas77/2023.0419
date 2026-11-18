"""
build_continental_hard_constraints.py
=============================================================================
Builds Hard Must-Links (ML) and Hard Cannot-Links (CL) for Continental Blackouts:
- ML: Events co-located in the same regional grid zone with matching trigger cause.
- CL: Class 0 (Transient Disturbances) vs Class 3 (Catastrophic Collapses),
      and asynchronous regional boundaries (e.g. Nordic vs Continental Europe).
Target Output: .\my work\DATASET 2\CONTINENTAL\continental_hard_constraints.json
=============================================================================
"""

import json
from pathlib import Path
import pandas as pd
import networkx as nx


def generate_continental_constraints():
    print("=" * 75)
    print("GENERATING HARD CONSTRAINTS FOR CONTINENTAL BLACKOUTS")
    print("=" * 75)

    base_dir = Path("my work") / "DATASET 2" / "CONTINENTAL"
    meta_path = base_dir / "continental_metadata_lookup.csv"
    output_json = base_dir / "continental_hard_constraints.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing '{meta_path}'. Run prepare_continental_pccc.py first.")

    df_meta = pd.read_csv(meta_path)
    print(f"[*] Loaded {len(df_meta):,} continental events.")

    df_meta["pccc_idx"] = df_meta["pccc_index"] if "pccc_index" in df_meta.columns else df_meta.index

    ml_pairs = []
    cl_pairs = []

    # 1. Hard Must-Links (ML): Same country/region and same root trigger
    geo_col = [c for c in df_meta.columns if any(k in c.lower() for k in ["country", "area", "region"])]
    cause_col = [c for c in df_meta.columns if any(k in c.lower() for k in ["cause", "trigger", "type"])]

    if geo_col and cause_col:
        g = geo_col[0]
        c = cause_col[0]
        print(f"[*] Building Hard Must-Links based on common ({g}, {c})...")
        df_meta["geo_cause"] = df_meta[g].astype(str) + "_" + df_meta[c].astype(str)
        for _, grp in df_meta.groupby("geo_cause"):
            indices = grp["pccc_idx"].tolist()
            if 1 < len(indices) <= 4:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        ml_pairs.append([int(indices[i]), int(indices[j])])
                        if len(ml_pairs) >= 150:
                            break
                if len(ml_pairs) >= 150:
                    break

    # 2. Hard Cannot-Links (CL)
    # A. Opposite Operational Risk Classes: Class 0 (Transient) vs Class 3 (Catastrophic)
    if "class_label" in df_meta.columns:
        print("[*] Adding Cannot-Links between Class 0 (Transient) and Class 3 (Catastrophic)...")
        c0 = df_meta[df_meta["class_label"] == 0]["pccc_idx"].tolist()
        c3 = df_meta[df_meta["class_label"] == 3]["pccc_idx"].tolist()
        for i in range(min(len(c0), 20)):
            for j in range(min(len(c3), 20)):
                cl_pairs.append([int(c0[i]), int(c3[j])])
                if len(cl_pairs) >= 150:
                    break
            if len(cl_pairs) >= 150:
                break

    # B. Asynchronous Synchronous Boundary Decoupling (e.g. Nordic vs CE)
    if geo_col:
        g = geo_col[0]
        nordic_idx = df_meta[df_meta[g].astype(str).str.contains("NO|SE|FI|Norway|Sweden|Finland", case=False, na=False)]["pccc_idx"].tolist()
        ce_idx = df_meta[df_meta[g].astype(str).str.contains("DE|FR|IT|ES|Germany|France|Italy|Spain", case=False, na=False)]["pccc_idx"].tolist()
        if nordic_idx and ce_idx:
            print("[*] Adding Cannot-Links across asynchronous European synchronous areas...")
            for u in nordic_idx[:15]:
                for v in ce_idx[:15]:
                    cl_pairs.append([int(u), int(v)])
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
    print("CONTINENTAL CONSTRAINTS SUMMARY:")
    print("=" * 75)
    print(f"[*] Hard Must-Links (ML):   {len(ml_pairs):,} pairs")
    print(f"[*] Hard Cannot-Links (CL): {len(feasible_cl):,} pairs")
    print(f"[*] Dropped Conflicting CL: {dropped_cl}")
    print(f"[+] Output JSON saved to:   '{output_json}'")
    print("=" * 75)


if __name__ == "__main__":
    generate_continental_constraints()