"""
prepare_generation_pccc.py
=============================================================================
Generation fleet feature extraction and classification pipeline for PCCC.
Target Output Directory: .\my work\DATASET 1\GENERATION\
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_generation_pipeline():
    print("=" * 75)
    print("EUROPEAN GENERATION ASSET EXTRACTION & PCCC PREPARATION")
    print("=" * 75)

    # 1. Define explicit target output directory
    output_dir = Path("my work") / "DATASET 1" / "GENERATION"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Output Directory: {output_dir.resolve()}")

    # 2. Locate Generator Input CSV or Excel file
    cwd = Path.cwd()
    search_paths = [
        output_dir / "00_Generators_Outage_statistics_preprocessed_ENTSO_E_data.csv",
        cwd / "my work" / "00_Generators_Outage_statistics_preprocessed_ENTSO_E_data.csv",
        cwd / "00_Generators_Outage_statistics_preprocessed_ENTSO_E_data.csv",
    ]

    input_file = None
    for p in search_paths:
        if p.exists():
            input_file = p
            break

    if not input_file:
        candidates = list(cwd.rglob("*00_Generators*.csv"))
        if candidates:
            input_file = candidates[0]

    if not input_file:
        raise FileNotFoundError(
            "Could not locate '00_Generators_Outage_statistics_preprocessed_ENTSO_E_data.csv'.\n"
            "Please ensure the file is in 'my work\\DATASET 1\\GENERATION\\' or in your project root."
        )

    print(f"[*] Reading generator operational logs from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"[*] Total Raw Rows Loaded: {len(df):,}")

    # Standardize column headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Resolve primary asset ID column (EIC / Name)
    id_col = None
    for cand in ["powerresourceeic", "power_resource_eic", "resource_name", "unit_name", "asset_id", "mrid"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = df.columns[0]
        print(f"[!] Warning: Fallback using '{id_col}' as primary asset identifier.")

    print(f"[*] Primary Generator Identifier: '{id_col}' ({df[id_col].nunique():,} unique units)")

    # 3. Numeric Series Extraction & Cleaning
    def get_series(name, default=0.0):
        matches = [c for c in df.columns if c == name or c.startswith(name + "_")]
        if matches:
            return pd.to_numeric(df[matches[0]], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    df["_ah"] = get_series("ah", 0.0)
    df["_poh"] = get_series("poh", 0.0)
    df["_foh"] = get_series("foh", 0.0)
    df["_pdh"] = get_series("pdh", 0.0)
    df["_fdh"] = get_series("fdh", 0.0)
    df["_n04"] = get_series("n_04", 0.0)
    df["_n40"] = get_series("n_40", 0.0)

    # Aggregations by unique generator unit
    agg_spec = {
        "_ah": "sum",
        "_poh": "sum",
        "_foh": "sum",
        "_pdh": "sum",
        "_fdh": "sum",
        "_n04": "sum",
        "_n40": "sum",
    }

    # Preserve metadata if available
    for meta in ["mapcode", "mapcode1", "productiontype", "production_type", "locationname"]:
        if meta in df.columns:
            agg_spec[meta] = "first"

    grouped = df.groupby(id_col).agg(agg_spec).reset_index()
    grouped = grouped.rename(columns={id_col: "generator_id"})

    # 4. Compute 5-State Continuous Markov Probabilities & IEEE Std 762 EFOR
    total_hours = (grouped["_ah"] + grouped["_poh"] + grouped["_foh"] + 
                   grouped["_pdh"] + grouped["_fdh"]).replace(0, np.nan)

    grouped["P0"] = (grouped["_ah"] / total_hours).fillna(1.0).clip(0.0, 1.0)
    grouped["P1"] = (grouped["_pdh"] / total_hours).fillna(0.0).clip(0.0, 1.0)
    grouped["P2"] = (grouped["_fdh"] / total_hours).fillna(0.0).clip(0.0, 1.0)
    grouped["P3"] = (grouped["_poh"] / total_hours).fillna(0.0).clip(0.0, 1.0)
    grouped["P4"] = (grouped["_foh"] / total_hours).fillna(0.0).clip(0.0, 1.0)

    # Equivalent Forced Outage Rate (EFOR): (foh + fdh) / (ah + foh + fdh)
    denom_efor = (grouped["_ah"] + grouped["_foh"] + grouped["_fdh"]).replace(0, np.nan)
    grouped["EFOR"] = ((grouped["_foh"] + grouped["_fdh"]) / denom_efor).fillna(0.0).clip(0.0, 1.0)

    # Total Derated Fraction (P1 + P2)
    grouped["Derated_Fraction"] = (grouped["P1"] + grouped["P2"]).clip(0.0, 1.0)

    # 5. Export Aggregated Metrics CSV
    agg_metrics_path = output_dir / "generation_aggregated_metrics.csv"
    grouped.to_csv(agg_metrics_path, index=False)
    print(f"[+] Saved aggregated metrics -> '{agg_metrics_path}'")

    # 6. Balanced 2x2 Median Split Classification
    active_efor = grouped[grouped["EFOR"] > 0]["EFOR"]
    active_derate = grouped[grouped["Derated_Fraction"] > 0]["Derated_Fraction"]

    tau_efor = active_efor.median() if len(active_efor) > 0 else 0.005
    tau_derate = active_derate.median() if len(active_derate) > 0 else 0.01

    print("\n" + "-" * 75)
    print("MEDIAN SPLIT THRESHOLDS (GENERATION FLEET)")
    print("-" * 75)
    print(f"[*] EFOR Threshold:             {tau_efor:.6f}")
    print(f"[*] Derated Fraction Threshold: {tau_derate:.6f}")
    print("-" * 75 + "\n")

    def assign_gen_class(row):
        h_efor = row["EFOR"] >= tau_efor
        h_derate = row["Derated_Fraction"] >= tau_derate
        if not h_efor and not h_derate:
            return 0  # Class 0: Baseload Stable (Low EFOR, Low Derating)
        elif not h_efor and h_derate:
            return 1  # Class 1: Flexible / Throttled (Low EFOR, High Derating)
        elif h_efor and not h_derate:
            return 2  # Class 2: High Trip Volatile (High EFOR, Low Derating)
        else:
            return 3  # Class 3: Chronic Unreliable (High EFOR, High Derating)

    grouped["class_label"] = grouped.apply(assign_gen_class, axis=1)

    print("BALANCED CLASS DISTRIBUTION:")
    counts = grouped["class_label"].value_counts().sort_index()
    for c_id, count in counts.items():
        pct = (count / len(grouped)) * 100
        print(f"  - Class {c_id}: {count:5d} units ({pct:5.1f}%)")

    # 7. Build PCCC Input Matrix with MinMaxScaler
    # Continuous features: [P1, P2, P3, P4, EFOR]
    feature_cols = ["P1", "P2", "P3", "P4", "EFOR"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(grouped[feature_cols].values)

    pccc_df = pd.DataFrame(X_scaled, columns=feature_cols)
    pccc_df["target_class"] = grouped["class_label"].values

    # 8. Export PCCC-ready and Metadata files
    pccc_ready_path = output_dir / "generation_pccc_ready.csv"
    pccc_df.to_csv(pccc_ready_path, index=False)
    print(f"\n[+] Exported PCCC-Ready Dataset -> '{pccc_ready_path}'")
    print(f"    - Dimensions: {pccc_df.shape[0]} generators x {pccc_df.shape[1] - 1} features (+ target_class)")

    meta_cols = ["generator_id", "class_label"]
    for m in ["mapcode", "mapcode1", "productiontype", "production_type", "locationname"]:
        if m in grouped.columns:
            meta_cols.append(m)

    meta_df = grouped[meta_cols].copy()
    meta_df.index.name = "pccc_index"
    meta_path = output_dir / "generation_metadata_lookup.csv"
    meta_df.to_csv(meta_path)
    print(f"[+] Exported Metadata Lookup Map -> '{meta_path}'")
    print("=" * 75)


if __name__ == "__main__":
    run_generation_pipeline()