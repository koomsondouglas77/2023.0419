"""
prepare_transmission_pccc.py
=============================================================================
Classifies European transmission assets into 4 operational risk archetypes,
strips string identifiers (EIC codes), scales continuous features, and formats
the dataset into PCCC format (X in cols 0..d-1, y in last column).

Author: Douglas Kojo Kumi Koomson (RRE Lab, ETH Zurich / Ashesi University)
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(
    input_csv="transmission_aggregated_metrics.csv",
    output_pccc_csv="transmission_pccc_ready.csv",
    output_meta_csv="transmission_metadata_lookup.csv",
):
    # 1. Locate input file
    path = Path(input_csv)
    if not path.exists():
        candidates = list(Path(".").rglob("*transmission_aggregated*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate '{input_csv}'.")
        path = candidates[0]

    print("=" * 75)
    print("TRANSMISSION ASSET CLASSIFICATION & PCCC FORMAT CONVERSION")
    print("=" * 75)
    print(f"[*] Reading aggregated metrics from: {path}")

    df = pd.read_csv(path)
    print(f"[*] Loaded {len(df):,} transmission assets.")

    # Standardize column headers
    df.columns = [c.strip() for c in df.columns]

    # Resolve primary identifier column
    id_col = None
    for cand in ["asset_id", "powerresourceeic", "power_resource_eic", "resource_name", "mrid"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = df.columns[0]

    # Ensure required numeric columns are present and clean
    numeric_features = [
        "P0", "P3", "P4", "FOR", "MTTR_hours", "lambda_forced_rate", "Delta_NTC_mean_MW"
    ]
    for col in numeric_features:
        if col not in df.columns:
            # Fallback check
            matched = [c for c in df.columns if col.split("_")[0].lower() in c.lower()]
            if matched:
                df[col] = pd.to_numeric(df[matched[0]], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 2. Determine empirical threshold boundaries
    # Filter non-zero values to compute realistic thresholds
    for_active = df[df["FOR"] > 0]["FOR"]
    mttr_active = df[df["MTTR_hours"] > 0]["MTTR_hours"]

    tau_for = for_active.median() if len(for_active) > 0 else df["FOR"].median()
    tau_mttr = mttr_active.median() if len(mttr_active) > 0 else df["MTTR_hours"].median()

    print("\n" + "-" * 75)
    print("EMPIRICAL THRESHOLDS (Method 1: Reliability Quadrants)")
    print("-" * 75)
    print(f"[*] FOR Active Median  (tau_for):  {tau_for:.6f}")
    print(f"[*] MTTR Active Median (tau_mttr): {tau_mttr:.2f} hours")
    print("-" * 75 + "\n")

    # 3. Assign 4 Operational Risk Classes
    def assign_class(row):
        is_high_for = row["FOR"] > tau_for
        is_high_mttr = row["MTTR_hours"] > tau_mttr

        if not is_high_for and not is_high_mttr:
            return 0  # Robust Base Backbone
        elif not is_high_for and is_high_mttr:
            return 1  # Maintenance-Intensive
        elif is_high_for and not is_high_mttr:
            return 2  # Volatile / Fast-Tripping
        else:
            return 3  # Chronic High-Risk Chokepoint

    df["class_label"] = df.apply(assign_class, axis=1)

    class_names = {
        0: "Robust Base Backbone",
        1: "Maintenance-Intensive",
        2: "Volatile / Fast-Tripping",
        3: "Chronic High-Risk Chokepoint",
    }
    df["class_name"] = df["class_label"].map(class_names)

    # Display class distribution
    print("CLASS DISTRIBUTION SUMMARY:")
    summary = df.groupby(["class_label", "class_name"]).size().reset_index(name="Asset_Count")
    summary["Percentage"] = (summary["Asset_Count"] / len(df)) * 100
    print(summary.to_string(index=False))
    print()

    # 4. Separate and export Metadata Lookup Table (Preserving original EIC)
    meta_cols = [id_col, "class_label", "class_name"]
    for extra in ["mapcode", "production_type", "in_area", "out_area"]:
        if extra in df.columns:
            meta_cols.append(extra)

    df_meta = df[meta_cols].copy()
    df_meta.index.name = "pccc_index"  # Line number 0..n-1 in PCCC data matrix
    df_meta.to_csv(output_meta_csv)
    print(f"[+] Saved Metadata Lookup Map -> '{output_meta_csv}'")

    # 5. Build PCCC Numerical Input Matrix
    # Scale features using StandardScaler (Z-score: mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features].values)

    pccc_df = pd.DataFrame(X_scaled, columns=numeric_features)
    # Append class_label as the very last column
    pccc_df["target_class"] = df["class_label"].values

    # Export clean PCCC CSV
    pccc_df.to_csv(output_pccc_csv, index=False)
    print(f"[+] Saved PCCC-Ready Dataset   -> '{output_pccc_csv}'")
    print(f"    - Dimensions: {pccc_df.shape[0]} objects x {pccc_df.shape[1] - 1} features (+ 1 class column)")
    print(f"    - Class label column: '{pccc_df.columns[-1]}' (Values: {sorted(pccc_df['target_class'].unique())})")
    print("=" * 75)


if __name__ == "__main__":
    prepare_data()