"""
prepare_sociotechnical_pccc.py
=============================================================================
Processes Dataset 3: Italian Socio-Technical Vulnerability Database.
Integrates failure events, weather indicators, IVI, and CVI indices across
Italian NUTS2 regions. Formulates an R^6 continuous feature matrix and 
classifies regions into 4 socio-technical vulnerability archetypes.

Target Directory: .\my work\DATASET 3\
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_sociotechnical_pipeline():
    print("=" * 75)
    print("ITALIAN SOCIO-TECHNICAL VULNERABILITY DATASET PREPARATION")
    print("=" * 75)

    # 1. Output directory
    out_dir = Path("my work") / "DATASET 3"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Directory: {out_dir.resolve()}")

    cwd = Path.cwd()

    def find_file(pattern):
        # Look in out_dir, cwd/my work, cwd, then search recursively
        search_dirs = [out_dir, cwd / "my work", cwd]
        for d in search_dirs:
            matches = list(d.glob(pattern))
            if matches:
                return matches[0]
        rec = list(cwd.rglob(pattern))
        return rec[0] if rec else None

    # 2. Locate Source Files
    f_all = find_file("*05_All_Indices*.xlsx")
    f_events = find_file("*01_Failure_Events*.xlsx")
    f_weather = find_file("*02_Weather*.xlsx")
    f_cvi = find_file("*03_CVI*.xlsx")
    f_ivi = find_file("*04_IVI*.xlsx")

    # If 05_All_Indices exists, it already aggregates NUTS2 indicators
    if f_all and f_all.exists():
        print(f"[*] Reading consolidated indices from: {f_all.name}")
        df_master = pd.read_excel(f_all)
    else:
        print("[*] 05_All_Indices not found directly. Loading individual constituent files...")
        if not (f_cvi and f_ivi):
            raise FileNotFoundError("Missing CVI or IVI Excel files in workspace.")
        df_cvi = pd.read_excel(f_cvi)
        df_ivi = pd.read_excel(f_ivi)
        
        # Standardize joining key (NUTS2 code or Region Name)
        key_col = [c for c in df_cvi.columns if any(k in c.lower() for k in ["nuts", "region", "cod"])][0]
        df_master = pd.merge(df_cvi, df_ivi, on=key_col, how="inner", suffixes=("_cvi", "_ivi"))

    df_master.columns = [c.strip().lower().replace(" ", "_") for c in df_master.columns]
    print(f"[*] Total regional units loaded: {len(df_master):,}")

    # Map column headers dynamically
    def get_col(candidates, default=None):
        for cand in candidates:
            for c in df_master.columns:
                if cand in c:
                    return c
        return default

    col_nuts = get_col(["nuts2", "nuts", "cod_reg", "region_code", "region", "nome"], default=df_master.columns[0])
    col_cvi = get_col(["cvi", "community", "social"], default=None)
    col_ivi = get_col(["ivi", "infra", "infrastructure"], default=None)
    col_lambda = get_col(["failure_rate", "lambda", "frequency", "n_failures", "events"], default=None)
    col_mttr = get_col(["mttr", "duration", "hours", "repair_time"], default=None)
    col_dns = get_col(["dns", "unsupplied", "loss", "ens"], default=None)
    col_weather = get_col(["weather", "hazard", "climate", "storm", "wind"], default=None)

    print("\n[*] COLUMN RESOLUTION:")
    print(f"    - NUTS2 Identifier:      '{col_nuts}'")
    print(f"    - Community Vuln (CVI):  '{col_cvi}'")
    print(f"    - Infra Vuln (IVI):      '{col_ivi}'")
    print(f"    - Failure Frequency:     '{col_lambda}'")
    print(f"    - MTTR / Duration:       '{col_mttr}'")
    print(f"    - DNS / Energy Loss:     '{col_dns}'")
    print(f"    - Severe Weather Index:  '{col_weather}'")

    # Numeric series extraction with safe defaults
    def to_num(c, default=0.0):
        if c and c in df_master.columns:
            return pd.to_numeric(df_master[c], errors="coerce").fillna(default)
        return pd.Series(default, index=df_master.index)

    df_master["_cvi"] = to_num(col_cvi, 0.5).clip(0.0, 1.0)
    df_master["_ivi"] = to_num(col_ivi, 0.5).clip(0.0, 1.0)
    df_master["_lambda"] = to_num(col_lambda, 1.0).clip(lower=0.0)
    df_master["_mttr"] = to_num(col_mttr, 10.0).clip(lower=0.0)
    df_master["_dns"] = to_num(col_dns, 0.0).clip(lower=0.0)
    df_master["_weather"] = to_num(col_weather, 0.3).clip(0.0, 1.0)

    # 3. Balanced 2x2 Median Split: Technical Grid Risk vs Community Vulnerability
    # Combine failure rate and MTTR into a technical stress proxy
    tech_risk = np.log1p(df_master["_lambda"]) + np.log1p(df_master["_mttr"])
    tau_tech = tech_risk.median()
    tau_cvi = df_master["_cvi"].median()

    print("\n" + "-" * 75)
    print("MEDIAN SPLIT THRESHOLDS (SOCIO-TECHNICAL VULNERABILITY):")
    print("-" * 75)
    print(f"[*] Technical Grid Stress Median (tau_tech): {tau_tech:.4f}")
    print(f"[*] Community Vulnerability Median (tau_cvi): {tau_cvi:.4f}")
    print("-" * 75)

    def assign_sociotech_class(row):
        h_tech = (np.log1p(row["_lambda"]) + np.log1p(row["_mttr"])) >= tau_tech
        h_cvi = row["_cvi"] >= tau_cvi
        if not h_tech and not h_cvi:
            return 0  # Class 0: Resilient Urban Backbone (Low Stress, Low CVI)
        elif not h_tech and h_cvi:
            return 1  # Class 1: Socially Vulnerable / Stable Grid (Low Stress, High CVI)
        elif h_tech and not h_cvi:
            return 2  # Class 2: Climate-Stressed / High-Coping (High Stress, Low CVI)
        else:
            return 3  # Class 3: Critical Multi-Hazard Hotspots (High Stress, High CVI)

    df_master["class_label"] = df_master.apply(assign_sociotech_class, axis=1)

    print("\nCLASS DISTRIBUTION:")
    counts = df_master["class_label"].value_counts().sort_index()
    for c_id, count in counts.items():
        pct = (count / len(df_master)) * 100
        print(f"  - Class {c_id}: {count:5d} units ({pct:5.1f}%)")

    # 4. Feature Matrix R^6 (Log transforms on heavy-tailed metrics, MinMax scaling)
    feat_df = pd.DataFrame()
    feat_df["log_lambda"] = np.log1p(df_master["_lambda"])
    feat_df["log_mttr"] = np.log1p(df_master["_mttr"])
    feat_df["log_dns"] = np.log1p(df_master["_dns"])
    feat_df["ivi"] = df_master["_ivi"]
    feat_df["cvi"] = df_master["_cvi"]
    feat_df["weather_hazard"] = df_master["_weather"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pccc_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    pccc_df["target_class"] = df_master["class_label"].values

    # 5. Export PCCC-Ready and Metadata Files
    pccc_file = out_dir / "sociotechnical_pccc_ready.csv"
    pccc_df.to_csv(pccc_file, index=False)
    print(f"\n[+] Exported PCCC-Ready Dataset -> '{pccc_file}'")
    print(f"    - Dimensions: {pccc_df.shape[0]} units x {pccc_df.shape[1] - 1} features (+ target_class)")

    meta_cols = [col_nuts, "class_label", "_cvi", "_ivi", "_lambda", "_mttr"]
    meta_df = df_master[meta_cols].copy()
    meta_df.index.name = "pccc_index"
    meta_file = out_dir / "sociotechnical_metadata_lookup.csv"
    meta_df.to_csv(meta_file)
    print(f"[+] Exported Metadata Lookup Map -> '{meta_file}'")
    print("=" * 75)


if __name__ == "__main__":
    run_sociotechnical_pipeline()