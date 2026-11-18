"""
prepare_continental_pccc.py
=============================================================================
Processes the Continental European Blackout Dataset:
1. Reads Continental_dataset.xlsx with case-insensitive column discovery.
2. Derives orthogonal features: ln(ENS), ln(DNS), ln(Duration), Geographic Scale.
3. Classifies events into 4 systemic risk archetypes via 2x2 median split.
4. Scales features to [0, 1] preserving Euclidean distance geometry.
5. Exports continental_pccc_ready.csv and continental_metadata_lookup.csv.
Target Directory: .\my work\DATASET 2\CONTINENTAL\
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_continental_pipeline():
    print("=" * 75)
    print("CONTINENTAL EUROPEAN BLACKOUT DATASET EXTRACTION & PCCC PREPARATION")
    print("=" * 75)

    # 1. Target output directory
    out_dir = Path("my work") / "DATASET 2" / "CONTINENTAL"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Directory: {out_dir.resolve()}")

    # 2. Locate input Excel file
    cwd = Path.cwd()
    search_paths = [
        out_dir / "Continental_dataset.xlsx",
        cwd / "my work" / "Continental_dataset.xlsx",
        cwd / "Continental_dataset.xlsx",
    ]
    input_file = None
    for p in search_paths:
        if p.exists():
            input_file = p
            break

    if not input_file:
        cands = list(cwd.rglob("*Continental*.xlsx"))
        if cands:
            input_file = cands[0]

    if not input_file:
        raise FileNotFoundError(
            "Could not locate 'Continental_dataset.xlsx'.\n"
            "Please place the file in 'my work\\DATASET 2\\CONTINENTAL\\' or project root."
        )

    print(f"[*] Reading dataset: {input_file}")
    xls = pd.ExcelFile(input_file)
    print(f"[*] Available Sheets: {xls.sheet_names}")
    sheet = xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet)
    print(f"[*] Loaded {len(df):,} records from sheet '{sheet}'")

    # Clean headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Dynamically match columns
    def find_col(candidates, default=None):
        for cand in candidates:
            for c in df.columns:
                if cand == c or cand in c:
                    return c
        return default

    col_id = find_col(["event_id", "id", "index", "no", "event"], default=df.columns[0])
    col_ens = find_col(["energy_not_supplied", "ens", "energy_lost", "energy_mwh", "mwh"], default=None)
    col_dns = find_col(["power_loss", "dns", "loss_mw", "power_mw", "mw"], default=None)
    col_dur = find_col(["duration", "restoration_time", "hours", "minutes", "dur"], default=None)
    col_country = find_col(["country", "countries", "area", "region", "location"], default=None)
    col_cause = find_col(["main_cause", "cause", "trigger", "root_cause", "type"], default=None)
    col_year = find_col(["year", "date", "time"], default=None)

    print("\n[*] COLUMN MAPPINGS IDENTIFIED:")
    print(f"    - Event Identifier: '{col_id}'")
    print(f"    - Energy Lost (ENS):'{col_ens}'")
    print(f"    - Power Loss (DNS): '{col_dns}'")
    print(f"    - Duration:         '{col_dur}'")
    print(f"    - Country / Area:   '{col_country}'")
    print(f"    - Main Cause:       '{col_cause}'")
    print(f"    - Year / Date:      '{col_year}'")

    # 3. Numeric Sanitation & Fallback Computation
    # Clean ENS (MWh)
    s_ens = pd.to_numeric(df[col_ens], errors="coerce").fillna(0.0) if col_ens else pd.Series(0.0, index=df.index)
    
    # Clean DNS (MW)
    s_dns = pd.to_numeric(df[col_dns], errors="coerce").fillna(0.0) if col_dns else pd.Series(0.0, index=df.index)
    
    # Clean Duration (hours)
    s_dur = pd.to_numeric(df[col_dur], errors="coerce").fillna(0.0) if col_dur else pd.Series(0.0, index=df.index)

    # Cross-fill if one metric is missing: ENS approx = DNS * Duration
    if (s_ens.sum() == 0) and (s_dns.sum() > 0) and (s_dur.sum() > 0):
        print("[!] Reconstructing approximate ENS = DNS * (Duration / 2)...")
        s_ens = s_dns * (s_dur / 2.0)
    elif (s_dns.sum() == 0) and (s_ens.sum() > 0) and (s_dur.sum() > 0):
        print("[!] Reconstructing approximate DNS = ENS / max(Duration, 1)...")
        s_dns = s_ens / np.maximum(s_dur, 1.0)

    df["_ens"] = s_ens.clip(lower=0.0)
    df["_dns"] = s_dns.clip(lower=0.0)
    df["_dur"] = s_dur.clip(lower=0.0)

    # Geographic / Complexity metric (e.g. number of countries involved if string list)
    if col_country:
        df["_country_count"] = df[col_country].astype(str).apply(
            lambda x: len([c for c in x.replace(";", ",").replace("/", ",").split(",") if c.strip()])
        ).clip(lower=1)
    else:
        df["_country_count"] = 1.0

    # 4. Balanced 2x2 Median Split (Loss Magnitude vs Recovery Duration)
    # Use ENS (or DNS if ENS is uniform) as the primary impact axis
    impact_series = df["_ens"] if df["_ens"].sum() > 0 else df["_dns"]
    active_impact = impact_series[impact_series > 0]
    active_dur = df[df["_dur"] > 0]["_dur"]

    tau_impact = active_impact.median() if len(active_impact) > 0 else 1000.0
    tau_dur = active_dur.median() if len(active_dur) > 0 else 5.0

    print("\n" + "-" * 75)
    print("MEDIAN SPLIT THRESHOLDS (CONTINENTAL DISTURBANCES):")
    print("-" * 75)
    print(f"[*] Impact Threshold (ENS/DNS): {tau_impact:.2f}")
    print(f"[*] Duration Threshold:        {tau_dur:.2f} hours/units")
    print("-" * 75)

    def assign_continental_class(row):
        h_impact = (row["_ens"] >= tau_impact) if row["_ens"] > 0 else (row["_dns"] >= tau_impact)
        h_dur = row["_dur"] >= tau_dur
        if not h_impact and not h_dur:
            return 0  # Class 0: Transient Grid Disturbance (Low Impact, Fast Restoration)
        elif not h_impact and h_dur:
            return 1  # Class 1: Extended Technical Recovery (Low Impact, Slow Restoration)
        elif h_impact and not h_dur:
            return 2  # Class 2: Acute High-Power Shock (High Impact, Fast Restoration)
        else:
            return 3  # Class 3: Catastrophic Systemic Collapse (High Impact, Prolonged Outage)

    df["class_label"] = df.apply(assign_continental_class, axis=1)

    print("\nCONTINENTAL RISK CLASS DISTRIBUTION:")
    counts = df["class_label"].value_counts().sort_index()
    for c_id, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  - Class {c_id}: {count:5d} events ({pct:5.1f}%)")

    # 5. Continuous Feature Space (R^4 with Log-Transformations)
    feat_df = pd.DataFrame()
    feat_df["log_ENS"] = np.log1p(df["_ens"])
    feat_df["log_DNS"] = np.log1p(df["_dns"])
    feat_df["log_Duration"] = np.log1p(df["_dur"])
    feat_df["log_Geographic_Spread"] = np.log1p(df["_country_count"])

    # Scale to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pccc_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    pccc_df["target_class"] = df["class_label"].values

    # 6. Export PCCC-ready and Metadata lookup files
    pccc_file = out_dir / "continental_pccc_ready.csv"
    pccc_df.to_csv(pccc_file, index=False)
    print(f"\n[+] Exported PCCC-Ready Dataset -> '{pccc_file}'")
    print(f"    - Dimensions: {pccc_df.shape[0]} events x {pccc_df.shape[1] - 1} features (+ target_class)")

    meta_cols = [col_id, "class_label", "_ens", "_dns", "_dur", "_country_count"]
    for m in [col_country, col_cause, col_year]:
        if m and m in df.columns:
            meta_cols.append(m)

    meta_df = df[meta_cols].copy()
    meta_df.index.name = "pccc_index"
    meta_file = out_dir / "continental_metadata_lookup.csv"
    meta_df.to_csv(meta_file)
    print(f"[+] Exported Metadata Lookup Map -> '{meta_file}'")
    print("=" * 75)


if __name__ == "__main__":
    run_continental_pipeline()