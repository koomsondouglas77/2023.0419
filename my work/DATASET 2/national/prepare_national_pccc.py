"""
prepare_national_pccc.py
=============================================================================
Processes the Italian National Blackout Dataset (TERNA):
1. Aggregates multi-failure events sharing the same Event ID (DNS, duration).
2. Derives orthogonal features: ln(DNS), ln(Duration), Voltage, Component Count.
3. Classifies events into 4 operational risk archetypes (Single vs Cascade dynamics).
4. Exports national_pccc_ready.csv and national_metadata_lookup.csv.
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_national_pipeline():
    print("=" * 75)
    print("NATIONAL BLACKOUT DATASET EXTRACTION & PCCC PREPARATION")
    print("=" * 75)

    # 1. Output directory
    out_dir = Path("my work") / "DATASET 2" / "NATIONAL"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Directory: {out_dir.resolve()}")

    # 2. Locate input Excel file
    cwd = Path.cwd()
    search_paths = [
        out_dir / "National_dataset.xlsx",
        cwd / "my work" / "National_dataset.xlsx",
        cwd / "National_dataset.xlsx",
    ]
    input_file = None
    for p in search_paths:
        if p.exists():
            input_file = p
            break
    if not input_file:
        cands = list(cwd.rglob("*National_dataset*.xlsx"))
        if cands:
            input_file = cands[0]

    if not input_file:
        raise FileNotFoundError("Could not find National_dataset.xlsx in project folder.")

    print(f"[*] Reading dataset: {input_file}")
    # Read the primary sheet
    xls = pd.ExcelFile(input_file)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet)
    print(f"[*] Raw rows loaded: {len(df):,} from sheet '{sheet}'")

    # Clean headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map core columns
    def find_col(candidates, default=None):
        for cand in candidates:
            for c in df.columns:
                if cand in c:
                    return c
        return default

    col_id = find_col(["event_id", "id_event", "eventid", "id"], default=df.columns[0])
    col_dns = find_col(["power_loss", "dns", "loss_mw", "power"], default=None)
    col_dur = find_col(["duration", "dur", "time_outage", "recovery"], default=None)
    col_volt = find_col(["voltage", "kv", "tension"], default=None)
    col_class = find_col(["classification", "class", "type_event"], default=None)
    col_cause = find_col(["main_cause", "cause", "origin"], default=None)
    col_zone = find_col(["region", "area", "zone", "substation"], default=None)

    print(f"    - Event ID:       '{col_id}'")
    print(f"    - Power Loss:     '{col_dns}'")
    print(f"    - Duration:       '{col_dur}'")
    print(f"    - Voltage Level:  '{col_volt}'")
    print(f"    - Classification: '{col_class}'")
    print(f"    - Cause:          '{col_cause}'")
    print(f"    - Grid Area:      '{col_zone}'")

    # 3. Numeric sanitation
    df["_dns"] = pd.to_numeric(df[col_dns], errors="coerce").fillna(0.0) if col_dns else 0.0
    df["_dur"] = pd.to_numeric(df[col_dur], errors="coerce").fillna(0.0) if col_dur else 0.0
    
    # Extract numerical voltage (e.g. '380 kV' -> 380)
    if col_volt:
        df["_volt"] = df[col_volt].astype(str).str.extract(r'(\d+)')[0]
        df["_volt"] = pd.to_numeric(df["_volt"], errors="coerce").fillna(132.0)
    else:
        df["_volt"] = 132.0

    # 4. Aggregate multiple component failures into single Event IDs
    print("[*] Aggregating multi-component failures by unique Event ID...")
    agg_rules = {
        "_dns": "sum",
        "_dur": "max",
        "_volt": "max",
    }
    if col_class:
        agg_rules[col_class] = "first"
    if col_cause:
        agg_rules[col_cause] = "first"
    if col_zone:
        agg_rules[col_zone] = "first"

    grouped = df.groupby(col_id).agg(agg_rules)
    # Component count per event
    grouped["n_components"] = df.groupby(col_id).size()
    grouped = grouped.reset_index()

    print(f"[*] Total aggregated distinct events: {len(grouped):,}")

    # 5. Define 4 Balanced Operational Risk Archetypes
    # Split using median DNS and median Duration for active events
    active_dns = grouped[grouped["_dns"] > 0]["_dns"]
    active_dur = grouped[grouped["_dur"] > 0]["_dur"]

    tau_dns = active_dns.median() if len(active_dns) > 0 else 5.0
    tau_dur = active_dur.median() if len(active_dur) > 0 else 30.0

    print("\n" + "-" * 75)
    print(f"[*] DNS Threshold:      {tau_dns:.2f} MW")
    print(f"[*] Duration Threshold: {tau_dur:.2f} min/sec")
    print("-" * 75)

    def assign_event_class(row):
        h_dns = row["_dns"] >= tau_dns
        h_dur = row["_dur"] >= tau_dur
        if not h_dns and not h_dur:
            return 0  # Class 0: Minor Localized Trip (Low DNS, Fast Restoration)
        elif not h_dns and h_dur:
            return 1  # Class 1: Chronic Low-Power Disruption (Low DNS, Long Repair)
        elif h_dns and not h_dur:
            return 2  # Class 2: High-Power Transient Outage (High DNS, Fast Restoration)
        else:
            return 3  # Class 3: Severe Systemic Blackout (High DNS, Long Repair, Cascading)

    grouped["class_label"] = grouped.apply(assign_event_class, axis=1)

    print("\nCLASS DISTRIBUTION:")
    counts = grouped["class_label"].value_counts().sort_index()
    for c_id, count in counts.items():
        pct = (count / len(grouped)) * 100
        print(f"  - Class {c_id}: {count:5d} events ({pct:5.1f}%)")

    # 6. Feature Space Formulation (Log transform to manage power-law tails)
    feat_df = pd.DataFrame()
    feat_df["log_DNS"] = np.log1p(grouped["_dns"].clip(lower=0))
    feat_df["log_Duration"] = np.log1p(grouped["_dur"].clip(lower=0))
    feat_df["Voltage"] = grouped["_volt"]
    feat_df["log_Components"] = np.log1p(grouped["n_components"])

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pccc_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    pccc_df["target_class"] = grouped["class_label"].values

    # 7. Export outputs
    pccc_file = out_dir / "national_pccc_ready.csv"
    pccc_df.to_csv(pccc_file, index=False)
    print(f"\n[+] Exported PCCC-Ready Dataset -> '{pccc_file}'")

    # Metadata map
    meta_cols = [col_id, "class_label", "_dns", "_dur", "_volt", "n_components"]
    for m in [col_class, col_cause, col_zone]:
        if m and m in grouped.columns:
            meta_cols.append(m)

    meta_df = grouped[meta_cols].copy()
    meta_df.index.name = "pccc_index"
    meta_file = out_dir / "national_metadata_lookup.csv"
    meta_df.to_csv(meta_file)
    print(f"[+] Exported Metadata Lookup Map -> '{meta_file}'")
    print("=" * 75)


if __name__ == "__main__":
    run_national_pipeline()