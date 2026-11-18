"""
build_24807_transmission_events.py
=============================================================================
Reconstructs the 24,807 consolidated transmission outage events from ENTSO-E
granular interval logs by applying the paper's 12-hour temporal clustering logic
(Gjorgiev et al., 2025, Section 2.3.2, Figures 1 & 2).

Author: Douglas Kojo Kumi Koomson (RRE Lab, ETH Zurich / Ashesi University)
=============================================================================
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd


def generate_24807_events(
    input_csv="00_Transmission_Outage_statistics_preprocessed_ENTSO_E_data.csv",
    output_csv="transmission_24807_events.csv",
    separation_threshold_hours=12.0
):
    # 1. Locate the input raw log file
    path = Path(input_csv)
    if not path.exists():
        candidates = list(Path(".").rglob("*Transmission*preprocessed*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate '{input_csv}'.")
        path = candidates[0]

    print(f"[*] Loading raw transmission interval logs from: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Standardize column headers to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify primary asset identifier
    id_col = None
    for cand in ["power_resource_eic", "powerresourceeic", "resource_name", "asset_name", "unit_name", "mrid"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = df.columns[0]
        print(f"[!] Fallback: using '{id_col}' as asset identifier.")

    # Identify timestamp columns
    start_col = None
    end_col = None
    for c in df.columns:
        if "start" in c and ("date" in c or "time" in c or "ts" in c or "outage" in c):
            start_col = c
        if "end" in c and ("date" in c or "time" in c or "ts" in c or "outage" in c):
            end_col = c

    # Fallback to general time/date columns if not found
    if not start_col or not end_col:
        time_cols = [c for c in df.columns if "time" in c or "date" in c or "ts" in c]
        if len(time_cols) >= 2:
            start_col, end_col = time_cols[0], time_cols[1]
        else:
            raise KeyError(f"Could not identify start and end timestamp columns. Available: {list(df.columns)}")

    print(f"    - Asset identifier: '{id_col}'")
    print(f"    - Start timestamp:  '{start_col}'")
    print(f"    - End timestamp:    '{end_col}'")
    print(f"    - Raw log rows:     {len(df):,}")

    # Parse timestamps
    df["start_dt"] = pd.to_datetime(df[start_col], errors="coerce")
    df["end_dt"] = pd.to_datetime(df[end_col], errors="coerce")

    # Drop invalid rows without timestamps
    df = df.dropna(subset=["start_dt", "end_dt", id_col]).copy()
    
    # Sort strictly by asset and start timestamp
    df = df.sort_values(by=[id_col, "start_dt", "end_dt"]).reset_index(drop=True)

    print(f"[*] Applying temporal clustering (Separation threshold = {separation_threshold_hours} h)...")

    # 2. Vectorized / sequential event consolidation per asset
    clustered_events = []
    threshold_delta = pd.Timedelta(hours=separation_threshold_hours)

    # Optional metadata columns to preserve
    meta_cols = [c for c in ["mapcode", "production_type", "in_area", "out_area", "type", "outage_type"] if c in df.columns]
    
    # Check for delta NTC
    dntc_col = [c for c in df.columns if "delta" in c and "ntc" in c]
    dntc_col = dntc_col[0] if dntc_col else None

    # Group by unique physical asset
    for asset_val, group in df.groupby(id_col, sort=False):
        group = group.reset_index(drop=True)

        current_event_start = group.at[0, "start_dt"]
        current_event_end = group.at[0, "end_dt"]
        
        # Track event characteristics
        event_dntc_vals = [group.at[0, dntc_col]] if dntc_col and pd.notnull(group.at[0, dntc_col]) else []
        event_types = [group.at[0, "type"]] if "type" in group.columns else []

        # Iterate over records for this asset
        for i in range(1, len(group)):
            next_start = group.at[i, "start_dt"]
            next_end = group.at[i, "end_dt"]

            # Overlap check OR gap <= threshold check
            # Overlap: next_start <= current_event_end
            # Gap <= 12h: (next_start - current_event_end) <= threshold_delta
            if next_start <= (current_event_end + threshold_delta):
                # Extend the ongoing physical event
                current_event_end = max(current_event_end, next_end)
                if dntc_col and pd.notnull(group.at[i, dntc_col]):
                    event_dntc_vals.append(group.at[i, dntc_col])
                if "type" in group.columns:
                    event_types.append(group.at[i, "type"])
            else:
                # Close out the finished event
                duration_hrs = (current_event_end - current_event_start).total_seconds() / 3600.0
                
                # Classification rule (Figure 4 in paper): Forced takes precedence if mixed
                classified_type = "Forced" if any("forced" in str(t).lower() for t in event_types) else "Planned"
                
                clustered_events.append({
                    "asset_id": asset_val,
                    "event_start": current_event_start,
                    "event_end": current_event_end,
                    "duration_hours": round(duration_hrs, 2),
                    "outage_type": classified_type,
                    "mean_delta_ntc": np.mean(event_dntc_vals) if event_dntc_vals else 0.0,
                    **{m: group.at[0, m] for m in meta_cols if m != "type"}
                })

                # Start new event
                current_event_start = next_start
                current_event_end = next_end
                event_dntc_vals = [group.at[i, dntc_col]] if dntc_col and pd.notnull(group.at[i, dntc_col]) else []
                event_types = [group.at[i, "type"]] if "type" in group.columns else []

        # Close out the last event for the asset
        duration_hrs = (current_event_end - current_event_start).total_seconds() / 3600.0
        classified_type = "Forced" if any("forced" in str(t).lower() for t in event_types) else "Planned"
        
        clustered_events.append({
            "asset_id": asset_val,
            "event_start": current_event_start,
            "event_end": current_event_end,
            "duration_hours": round(duration_hrs, 2),
            "outage_type": classified_type,
            "mean_delta_ntc": np.mean(event_dntc_vals) if event_dntc_vals else 0.0,
            **{m: group.at[0, m] for m in meta_cols if m != "type"}
        })

    events_df = pd.DataFrame(clustered_events)
    events_df.index.name = "event_id"

    # Export cleanly
    events_df.to_csv(output_csv, index=True)
    print("=" * 75)
    print(f"[+] CONSOLIDATION COMPLETE: Successfully built '{output_csv}'")
    print(f"    - Total Consolidated Physical Events: {len(events_df):,}")
    print(f"    - Target from Gjorgiev et al. (2025):  24,807 events")
    print("=" * 75)
    print("\nPreview of the 24,807 events file:")
    print(events_df.head(5).to_string())
    return events_df


if __name__ == "__main__":
    generate_24807_events()