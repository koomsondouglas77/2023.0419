"""
extract_from_raw_logs.py
Aggregates interval outage logs by asset to compute continuous
reliability metrics (P0, P3, P4, FOR, MTTR, Delta_NTC).
"""

from pathlib import Path
import numpy as np
import pandas as pd


def aggregate_raw_transmission_csv(
    csv_path="00_Transmission_Outage_statistics_preprocessed_ENTSO_E_data.csv",
    output_csv="transmission_aggregated_metrics.csv",
):
    path = Path(csv_path)
    if not path.exists():
        matches = list(Path(".").rglob("*Transmission*preprocessed*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Could not locate raw CSV file matching '{csv_path}'."
            )
        path = matches[0]

    print(f"[*] Reading raw preprocessed interval logs from: {path}")

    # Inspect first row to map column names flexibly
    preview = pd.read_csv(path, nrows=2)
    clean_cols = {c: c.strip().lower().replace(" ", "_") for c in preview.columns}

    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=clean_cols)

    # Identify primary asset ID column
    id_col = None
    for cand in [
        "power_resource_eic",
        "powerresourceeic",
        "resource_name",
        "asset_name",
        "unit_name",
        "mrid",
    ]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is None:
        raise KeyError(
            f"Could not find an asset ID column among: {list(df.columns)}"
        )

    print(f"    - Grouping by Asset Identifier: '{id_col}'")
    print(f"    - Total raw intervals: {len(df):,}")

    # Helper function to get numeric series safely
    def num(col_name):
        matches = [c for c in df.columns if col_name in c]
        if matches:
            return pd.to_numeric(df[matches[0]], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    df["_ah"] = num("ah")
    df["_poh"] = num("poh")
    df["_foh"] = num("foh")
    df["_n04"] = num("n_04")
    df["_n40"] = num("n_40")
    df["_dntc"] = num("delta_ntc")

    # Group by asset
    agg_spec = {
        "_ah": "sum",
        "_poh": "sum",
        "_foh": "sum",
        "_n04": "sum",
        "_n40": "sum",
        "_dntc": "mean",
    }

    # Retain metadata attributes if present
    for m in ["mapcode", "production_type", "in_area", "out_area"]:
        if m in df.columns:
            agg_spec[m] = "first"

    grouped = df.groupby(id_col).agg(agg_spec).reset_index()
    grouped = grouped.rename(columns={id_col: "asset_id"})

    # Compute continuous Markov & IEEE Std 762 metrics
    total_hours = grouped["_ah"] + grouped["_poh"] + grouped["_foh"]
    total_hours = total_hours.replace(0, np.nan)

    grouped["P0"] = (grouped["_ah"] / total_hours).fillna(1.0).round(6)
    grouped["P3"] = (grouped["_poh"] / total_hours).fillna(0.0).round(6)
    grouped["P4"] = (grouped["_foh"] / total_hours).fillna(0.0).round(6)

    denom_for = grouped["P0"] + grouped["P4"]
    grouped["FOR"] = np.where(
        denom_for > 0, (grouped["P4"] / denom_for).round(6), 0.0
    )

    grouped["MTTR_hours"] = np.where(
        grouped["_n40"] > 0, (grouped["_foh"] / grouped["_n40"]).round(2), 0.0
    )
    grouped["lambda_forced_rate"] = np.where(
        grouped["_ah"] > 0, (grouped["_n04"] / grouped["_ah"]).round(6), 0.0
    )
    grouped["Delta_NTC_mean_MW"] = grouped["_dntc"].round(2)

    # Final column ordering
    output_cols = ["asset_id"]
    for m in ["mapcode", "production_type", "in_area", "out_area"]:
        if m in grouped.columns:
            output_cols.append(m)

    output_cols += [
        "P0",
        "P3",
        "P4",
        "FOR",
        "MTTR_hours",
        "lambda_forced_rate",
        "Delta_NTC_mean_MW",
    ]
    final_df = grouped[output_cols]

    final_df.to_csv(output_csv, index=False)
    print(f"[+] Successfully aggregated {len(final_df)} unique assets.")
    print(f"[+] Output saved to: '{output_csv}'")
    print("\nPreview of extracted features:")
    print(final_df.head(5).to_string())


if __name__ == "__main__":
    aggregate_raw_transmission_csv()