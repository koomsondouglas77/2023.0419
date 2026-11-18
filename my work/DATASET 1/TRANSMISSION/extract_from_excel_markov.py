"""
extract_from_excel_markov.py
Extracts precomputed Markov state probabilities and transition rates
directly from the aggregated transmission Excel file.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def extract_from_markov_excel(
    excel_path="05_Transmission_Statistics_powerresourceeic_Markov.xlsx",
    output_csv="transmission_extracted_features.csv",
):
    path = Path(excel_path)
    if not path.exists():
        # Fallback search if placed in a subfolder
        matches = list(Path(".").rglob("*05_Transmission*Markov*.xlsx"))
        if not matches:
            raise FileNotFoundError(
                f"Could not locate '{excel_path}' in the current directory or subfolders."
            )
        path = matches[0]

    print(f"[*] Reading aggregated Markov data from: {path}")
    xls = pd.ExcelFile(path)
    print(f"    - Available sheets: {xls.sheet_names}")

    # Read the primary sheet (or first sheet)
    sheet_name = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(f"    - Raw data shape: {df.shape}")

    # Standardize column headers to lowercase
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Resolve primary identifier column (Power Resource EIC / Name)
    id_col = None
    for cand in [
        "powerresourceeic",
        "power_resource_eic",
        "eic",
        "asset_name",
        "name",
        "mrid",
    ]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is None:
        # Fallback to the very first column if no named ID is matched
        id_col = df.columns[0]
        print(
            f"[!] Warning: Using first column '{id_col}' as primary asset identifier."
        )

    # Build clean output dataframe
    extracted = pd.DataFrame()
    extracted["asset_id"] = df[id_col].astype(str)

    # Extract metadata if available
    for meta in ["mapcode", "productiontype", "in_area", "out_area"]:
        if meta in df.columns:
            extracted[meta] = df[meta]

    # Extract Markov State Probabilities (P0, P3, P4)
    # Checks for P0 / p_0 / p0 variations
    for state, default_val in [
        ("p0", 1.0),
        ("p3", 0.0),
        ("p4", 0.0),
        ("for", 0.0),
    ]:
        matched = [c for c in df.columns if c == state or c.startswith(f"{state}_")]
        if matched:
            extracted[state.upper()] = (
                pd.to_numeric(df[matched[0]], errors="coerce")
                .fillna(default_val)
                .round(6)
            )

    # Extract transition rates (alpha_04, alpha_40, alpha_03, alpha_30) if present
    rate_cols = [c for c in df.columns if "alpha" in c or c.startswith("a_")]
    for rc in rate_cols:
        extracted[rc] = (
            pd.to_numeric(df[rc], errors="coerce").fillna(0.0).round(6)
        )

    # Export cleanly
    extracted.to_csv(output_csv, index=False)
    print(f"[+] Successfully extracted {len(extracted)} transmission assets.")
    print(f"[+] Output saved to: '{output_csv}'")
    print("\nPreview of extracted features:")
    print(extracted.head(5).to_string())


if __name__ == "__main__":
    extract_from_markov_excel()