"""
populate_old_feature_matrix.py
=============================================================================
Populates the first three columns (log_lambda, log_mttr, log_dns) of the 
old feature matrix (sociotechnical_pccc_ready_Updated.csv) using 
01_Failure_Events.xlsx.

- Keeps the original 20-region order and existing CVI, IVI values.
- Marks uncalculated weather_hazard values as missing (NaN), per James's instruction.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def populate_features():
    print("=" * 80)
    print("POPULATING OLD SOCIOTECHNICAL FEATURE MATRIX")
    print("=" * 80)

    # 1. Locate files
    cwd = Path.cwd()
    def find_file(pattern):
        matches = list(cwd.glob(pattern)) + list(cwd.rglob(pattern))
        return matches[0] if matches else None

    f_events = find_file("*01_Failure_Events*.xlsx")
    f_csv = find_file("*sociotechnical_pccc_ready_Updated*.csv")

    if not f_events or not f_events.exists():
        raise FileNotFoundError("Could not locate '01_Failure_Events.xlsx'.")
    if not f_csv or not f_csv.exists():
        raise FileNotFoundError("Could not locate 'sociotechnical_pccc_ready_Updated.csv'.")

    print(f"[*] Failure events file: {f_events.name}")
    print(f"[*] Target matrix CSV:   {f_csv.name}")

    # 2. Canonical 20-region sequence matching rows 0 to 19
    regions_order = [
        'Abruzzo',
        'Basilicata',
        'Calabria',
        'Campania',
        'Emilia-Romagna',
        'Friuli-Venezia Giulia',
        'Lazio',
        'Liguria',
        'Lombardia',
        'Marche',
        'Molise',
        'Piemonte',
        'Puglia',
        'Sardegna',
        'Sicilia',
        'Toscana',
        'Trentino-Alto Adige',
        'Umbria',
        'Valle d’Aosta/Vallée d’Aoste',
        'Veneto'
    ]

    # 3. Load failure events data
    df_fail = pd.read_excel(f_events)

    def std_region(name):
        name = str(name).strip()
        if 'aosta' in name.lower():
            return 'Valle d’Aosta/Vallée d’Aoste'
        return name

    df_fail['std_region'] = df_fail['Region'].apply(std_region)
    n_years = float(df_fail['Year'].nunique()) if 'Year' in df_fail.columns else 10.0
    print(f"[*] Aggregating {len(df_fail):,} failure records across {n_years:.0f} years (2013-2022)...")

    # 4. Compute metrics per region
    records = []
    for reg in regions_order:
        sub = df_fail[df_fail['std_region'] == reg]
        n_ev = len(sub)

        lam = n_ev / n_years
        mttr = sub['DurationMin'].mean() if n_ev > 0 else 0.0
        dns = (sub['MWh_lost'].sum() / n_years) if n_ev > 0 else 0.0

        records.append({
            'Region': reg,
            'lambda': lam,
            'mttr': mttr,
            'dns': dns,
            'log_lambda': np.log1p(lam),
            'log_mttr': np.log1p(mttr),
            'log_dns': np.log1p(dns)
        })

    df_metrics = pd.DataFrame(records)

    # 5. Scale to [0, 1] using MinMaxScaler (matching IVI and CVI)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_metrics[['log_lambda', 'log_mttr', 'log_dns']])

    # 6. Read and update the target CSV
    df_old = pd.read_csv(f_csv)
    df_old.columns = [c.strip() for c in df_old.columns]

    # Populate first three columns
    df_old['log_lambda'] = np.round(scaled_features[:, 0], 6)
    df_old['log_mttr'] = np.round(scaled_features[:, 1], 6)
    df_old['log_dns'] = np.round(scaled_features[:, 2], 6)

    # Per James's rule: uncalculated values marked as missing (NaN), not 0
    df_old['weather_hazard'] = np.nan

    # 7. Save output
    output_path = f_csv
    df_old.to_csv(output_path, index=False)
    print(f"\n[+] Successfully updated: '{output_path.resolve()}'")
    print("\nFIRST 5 UPDATED ROWS:")
    print(df_old.head(5))
    print("=" * 80)


if __name__ == "__main__":
    populate_features()