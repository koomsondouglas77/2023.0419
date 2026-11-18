"""
rebuild_sociotechnical_features_v3.py
=============================================================================
Constructs the complete 9-dimensional Sociotechnical Feature Space (R^9):
- Drops N_failures to avoid collinearity with MTTR.
- Aggregates MTTR (mean DurationMin) and total MWh_lost from 01_Failure_Events.xlsx.
- Retains all 3 individual ERA5 weather fields (t2m, i10fg, ro).
- Incorporates CVI, IVI, HDI, and FI from 05_All_Indices.xlsx.
- Normalizes features to [0, 1] using MinMaxScaler.
- Partitions regions into 4 balanced classes via a 2x2 median split.
- Exports sociotechnical_pccc_ready.csv and sociotechnical_metadata_lookup.csv.
=============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def std_region(name):
    """Standardizes Italian region naming conventions across files."""
    name = str(name).strip()
    if 'aosta' in name.lower():
        return "Valle d’Aosta"
    if 'trentino' in name.lower():
        return "Trentino-Alto Adige"
    if 'friuli' in name.lower():
        return "Friuli-Venezia Giulia"
    if 'emilia' in name.lower():
        return "Emilia-Romagna"
    return name


def build_9d_sociotechnical_data():
    print("=" * 80)
    print("BUILDING COMPLETE 9-DIMENSIONAL SOCIOTECHNICAL FEATURE MATRIX (R^9)")
    print("=" * 80)

    # 1. Load and aggregate failure data
    print("[*] Processing 01_Failure_Events.xlsx...")
    df_fail = pd.read_excel("my work\\DATASET 3\\01_Failure_Events.xlsx")
    df_fail['std_region'] = df_fail['Region'].apply(std_region)

    fail_agg = df_fail.groupby('std_region').agg(
        mean_duration_min=('DurationMin', 'mean'),  # MTTR
        total_mwh_lost=('MWh_lost', 'sum')          # Energy unsupplied
    ).reset_index()

    # 2. Load and aggregate weather data (all 3 variables)
    print("[*] Processing 02_Weather_Averages_NUTS2.xlsx...")
    df_weath = pd.read_excel("my work\\DATASET 3\\02_Weather_Averages_NUTS2.xlsx")
    df_weath['std_region'] = df_weath['zone'].apply(std_region)

    weath_agg = df_weath.groupby('std_region').agg(
        mean_t2m=('t2m', 'mean'),      # Temperature
        mean_i10fg=('i10fg', 'mean'),  # Wind gust
        mean_ro=('ro', 'mean')         # Runoff
    ).reset_index()

    # 3. Load consolidated socio-infrastructure indices (CVI, IVI, HDI, FI)
    print("[*] Processing 05_All_Indices.xlsx...")
    df_all = pd.read_excel("my work\\DATASET 3\\05_All_Indices.xlsx")
    df_all['std_region'] = df_all['Region'].apply(std_region)

    # 4. Merge all sources across the 20 NUTS2 regions
    merged = pd.merge(df_all[['std_region', 'NUTS2', 'CVI', 'IVI', 'HDI', 'FI']], fail_agg, on='std_region')
    merged = pd.merge(merged, weath_agg, on='std_region')

    print(f"[+] Aligned all {len(merged)} NUTS2 regions.")

    # 5. Define 4 Ground-Truth Classes (2x2 median split)
    # Technical Grid Impact vs Community Vulnerability (CVI)
    tech_risk = np.log1p(merged['mean_duration_min']) + np.log1p(merged['total_mwh_lost'])
    tau_tech = tech_risk.median()
    tau_cvi = merged['CVI'].median()

    def assign_class(row):
        h_tech = (np.log1p(row['mean_duration_min']) + np.log1p(row['total_mwh_lost'])) >= tau_tech
        h_cvi = row['CVI'] >= tau_cvi
        if not h_tech and not h_cvi:
            return 0  # Class 0: Resilient Urban / Rapid Recovery (Low Impact, Low CVI)
        elif not h_tech and h_cvi:
            return 1  # Class 1: Vulnerable Community / Stable Grid (Low Impact, High CVI)
        elif h_tech and not h_cvi:
            return 2  # Class 2: Climate-Stressed / High Coping (High Impact, Low CVI)
        else:
            return 3  # Class 3: Critical Multi-Hazard Hotspots (High Impact, High CVI)

    merged['target_class'] = merged.apply(assign_class, axis=1)

    print("\nCLASS DISTRIBUTION:")
    for c_id, count in merged['target_class'].value_counts().sort_index().items():
        print(f"  - Class {c_id}: {count} regions ({(count/len(merged))*100:.1f}%)")

    # 6. Formulate 9-Dimensional Continuous Feature Space
    feat_df = pd.DataFrame()
    feat_df['log_duration_min'] = np.log1p(merged['mean_duration_min'])
    feat_df['log_mwh_lost'] = np.log1p(merged['total_mwh_lost'])
    feat_df['cvi'] = merged['CVI']
    feat_df['ivi'] = merged['IVI']
    feat_df['hdi'] = merged['HDI']
    feat_df['fi'] = merged['FI']
    feat_df['temp_k'] = merged['mean_t2m']
    feat_df['wind_gust'] = merged['mean_i10fg']
    feat_df['runoff'] = merged['mean_ro']

    # Scale to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pccc_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    pccc_df['target_class'] = merged['target_class'].values

    # 7. Export Outputs
    out_dir = Path("my work") / "DATASET 3"
    out_dir.mkdir(parents=True, exist_ok=True)

    pccc_csv = out_dir / "sociotechnical_pccc_ready.csv"
    pccc_df.to_csv(pccc_csv, index=False)
    print(f"\n[+] Exported 9D PCCC dataset -> '{pccc_csv}'")

    meta_cols = [
        'std_region', 'NUTS2', 'target_class',
        'mean_duration_min', 'total_mwh_lost',
        'CVI', 'IVI', 'HDI', 'FI',
        'mean_t2m', 'mean_i10fg', 'mean_ro'
    ]
    meta_df = merged[meta_cols].copy()
    meta_df.index.name = "pccc_index"
    meta_csv = out_dir / "sociotechnical_metadata_lookup.csv"
    meta_df.to_csv(meta_csv)
    print(f"[+] Exported metadata lookup -> '{meta_csv}'")
    print("=" * 80)


if __name__ == "__main__":
    build_9d_sociotechnical_data()