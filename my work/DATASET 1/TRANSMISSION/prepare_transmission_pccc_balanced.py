"""
prepare_transmission_pccc_balanced.py
Fixes feature distortion and class imbalance for PCCC transmission clustering.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_balanced_data(
    input_csv="transmission_aggregated_metrics.csv",
    output_pccc_csv="transmission_pccc_ready.csv",
    output_meta_csv="transmission_metadata_lookup.csv",
):
    path = Path(input_csv)
    if not path.exists():
        candidates = list(Path(".").rglob("*transmission_aggregated*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate '{input_csv}'.")
        path = candidates[0]

    print(f"[*] Reading data from: {path}")
    df = pd.read_csv(path)

    # 1. Standardize numeric columns
    for c in ["P3", "P4", "FOR", "MTTR_hours", "Delta_NTC_mean_MW"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 2. Balanced 4-Class Segmentation (2x2 Median Split on active assets)
    # Split on FOR and MTTR to balance the four quadrants
    med_for = df["FOR"].median()
    med_mttr = df["MTTR_hours"].median()

    # If median is 0 due to zero-inflation, use median of non-zero values
    if med_for == 0:
        med_for = (
            df[df["FOR"] > 0]["FOR"].median() if len(df[df["FOR"] > 0]) > 0 else 0.001
        )
    if med_mttr == 0:
        med_mttr = (
            df[df["MTTR_hours"] > 0]["MTTR_hours"].median()
            if len(df[df["MTTR_hours"] > 0]) > 0
            else 10.0
        )

    print(f"[*] Split Thresholds -> FOR: {med_for:.6f} | MTTR: {med_mttr:.2f} h")

    def assign_balanced_class(row):
        h_for = row["FOR"] >= med_for
        h_mttr = row["MTTR_hours"] >= med_mttr
        if not h_for and not h_mttr:
            return 0  # Low FOR, Fast Recovery
        elif not h_for and h_mttr:
            return 1  # Low FOR, Slow Recovery (Maintenance)
        elif h_for and not h_mttr:
            return 2  # High FOR, Fast Recovery (Weather/Transient)
        else:
            return 3  # High FOR, Slow Recovery (Chokepoint)

    df["class_label"] = df.apply(assign_balanced_class, axis=1)

    print("\nBALANCED CLASS DISTRIBUTION:")
    print(df["class_label"].value_counts(normalize=True) * 100)

    # 3. Transform features without destroying simplex bounds
    # Non-collinear 4-dimensional representation:
    # - P3: Planned outage fraction (kept on [0, 1])
    # - P4: Forced outage fraction (kept on [0, 1])
    # - Log-transformed MTTR: dampened outlier effect
    # - Log-transformed Delta_NTC: dampened capacity outlier effect
    df_feat = pd.DataFrame()
    df_feat["P3"] = df["P3"].clip(0.0, 1.0)
    df_feat["P4"] = df["P4"].clip(0.0, 1.0)
    df_feat["log_MTTR"] = np.log1p(df["MTTR_hours"].clip(lower=0))
    df_feat["log_DNTC"] = np.log1p(df["Delta_NTC_mean_MW"].clip(lower=0))

    # MinMax Scale everything uniformly to [0, 1] so no single axis dominates
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_feat.values)

    pccc_df = pd.DataFrame(X_scaled, columns=df_feat.columns)
    pccc_df["target_class"] = df["class_label"].values

    # 4. Save metadata lookup
    id_col = [c for c in df.columns if "id" in c or "eic" in c]
    id_col = id_col[0] if id_col else df.columns[0]
    meta_cols = [id_col, "class_label"]
    for extra in ["mapcode", "mapcode1", "mapcode2", "in_area", "out_area"]:
        if extra in df.columns:
            meta_cols.append(extra)

    df_meta = df[meta_cols].copy()
    df_meta.index.name = "pccc_index"
    df_meta.to_csv(output_meta_csv)

    # 5. Save PCCC ready CSV
    pccc_df.to_csv(output_pccc_csv, index=False)
    print(
        f"\n[+] Saved balanced PCCC dataset -> '{output_pccc_csv}' (Features: {X_scaled.shape[1]}, Objects: {X_scaled.shape[0]})"
    )


if __name__ == "__main__":
    prepare_balanced_data()