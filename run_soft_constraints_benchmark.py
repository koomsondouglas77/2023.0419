"""
run_soft_constraints_benchmark.py
=============================================================================
Seed 24 Soft Must-Link (SML) and Soft Cannot-Link (SCL) benchmarks
on appendicitis_data.csv using:
  - Baseline: appendicitis_constraints_0.05.json (Hard Constraints Only)
  - SML File: appendicitis_constraints_0.05_with_sml.json (tested at w=1.0 and w=0.01)
  - SCL File: appendicitis_constraints_0.05_with_scl.json (tested at w=1.0 and w=0.01)

Exports:
  - Uniquely named raw iteration logs in results/soft_benchmark/logs/
  - Uniquely named cluster assignments in results/soft_benchmark/assignments/
  - Final Comparison Table recording both assignment & optimization violations
    in results/soft_benchmark/final_soft_constraints_table.csv
=============================================================================
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from src.pccc import pccc


def compute_full_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes exact Within-Cluster Sum of Squares (Inertia) in original feature space."""
    wcss = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            wcss += np.sum((pts - centroid) ** 2)
    return float(wcss)


def run_soft_benchmark():
    print("=" * 95)
    print("SEED 24 SOFT MUST-LINK (SML) & SOFT CANNOT-LINK (SCL) BENCHMARK SUITE")
    print("=" * 95)

    # 1. Paths to dataset and exact constraint files
    dataset_path = Path("data/processed data/COL1/appendicitis_data.csv")
    cs_dir = Path("data/processed data/COL1/constraint sets")

    f_base = cs_dir / "appendicitis_constraints_0.05.json"
    f_sml = cs_dir / "appendicitis_constraints_0.05_with_sml.json"
    f_scl = cs_dir / "appendicitis_constraints_0.05_with_scl.json"

    for p in [dataset_path, f_base, f_sml, f_scl]:
        if not p.exists():
            raise FileNotFoundError(f"Could not locate required file: {p.resolve()}")

    # 2. Setup output directories
    out_root = Path("results") / "soft_benchmark"
    logs_dir = out_root / "logs"
    assign_dir = out_root / "assignments"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assign_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load original 106-sample dataset
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    k = len(np.unique(y))
    seed = 24

    with open(f_base, "r") as f:
        cs_base = json.load(f)
    with open(f_sml, "r") as f:
        cs_sml = json.load(f)
    with open(f_scl, "r") as f:
        cs_scl = json.load(f)

    # 4. Define the 5 explicit test configurations
    # Evaluates baseline, 100% certainty (w=1.0), and 1% certainty (w=0.01) for SML and SCL
    scenarios = [
        {
            "id": "01_baseline_hard_only",
            "label": "(a) Baseline (Hard Constraints Only)",
            "source_file": f_base.name,
            "ml": cs_base.get("ml", []),
            "cl": cs_base.get("cl", []),
            "sml": [],
            "scl": [],
            "sml_w": None,
            "scl_w": None,
            "weight_desc": "N/A",
        },
        {
            "id": "02_sml_w1.00_100pct",
            "label": "(b) SML 100% Certainty (w = 1.0)",
            "source_file": f_sml.name,
            "ml": cs_sml.get("ml", []),
            "cl": cs_sml.get("cl", []),
            "sml": cs_sml.get("sml", []),
            "scl": cs_sml.get("scl", []),
            "sml_w": [1.0] * len(cs_sml.get("sml", [])) if cs_sml.get("sml") else None,
            "scl_w": cs_sml.get("scl_proba") or None,
            "weight_desc": "1.00 (100%)",
        },
        {
            "id": "03_sml_w0.01_1pct",
            "label": "(c) SML 1% Certainty (w = 0.01)",
            "source_file": f_sml.name,
            "ml": cs_sml.get("ml", []),
            "cl": cs_sml.get("cl", []),
            "sml": cs_sml.get("sml", []),
            "scl": cs_sml.get("scl", []),
            "sml_w": [0.01] * len(cs_sml.get("sml", [])) if cs_sml.get("sml") else None,
            "scl_w": cs_sml.get("scl_proba") or None,
            "weight_desc": "0.01 (1%)",
        },
        {
            "id": "04_scl_w1.00_100pct",
            "label": "(d) SCL 100% Certainty (w = 1.0)",
            "source_file": f_scl.name,
            "ml": cs_scl.get("ml", []),
            "cl": cs_scl.get("cl", []),
            "sml": cs_scl.get("sml", []),
            "scl": cs_scl.get("scl", []),
            "sml_w": cs_scl.get("sml_proba") or None,
            "scl_w": [1.0] * len(cs_scl.get("scl", [])) if cs_scl.get("scl") else None,
            "weight_desc": "1.00 (100%)",
        },
        {
            "id": "05_scl_w0.01_1pct",
            "label": "(e) SCL 1% Certainty (w = 0.01)",
            "source_file": f_scl.name,
            "ml": cs_scl.get("ml", []),
            "cl": cs_scl.get("cl", []),
            "sml": cs_scl.get("sml", []),
            "scl": cs_scl.get("scl", []),
            "sml_w": cs_scl.get("sml_proba") or None,
            "scl_w": [0.01] * len(cs_scl.get("scl", [])) if cs_scl.get("scl") else None,
            "weight_desc": "0.01 (1%)",
        },
    ]

    summary_rows = []

    # 5. Execute all 5 runs and save distinct logs & assignments
    for sc in scenarios:
        t0 = time.perf_counter()
        output = pccc(
            X,
            k,
            ml=sc["ml"],
            cl=sc["cl"],
            sml=sc["sml"],
            scl=sc["scl"],
            sml_weights=sc["sml_w"],
            scl_weights=sc["scl_w"],
            cluster_repositioning="none",
            dynamic_n_neighbors="none",
            random_state=seed,
            log_flag=True,
        )
        wall_time = time.perf_counter() - t0

        if not output or "labels" not in output or output["labels"] is None:
            print(f"[!] Scenario '{sc['label']}' returned INFEASIBLE.")
            continue

        labels = output["labels"]
        full_inertia = compute_full_inertia(X, labels)
        ari = float(adjusted_rand_score(y, labels))
        sil = float(silhouette_score(X, labels))

        # Save unique cluster assignment CSV
        assign_df = df.copy()
        assign_df["Predicted_Cluster"] = labels
        assign_file = assign_dir / f"assignment_seed{seed}_{sc['id']}.csv"
        assign_df.to_csv(assign_file, index=True, index_label="Object_Index")

        # Save unique raw iteration log CSV
        iters = output.get("log", {}).get("iterations", [])
        df_iter = pd.DataFrame(iters)
        df_iter.columns = [c.strip() for c in df_iter.columns]
        df_iter.insert(0, "iteration", np.arange(1, len(df_iter) + 1))
        df_iter.insert(1, "scenario_id", sc["id"])
        df_iter.insert(2, "constraint_file", sc["source_file"])
        df_iter.insert(3, "random_state", seed)
        df_iter["full_space_inertia"] = full_inertia
        df_iter["contraction_residual"] = full_inertia - df_iter["ofv_distance"]

        log_file = logs_dir / f"iter_log_seed{seed}_{sc['id']}.csv"
        df_iter.to_csv(log_file, index=False)

        last = df_iter.iloc[-1]

        # Extract assignment vs optimization violations directly from the log
        sml_vio_assign = int(last.get("n_sml_vio_in_assignment", 0))
        sml_vio_opt = float(last.get("n_sml_vio_in_optimization", 0.0))
        scl_vio_assign = int(last.get("n_scl_vio_in_assignment", 0))
        scl_vio_opt = float(last.get("n_scl_vio_in_optimization", 0.0))

        summary_rows.append({
            "Scenario": sc["label"],
            "Constraint_File": sc["source_file"],
            "Saved_Log_File": log_file.name,
            "ML": len(sc["ml"]),
            "CL": len(sc["cl"]),
            "SML": len(sc["sml"]),
            "SCL": len(sc["scl"]),
            "Certainty_Weight": sc["weight_desc"],
            "Iterations": len(df_iter),
            "SML_Vio_Assignment": sml_vio_assign,
            "SML_Vio_Optimization": round(sml_vio_opt, 4),
            "SCL_Vio_Assignment": scl_vio_assign,
            "SCL_Vio_Optimization": round(scl_vio_opt, 4),
            "Penalty_Rho": round(float(last["penalty"]), 4),
            "OFV_Distance": round(float(last["ofv_distance"]), 4),
            "OFV_Penalty": round(float(last["ofv_penalty"]), 4),
            "OFV_Total": round(float(last["ofv_total"]), 4),
            "Full_Space_Inertia": round(full_inertia, 4),
            "ARI": round(ari, 4),
            "Silhouette": round(sil, 4),
            "Runtime_s": round(float(output.get("log", {}).get("total_runtime", wall_time)), 4),
        })

        print(f"[+] {sc['label']}")
        print(f"    - Log Saved:              {log_file}")
        print(f"    - SML Vio (Assign | Opt): {sml_vio_assign} | {sml_vio_opt}")
        print(f"    - SCL Vio (Assign | Opt): {scl_vio_assign} | {scl_vio_opt}")
        print(f"    - OFV_dist: {last['ofv_distance']:.4f} | OFV_pen: {last['ofv_penalty']:.4f} | OFV_total: {last['ofv_total']:.4f} | Inertia: {full_inertia:.4f}\n")

    # 6. Export Final Comparison Table
    df_final = pd.DataFrame(summary_rows)
    final_table_path = out_root / "final_soft_constraints_table.csv"
    df_final.to_csv(final_table_path, index=False)

    print("=" * 95)
    print("FINAL SOFT CONSTRAINTS COMPARISON TABLE (SEED 24 — GENERATED FROM LOGS)")
    print("=" * 95)
    print(df_final.to_string(index=False))
    print(f"\n[+] Saved final table -> '{final_table_path.resolve()}'")
    print("=" * 95)


if __name__ == "__main__":
    run_soft_benchmark()