"""
reproduce_appendicitis_benchmark.py
=============================================================================
Resolves all remaining feedback for the Appendicitis benchmark:
1. Executes Seeds 24, 26, and 30 across 5%, 10%, 15%, and 20% constraint sets
   using the original 106-sample appendicitis_data.csv (with header & labels).
2. Saves separate, uniquely named iteration logs and cluster assignments for
   every (constraint_set, seed) pair—eliminating file overwrites/collisions.
3. Reconciles the Seed 24 iteration objective (ofv_distance = 417.0804) with
   the final full-space Inertia (495.5982) via the contraction residual (78.5178).
4. Runs consistent Soft Must-Link (SML) and Soft Cannot-Link (SCL) confidence
   tests (w = 1.0 vs w = 0.1) with distinct, unambiguous log filenames.
5. Programmatically regenerates the Table 6 comparison table directly from the
   saved log files using:
       Opt. Gap (%) = ((Your Min Inertia - Optimal Inertia) / Optimal Inertia) * 100
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
    """Computes exact Within-Cluster Sum of Squares (Inertia) in original space."""
    wcss = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            wcss += np.sum((pts - centroid) ** 2)
    return float(wcss)


def audit_pairs(pairs, labels, expected_same=True):
    """Counts satisfied and violated pairwise constraints."""
    if not pairs:
        return 0, 0
    sat = sum(
        1 for u, v in pairs
        if (expected_same and labels[int(u)] == labels[int(v)])
        or (not expected_same and labels[int(u)] != labels[int(v)])
    )
    return sat, len(pairs) - sat


def run_benchmark_and_regenerate_tables():
    print("=" * 85)
    print("APPENDICITIS REPRODUCIBLE BENCHMARK & TABLE REGENERATION SUITE")
    print("=" * 85)

    # 1. Setup clean output directories
    base_out = Path("results")
    logs_dir = base_out / "logs"
    assign_dir = base_out / "assignments"
    tables_dir = base_out / "tables"
    for d in [logs_dir, assign_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load original 106-sample dataset (retaining header and label column)
    data_file = Path("data/processed data/COL1/appendicitis_data.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset: {data_file.resolve()}")

    df = pd.read_csv(data_file)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    k = len(np.unique(y))
    print(f"[*] Loaded '{data_file}' -> n={len(X)} samples, d={X.shape[1]} features, k={k}")

    # Paper Table 6 Reference Values (Baumann & Hochbaum 2024)
    paper_table6_ref = {
        "0.05": {"opt_inertia": 491.80, "paper_pccc_inertia": 491.80, "paper_pccc_time": 0.3, "opt_time": 387.0},
        "0.1":  {"opt_inertia": 544.18, "paper_pccc_inertia": 544.18, "paper_pccc_time": 0.3, "opt_time": 116.0},
        "0.15": {"opt_inertia": 612.93, "paper_pccc_inertia": 612.93, "paper_pccc_time": 0.1, "opt_time": 32.0},
        "0.2":  {"opt_inertia": 612.93, "paper_pccc_inertia": 612.93, "paper_pccc_time": 0.1, "opt_time": 28.0},
    }

    # Exact filename tags matching the repository files (0.05, 0.1, 0.15, 0.2)
    percentages = ["0.05", "0.1", "0.15", "0.2"]
    seeds = [24, 26, 30]
    run_records = []

    # ==========================================================================
    # PART 1: MULTI-SEED HARD CONSTRAINT BENCHMARK (SEEDS 24, 26, 30)
    # ==========================================================================
    print("\n" + "-" * 85)
    print("PART 1: RUNNING HARD-CONSTRAINT BENCHMARKS (SEEDS 24, 26, 30)")
    print("-" * 85)

    for pc in percentages:
        cs_file = Path(f"data/processed data/COL1/constraint sets/appendicitis_constraints_{pc}.json")
        if not cs_file.exists():
            raise FileNotFoundError(f"Could not find constraint file: {cs_file.resolve()}")

        with open(cs_file, "r") as f:
            constraints = json.load(f)

        ml = constraints.get("ml", [])
        cl = constraints.get("cl", [])

        for seed in seeds:
            tag = f"appendicitis_cs{pc}_seed{seed}"
            t0 = time.perf_counter()
            out = pccc(
                X,
                k,
                ml=ml,
                cl=cl,
                cluster_repositioning="none",
                dynamic_n_neighbors="none",
                random_state=seed,
                log_flag=True,
            )
            wall_time = time.perf_counter() - t0

            if not out or "labels" not in out:
                raise RuntimeError(f"Unexpected infeasibility on {tag}")

            labels = out["labels"]
            inertia = compute_full_inertia(X, labels)
            ari = float(adjusted_rand_score(y, labels))
            sil = float(silhouette_score(X, labels))
            sat_ml, vio_ml = audit_pairs(ml, labels, expected_same=True)
            sat_cl, vio_cl = audit_pairs(cl, labels, expected_same=False)

            # Save unique assignment CSV
            assign_df = df.copy()
            assign_df["Predicted_Cluster"] = labels
            assign_path = assign_dir / f"assignment_{tag}.csv"
            assign_df.to_csv(assign_path, index=True, index_label="Object_Index")

            # Save unique iteration log CSV
            iters = out.get("log", {}).get("iterations", [])
            iter_df = pd.DataFrame(iters)
            iter_df.insert(0, "iteration", np.arange(1, len(iter_df) + 1))
            iter_df.insert(1, "constraint_set", pc)
            iter_df.insert(2, "random_state", seed)
            iter_df.columns = [c.strip() for c in iter_df.columns]
            iter_df["final_full_inertia"] = inertia
            iter_df["contraction_residual"] = inertia - iter_df["ofv_distance"]
            log_path = logs_dir / f"iter_log_{tag}.csv"
            iter_df.to_csv(log_path, index=False)

            final_ofv_dist = float(iter_df["ofv_distance"].iloc[-1])
            residual = inertia - final_ofv_dist
            internal_rt = float(out.get("log", {}).get("total_runtime", wall_time))

            run_records.append({
                "pc": pc,
                "seed": seed,
                "inertia": inertia,
                "final_ofv_distance": final_ofv_dist,
                "contraction_residual": residual,
                "ari": ari,
                "silhouette": sil,
                "vio_ml": vio_ml,
                "vio_cl": vio_cl,
                "runtime_s": internal_rt,
                "log_file": str(log_path),
                "assignment_file": str(assign_path),
            })

            print(
                f"[+] CS {int(round(float(pc)*100)):2d}% | Seed {seed} | "
                f"Inertia: {inertia:8.4f} | OFV_dist: {final_ofv_dist:8.4f} | "
                f"Residual: {residual:7.4f} | ARI: {ari:.4f} | Vio: {vio_ml + vio_cl}"
            )

    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(logs_dir / "all_seed_runs_master_log.csv", index=False)

    # ==========================================================================
    # PART 2: REGENERATE TABLE 6 COMPARISON DIRECTLY FROM THE LOGS
    # ==========================================================================
    print("\n" + "-" * 85)
    print("PART 2: REGENERATING TABLE 6 COMPARISON FROM LOGS")
    print("-" * 85)

    table6_rows = []
    for pc in percentages:
        sub = df_runs[df_runs["pc"] == pc]
        ref = paper_table6_ref[pc]

        min_inertia = float(sub["inertia"].min())
        mean_inertia = float(sub["inertia"].mean())
        opt_inertia = ref["opt_inertia"]

        # Exact supervisor formula: (your min Inertia - optimal inertia) / optimal inertia * 100
        opt_gap_pct = ((min_inertia - opt_inertia) / opt_inertia) * 100.0
        total_runtime_3seeds = float(sub["runtime_s"].sum())
        mean_ari = float(sub["ari"].mean())
        mean_sil = float(sub["silhouette"].mean())

        table6_rows.append({
            "Dataset": "Appendicitis",
            "Constraint_Set": f"{int(round(float(pc)*100))}%",
            "PC_SOS_SDP_Optimal_Inertia": round(opt_inertia, 2),
            "Paper_PCCC_Min_Inertia": round(ref["paper_pccc_inertia"], 2),
            "Our_PCCC_Min_Inertia": round(min_inertia, 4),
            "Our_PCCC_Mean_Inertia": round(mean_inertia, 4),
            "Opt_Gap_Pct": round(opt_gap_pct, 4),
            "Our_Total_Runtime_3Seeds_s": round(total_runtime_3seeds, 4),
            "Paper_PCCC_Runtime_s": ref["paper_pccc_time"],
            "Our_Mean_ARI": round(mean_ari, 4),
            "Our_Mean_Silhouette": round(mean_sil, 4),
        })

    df_table6 = pd.DataFrame(table6_rows)
    table6_csv = tables_dir / "regenerated_table6_comparison.csv"
    df_table6.to_csv(table6_csv, index=False)
    print(df_table6.to_string(index=False))
    print(f"\n[+] Regenerated Table 6 saved -> '{table6_csv}'")

    # ==========================================================================
    # PART 3: CONSISTENT SOFT CONSTRAINT (SML & SCL) TESTS WITH NAMED LOGS
    # ==========================================================================
    print("\n" + "-" * 85)
    print("PART 3: SOFT CONSTRAINTS (SML & SCL) CONFIDENCE WEIGHT TESTS (SEED 24)")
    print("-" * 85)

    base_cs_file = Path("data/processed data/COL1/constraint sets/appendicitis_constraints_0.05.json")
    with open(base_cs_file, "r") as f:
        base_cs = json.load(f)

    base_ml = list(base_cs["ml"])
    base_cl = list(base_cs["cl"])

    # Define 5 explicit test scenarios with unique, unambiguous log names
    soft_scenarios = [
        {
            "name": "01_baseline_hard_only",
            "ml": base_ml,
            "cl": base_cl,
            "sml": [],
            "scl": [],
            "sml_w": None,
            "scl_w": None,
        },
        {
            "name": "02_sml_pair_0_105_conf_1.0",
            "ml": base_ml,
            "cl": base_cl,
            "sml": [[0, 105]],
            "scl": [],
            "sml_w": [1.0],
            "scl_w": None,
        },
        {
            "name": "03_sml_pair_0_105_conf_0.1",
            "ml": base_ml,
            "cl": base_cl,
            "sml": [[0, 105]],
            "scl": [],
            "sml_w": [0.1],
            "scl_w": None,
        },
        {
            "name": "04_scl_pair_36_84_conf_1.0",
            "ml": [p for p in base_ml if p != [36, 84]],
            "cl": base_cl,
            "sml": [],
            "scl": [[36, 84]],
            "sml_w": None,
            "scl_w": [1.0],
        },
        {
            "name": "05_scl_pair_36_84_conf_0.1",
            "ml": [p for p in base_ml if p != [36, 84]],
            "cl": base_cl,
            "sml": [],
            "scl": [[36, 84]],
            "sml_w": None,
            "scl_w": [0.1],
        },
    ]

    soft_summary = []
    for sc in soft_scenarios:
        out_s = pccc(
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
            random_state=24,
            log_flag=True,
        )

        labels_s = out_s["labels"]
        inertia_s = compute_full_inertia(X, labels_s)
        iters_s = pd.DataFrame(out_s["log"]["iterations"])
        iters_s.insert(0, "iteration", np.arange(1, len(iters_s) + 1))
        iters_s.columns = [c.strip() for c in iters_s.columns]

        scen_log_path = logs_dir / f"iter_log_seed24_{sc['name']}.csv"
        iters_s.to_csv(scen_log_path, index=False)

        last_row = iters_s.iloc[-1]
        _, vio_sml = audit_pairs(sc["sml"], labels_s, expected_same=True)
        _, vio_scl = audit_pairs(sc["scl"], labels_s, expected_same=False)

        soft_summary.append({
            "Scenario": sc["name"],
            "Log_File": scen_log_path.name,
            "SML_Violations": vio_sml,
            "SCL_Violations": vio_scl,
            "Final_Penalty_Rho": round(float(last_row["penalty"]), 4),
            "OFV_Distance": round(float(last_row["ofv_distance"]), 4),
            "OFV_Penalty": round(float(last_row["ofv_penalty"]), 4),
            "OFV_Total": round(float(last_row["ofv_total"]), 4),
            "Full_Space_Inertia": round(inertia_s, 4),
        })

        print(
            f"[+] {sc['name']:<28} | SML_Vio: {vio_sml} | SCL_Vio: {vio_scl} | "
            f"OFV_Dist: {last_row['ofv_distance']:8.4f} | OFV_Pen: {last_row['ofv_penalty']:7.4f} | "
            f"Inertia: {inertia_s:8.4f}"
        )

    df_soft_summary = pd.DataFrame(soft_summary)
    soft_table_path = tables_dir / "soft_constraints_verification_table.csv"
    df_soft_summary.to_csv(soft_table_path, index=False)
    print(f"\n[+] Soft constraint summary saved -> '{soft_table_path}'")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark_and_regenerate_tables()