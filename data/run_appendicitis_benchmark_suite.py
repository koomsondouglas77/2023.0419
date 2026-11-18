"""
run_appendicitis_benchmark_suite.py
=============================================================================
Full reproducible benchmark runner for Appendicitis with 5% constraints.
Executes seeds 24, 26, and 30 on the original dataset (preserving headers & labels).
Exports:
  1. appendicitis_seed_24_assignment.csv
  2. appendicitis_seed_26_assignment.csv
  3. appendicitis_seed_30_assignment.csv
  4. appendicitis_benchmark_summary.csv
=============================================================================
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ---------------------------------------------------------------------------
# Path Configuration: Ensure project root and current working dir are in sys.path
# ---------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SCRIPT_DIR = CURRENT_FILE.parent
PARENT_DIR = SCRIPT_DIR.parent
CWD = Path.cwd()

for p in [str(CWD), str(PARENT_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Dynamic import: handles both 'from src.pccc' and 'from pccc'
try:
    from src.pccc import pccc
except ModuleNotFoundError:
    try:
        from pccc import pccc
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not locate pccc module in {sys.path[:3]}. "
            f"Please ensure pccc.py is located in the root or in a 'src' folder."
        ) from e


def compute_inertia(X_mat: np.ndarray, labels: np.ndarray) -> float:
    """Computes standard Within-Cluster Sum of Squares (Inertia/WCSS)."""
    wcss = 0.0
    for l in np.unique(labels):
        pts = X_mat[labels == l]
        if len(pts) > 0:
            center = pts.mean(axis=0)
            wcss += np.sum((pts - center) ** 2)
    return float(wcss)


def audit_hard_constraints(pairs, labels, expected_same=True):
    """
    Audits satisfaction of pairwise constraints.
    Returns: (satisfied_count, violation_count, satisfaction_percentage)
    """
    if len(pairs) == 0:
        return 0, 0, 100.0
    satisfied = 0
    for u, v in pairs:
        u_idx, v_idx = int(u), int(v)
        if expected_same and (labels[u_idx] == labels[v_idx]):
            satisfied += 1
        elif (not expected_same) and (labels[u_idx] != labels[v_idx]):
            satisfied += 1
    total = len(pairs)
    violations = total - satisfied
    sat_pct = (satisfied / total) * 100.0
    return satisfied, violations, sat_pct


def run_benchmark():
    # 1. Locate original CSV (with header) and constraint JSON
    data_candidates = list(Path('.').rglob('*appendicitis_data.csv'))
    # Exclude files with 'copy' in the name to keep the pristine original
    data_candidates = [p for p in data_candidates if 'copy' not in p.name.lower()]
    if not data_candidates:
        # Fallback check relative to script parent
        data_candidates = list(PARENT_DIR.rglob('*appendicitis_data.csv'))
        data_candidates = [p for p in data_candidates if 'copy' not in p.name.lower()]
    if not data_candidates:
        raise FileNotFoundError("Could not locate original appendicitis_data.csv")
    data_file = data_candidates[0]

    constraint_candidates = list(Path('.').rglob('*appendicitis_constraints_0.05.json'))
    if not constraint_candidates:
        constraint_candidates = list(PARENT_DIR.rglob('*appendicitis_constraints_0.05.json'))
    if not constraint_candidates:
        raise FileNotFoundError("Could not locate appendicitis_constraints_0.05.json")
    constraint_file = constraint_candidates[0]

    print("=" * 80)
    print("APPENDICITIS 5% BENCHMARK SUITE (SEEDS 24, 26, 30)")
    print("=" * 80)
    print(f"[*] Dataset:     {data_file}")
    print(f"[*] Constraints: {constraint_file}")

    # 2. Load dataset preserving headers, decouple X and y
    df = pd.read_csv(data_file)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    n_samples, n_features = X.shape
    n_clusters = len(np.unique(y))
    print(f"[*] Loaded {n_samples} samples, {n_features} features, {n_clusters} clusters.")

    # 3. Load 5% hard constraints
    with open(constraint_file, 'r') as f:
        c_data = json.load(f)
    ml = c_data.get('ml', [])
    cl = c_data.get('cl', [])
    print(f"[*] Hard Constraints: ML = {len(ml)} pairs, CL = {len(cl)} pairs")
    print("=" * 80)

    seeds = [24, 26, 30]
    records = []

    for seed in seeds:
        print(f"\n---> Executing Seed {seed}...")
        t0 = time.perf_counter()
        out = pccc(
            X,
            n_clusters,
            ml=ml,
            cl=cl,
            sml=[],
            scl=[],
            cluster_repositioning='none',
            dynamic_n_neighbors='none',
            random_state=seed,
            log_flag=True
        )
        wall_runtime = time.perf_counter() - t0
        labels = out['labels']

        # Quantitative metrics
        inertia = compute_inertia(X, labels)
        ari = float(adjusted_rand_score(y, labels))
        sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0

        # Constraint audits (Corrected unpacking)
        sat_ml, vio_ml, pct_ml = audit_hard_constraints(ml, labels, expected_same=True)
        sat_cl, vio_cl, pct_cl = audit_hard_constraints(cl, labels, expected_same=False)

        internal_runtime = out.get('log', {}).get('total_runtime', wall_runtime)
        prep_time = out.get('log', {}).get('cpu_preprocessing', 0.0)
        iterations = out.get('log', {}).get('n_iter', len(out.get('log', {}).get('iterations', [])))

        records.append({
            'Seed': seed,
            'Inertia': inertia,
            'ARI': ari,
            'Silhouette': sil,
            'ML_Violations': vio_ml,
            'CL_Violations': vio_cl,
            'All_Hard_Satisfied': (vio_ml == 0 and vio_cl == 0),
            'Iterations': iterations,
            'Wall_Runtime_s': wall_runtime,
            'Internal_Runtime_s': internal_runtime,
            'Preprocessing_s': prep_time
        })

        # Export individual cluster assignment CSV for supervisor review
        assign_df = df.copy()
        assign_df['Predicted_Cluster'] = labels
        out_csv_name = f'appendicitis_seed_{seed}_assignment.csv'
        assign_df.to_csv(out_csv_name, index=True, index_label='Object_Index')
        print(f"     Inertia: {inertia:.4f} | ARI: {ari:.4f} | Violations: {vio_ml + vio_cl} | Runtime: {wall_runtime:.4f}s")
        print(f"     [+] Saved assignment: '{out_csv_name}'")

    # 4. Summary Table & Optimality Gap Calculation
    sum_df = pd.DataFrame(records)
    summary_csv = 'appendicitis_benchmark_summary.csv'
    sum_df.to_csv(summary_csv, index=False)
    print(f"\n[+] Full benchmark summary saved to: '{summary_csv}'")

    optimal_inertia = 491.80  # Paper Table 6 (PC-SOS-SDP solver)
    min_inertia = float(sum_df['Inertia'].min())
    mean_inertia = float(sum_df['Inertia'].mean())
    opt_gap = ((min_inertia - optimal_inertia) / optimal_inertia) * 100.0
    mean_gap = ((mean_inertia - optimal_inertia) / optimal_inertia) * 100.0

    print("\n" + "=" * 80)
    print("FINAL TABLE 6 BENCHMARK COMPARISON:")
    print("=" * 80)
    print(f"Paper Optimal Inertia (PC-SOS-SDP): {optimal_inertia:.4f}")
    print(f"Paper PCCC Reported Min Inertia:    491.8000")
    print(f"Our Minimum Inertia (across 3 runs): {min_inertia:.4f}")
    print(f"Our Mean Inertia (across 3 runs):    {mean_inertia:.4f}")
    print(f"Optimality Gap (Min vs Optimum):     {opt_gap:.4f}%")
    print(f"Mean Optimality Gap:                 {mean_gap:.4f}%")
    print("=" * 80)


if __name__ == '__main__':
    run_benchmark()