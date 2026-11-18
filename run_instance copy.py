# © 2024, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods
# Philipp Baumann
# Production Script: PCCC Execution, Inertia Metric, Constraint Audit, and Telemetry CSV Exporters

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from src.pccc import pccc

# ==============================================================================
# 1. Configuration & Path Setup
# ==============================================================================
dataset_path = 'my work\\DATASET 1\\GENERATION\\generation_pccc_ready.csv'
constraint_path = 'my work\\DATASET 1\\GENERATION\\my work\\DATASET 1\\paper_noisy_constraints\\generation_noisy_constraints_0.50_0.1.json'
random_seed = 24

# Enforce clean OS-agnostic path objects
dataset_file = Path(dataset_path)
constraint_file = Path(constraint_path)

if not dataset_file.exists():
    raise FileNotFoundError(f"Could not locate dataset file: {dataset_file.resolve()}")
if not constraint_file.exists():
    raise FileNotFoundError(f"Could not locate constraint file: {constraint_file.resolve()}")

# ==============================================================================
# 2. Data Loading & Feature-Label Decoupling (No Label Leakage)
# ==============================================================================
df = pd.read_csv(dataset_file)

# Separate input features (X) from the ground-truth target vector (y)
X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values
n_clusters = len(np.unique(y))
n_samples, n_features = X.shape

print("=" * 80)
print(f"[*] DATASET LOADED: {dataset_file.name}")
print(f"[*] Objects (n): {n_samples} | Features (d): {n_features} | Clusters (k): {n_clusters}")
print("=" * 80)

# ==============================================================================
# 3. Load Pairwise Constraints
# ==============================================================================
with open(constraint_file, "r") as f:
    constraints = json.load(f)

ml = constraints.get("ml", [])
cl = constraints.get("cl", [])
sml = constraints.get("sml", [])
scl = constraints.get("scl", [])
sml_weights = constraints.get("sml_proba", None)
scl_weights = constraints.get("scl_proba", None)

if not sml_weights:
    sml_weights = None
if not scl_weights:
    scl_weights = None

print("[*] CONSTRAINTS SUMMARY:")
print(f"    - Hard Must-Links (ML):    {len(ml)}")
print(f"    - Hard Cannot-Links (CL):  {len(cl)}")
print(f"    - Soft Must-Links (SML):   {len(sml)}")
print(f"    - Soft Cannot-Links (SCL): {len(scl)}")
print("=" * 80)

# ==============================================================================
# 4. Execute PCCC Optimization Algorithm
# ==============================================================================
print("[*] Launching PCCC solver...")
start_time = time.perf_counter()

output = pccc(
    X,
    n_clusters,
    ml=ml,
    cl=cl,
    sml=sml,
    scl=scl,
    sml_weights=sml_weights,
    scl_weights=scl_weights,
    cluster_repositioning="none",  # 'none' as specified by Table 6 benchmark settings
    dynamic_n_neighbors="none",     # 'none' as specified by Table 6 benchmark settings
    random_state=random_seed,
    log_flag=True
)

elapsed_wall_time = time.perf_counter() - start_time
pred_labels = output["labels"]

# ==============================================================================
# 5. Metric Calculations (Inertia, ARI, Silhouette)
# ==============================================================================
def compute_inertia(X_mat: np.ndarray, labels: np.ndarray) -> float:
    """Calculates Within-Cluster Sum of Squares (Inertia) on original features."""
    wcss = 0.0
    for c in np.unique(labels):
        cluster_members = X_mat[labels == c]
        if len(cluster_members) > 0:
            centroid = cluster_members.mean(axis=0)
            wcss += np.sum((cluster_members - centroid) ** 2)
    return float(wcss)

inertia_val = compute_inertia(X, pred_labels)
ari_val = float(adjusted_rand_score(y, pred_labels))
sil_val = float(silhouette_score(X, pred_labels)) if len(np.unique(pred_labels)) > 1 else -1.0

# Extract PCCC internal runtime
pccc_internal_runtime = output.get("log", {}).get("total_runtime", elapsed_wall_time)

print("\n" + "=" * 80)
print("1. EVALUATION METRICS")
print("=" * 80)
print(f"[*] Adjusted Rand Index (ARI):     {ari_val:.4f}")
print(f"[*] Silhouette Score:             {sil_val:.4f}")
print(f"[*] Inertia (WCSS):               {inertia_val:.4f}")
print(f"[*] Wall-Clock Elapsed Time:      {elapsed_wall_time:.4f} s")
print(f"[*] PCCC Log Internal Runtime:    {pccc_internal_runtime:.4f} s")

# ==============================================================================
# 6. Pairwise Constraint Satisfaction Audit
# ==============================================================================
def audit_constraints(pairs, labels, expected_same=True):
    if len(pairs) == 0:
        return 0, 0, 100.0
    satisfied = 0
    total = len(pairs)
    for p in pairs:
        u, v = int(p[0]), int(p[1])
        if expected_same and (labels[u] == labels[v]):
            satisfied += 1
        elif (not expected_same) and (labels[u] != labels[v]):
            satisfied += 1
    violations = total - satisfied
    sat_pct = (satisfied / total) * 100.0
    return satisfied, violations, sat_pct

sat_ml, vio_ml, pct_ml = audit_constraints(ml, pred_labels, expected_same=True)
sat_cl, vio_cl, pct_cl = audit_constraints(cl, pred_labels, expected_same=False)
sat_sml, vio_sml, pct_sml = audit_constraints(sml, pred_labels, expected_same=True)
sat_scl, vio_scl, pct_scl = audit_constraints(scl, pred_labels, expected_same=False)

print("\n" + "=" * 80)
print("2. PAIRWISE CONSTRAINT SATISFACTION AUDIT")
print("=" * 80)
print(f"[*] Hard Must-Links (ML):   {sat_ml}/{len(ml)} satisfied ({pct_ml:.1f}%) | Violated: {vio_ml}")
print(f"[*] Hard Cannot-Links (CL): {sat_cl}/{len(cl)} satisfied ({pct_cl:.1f}%) | Violated: {vio_cl}")
print(f"[*] Soft Must-Links (SML):  {sat_sml}/{len(sml)} satisfied ({pct_sml:.1f}%) | Violated: {vio_sml}")
print(f"[*] Soft Cannot-Links (SCL):{sat_scl}/{len(scl)} satisfied ({pct_scl:.1f}%) | Violated: {vio_scl}")

# ==============================================================================
# 7. Cluster Inspection Summary Table
# ==============================================================================
cluster_summary = []
for c_id in range(n_clusters):
    members = np.where(pred_labels == c_id)[0]
    count = len(members)
    unique_true, true_counts = np.unique(y[members], return_counts=True)
    true_dist = dict(zip(unique_true, true_counts))

    cluster_summary.append({
        "Cluster": c_id,
        "Size": count,
        "Percentage": f"{(count / n_samples) * 100:.1f}%",
        "Ground_Truth_Breakdown": true_dist
    })

summary_df = pd.DataFrame(cluster_summary)
print("\n" + "=" * 80)
print("3. CLUSTER ASSIGNMENT BREAKDOWN")
print("=" * 80)
print(summary_df.to_string(index=False))

# ==============================================================================
# 8. Export Labeled Dataset and Raw Logs to CSV
# ==============================================================================
# A. Clustered Objects
output_csv = "pccc_clustered_output.csv"
df_out = df.copy()
df_out["Predicted_Cluster"] = pred_labels
df_out.to_csv(output_csv, index=True, index_label="Object_Index")
print(f"\n[+] Clustered objects saved to: '{output_csv}'")

# B. Raw Telemetry Logs (Iteration & Run Summary)
if "log" in output and output["log"]:
    pccc_log = output["log"]

    # Export Iteration-by-Iteration telemetry
    if "iterations" in pccc_log and pccc_log["iterations"]:
        df_iter_log = pd.DataFrame(pccc_log["iterations"])
        df_iter_log.insert(0, "iteration", np.arange(1, len(df_iter_log) + 1))
        df_iter_log.insert(1, "dataset", dataset_file.name)
        df_iter_log.insert(2, "random_state", random_seed)

        iter_log_csv = "pccc_raw_iteration_log.csv"
        df_iter_log.to_csv(iter_log_csv, index=False)
        print(f"[+] Raw solver iteration telemetry exported to: '{iter_log_csv}'")

    # Export High-level Run Summary
    run_summary_record = {
        "dataset": [dataset_file.name],
        "constraint_set": [constraint_file.name],
        "random_state": [random_seed],
        "n_objects": [pccc_log.get("n_objects", n_samples)],
        "n_supernodes": [pccc_log.get("n_representatives", np.nan)],
        "iterations_count": [pccc_log.get("n_iter", len(pccc_log.get("iterations", [])))],
        "total_runtime_s": [pccc_internal_runtime],
        "preprocessing_time_s": [pccc_log.get("cpu_preprocessing", np.nan)],
        "inertia_wcss": [inertia_val],
        "ari": [ari_val],
        "silhouette": [sil_val],
        "ml_satisfied_pct": [pct_ml],
        "cl_satisfied_pct": [pct_cl]
    }
    df_run_summary = pd.DataFrame(run_summary_record)
    run_summary_csv = "pccc_run_summary_log.csv"
    df_run_summary.to_csv(run_summary_csv, index=False)
    print(f"[+] High-level run summary exported to: '{run_summary_csv}'")

print("=" * 80)
print("[+] EXECUTION AND EXPORT COMPLETED SUCCESSFULLY")
print("=" * 80)