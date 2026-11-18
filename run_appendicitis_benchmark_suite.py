"""run_appendicitis_benchmark_suite.py

=============================================================================
Full reproducible benchmark runner for Appendicitis with 5% constraints.
Executes seeds 24, 26, and 30 on the original dataset (keeping headers & labels).
Exports:
  1. appendicitis_seed_24_assignment.csv
  2. appendicitis_seed_26_assignment.csv
  3. appendicitis_seed_30_assignment.csv
  4. appendicitis_benchmark_summary.csv
=============================================================================
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from src.pccc import pccc


def compute_inertia(X_mat, labels):
  """Computes standard Within-Cluster Sum of Squares (Inertia)."""
  wcss = 0.0
  for l in np.unique(labels):
    pts = X_mat[labels == l]
    if len(pts) > 0:
      center = pts.mean(axis=0)
      wcss += np.sum((pts - center) ** 2)
  return float(wcss)


def audit_hard_constraints(pairs, labels, expected_same=True):
  """Audits 100% satisfaction of pairwise links."""
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
  return satisfied, total - satisfied, (satisfied / total) * 100.0


def run_benchmark():
  # 1. Paths (Use original CSV with header!)
  data_candidates = list(Path('.').rglob('*appendicitis_data.csv'))
  data_candidates = [
      p for p in data_candidates if 'copy' not in p.name.lower()
  ]
  if not data_candidates:
    raise FileNotFoundError('Could not locate original appendicitis_data.csv')
  data_file = data_candidates[0]

  constraint_candidates = list(
      Path('.').rglob('*appendicitis_constraints_0.05.json')
  )
  if not constraint_candidates:
    raise FileNotFoundError(
        'Could not locate appendicitis_constraints_0.05.json'
    )
  constraint_file = constraint_candidates[0]

  print('=' * 80)
  print('APPENDICITIS 5% BENCHMARK SUITE (SEEDS 24, 26, 30)')
  print('=' * 80)
  print(f'[*] Dataset:     {data_file}')
  print(f'[*] Constraints: {constraint_file}')

  # 2. Load dataset preserving headers, decouple X and y
  df = pd.read_csv(data_file)
  X = df.iloc[:, :-1].values.astype(float)
  y = df.iloc[:, -1].values
  n_samples, n_features = X.shape
  n_clusters = len(np.unique(y))
  print(
      f'[*] Loaded {n_samples} samples, {n_features} features,'
      f' {n_clusters} clusters.'
  )

  # 3. Load 5% hard constraints
  with open(constraint_file) as f:
    c_data = json.load(f)
  ml = c_data.get('ml', [])
  cl = c_data.get('cl', [])
  print(f'[*] Hard Constraints: ML = {len(ml)} pairs, CL = {len(cl)} pairs')
  print('=' * 80)

  seeds = [24, 26, 30]
  records = []

  for seed in seeds:
    print(f'\n---> Executing Seed {seed}...')
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
        log_flag=True,
    )
    wall_runtime = time.perf_counter() - t0
    labels = out['labels']

    # Metrics
    inertia = compute_inertia(X, labels)
    ari = float(adjusted_rand_score(y, labels))
    sil = (
        float(silhouette_score(X, labels))
        if len(np.unique(labels)) > 1
        else -1.0
    )

    sat_ml, vio_ml, pct_ml = audit_hard_constraints(ml, labels, True)
    sat_cl, vio_cl, pct_cl = audit_constraints_cl = audit_hard_constraints(
        cl, labels, False
    )

    internal_runtime = out.get('log', {}).get('total_runtime', wall_runtime)
    prep_time = out.get('log', {}).get('cpu_preprocessing', 0.0)
    iterations = out.get('log', {}).get('n_iter', len(out['log']['iterations']))

    records.append({
        'Seed': seed,
        'Inertia': inertia,
        'ARI': ari,
        'Silhouette': sil,
        'ML_Violations': vio_ml,
        'CL_Violations': pct_cl[1],
        'All_Hard_Satisfied': (vio_ml == 0 and pct_cl[1] == 0),
        'Iterations': iterations,
        'Wall_Runtime_s': wall_runtime,
        'Internal_Runtime_s': internal_runtime,
        'Preprocessing_s': prep_time,
    })

    # Export individual cluster assignment file
    assign_df = df.copy()
    assign_df['Predicted_Cluster'] = labels
    assign_df.to_csv(
        f'appendicitis_seed_{seed}_assignment.csv',
        index=True,
        index_label='Object_Index',
    )
    print(
        f'     Inertia: {inertia:.4f} | ARI: {ari:.4f} | Violations: {vio_ml + pct_cl[1]} | Runtime: {wall_runtime:.4f}s'
    )
    print(
        f'     Saved: appendicitis_seed_{seed}_assignment.csv'
    )

  # 4. Summary Table & Optimality Gap Calculation
  sum_df = pd.DataFrame(records)
  sum_df.to_csv('appendicitis_benchmark_summary.csv', index=False)

  optimal_inertia = 491.80  # Paper Table 6 (PC-SOS-SDP solver)
  min_inertia = sum_df['Inertia'].min()
  mean_inertia = sum_df['Inertia'].mean()
  opt_gap = ((min_inertia - optimal_inertia) / optimal_inertia) * 100.0
  mean_gap = ((mean_inertia - optimal_inertia) / optimal_inertia) * 100.0

  print('\n' + '=' * 80)
  print('FINAL TABLE 6 COMPARISON:')
  print('=' * 80)
  print(f'Paper Optimal Inertia (PC-SOS-SDP): {optimal_inertia:.4f}')
  print(f'Paper PCCC Min Inertia:              491.8000')
  print(f'Our Minimum Inertia (across 3 runs): {min_inertia:.4f}')
  print(f'Our Mean Inertia (across 3 runs):    {mean_inertia:.4f}')
  print(f'Optimality Gap:                      {opt_gap:.4f}%')
  print(f'Mean Optimality Gap:                 {mean_gap:.4f}%')
  print('=' * 80)


if __name__ == '__main__':
  run_benchmark()