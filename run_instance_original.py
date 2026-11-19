# © 2024, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

# %% Import packages
import json
import sys
import pandas as pd
import numpy as np
from src.pccc import pccc
from sklearn.metrics import adjusted_rand_score

# %% Select data and constraint set
dataset = 'data\\processed data\\COL1\\appendicitis_data.csv'
constraint_set = 'data\\processed data\\COL1\\constraint sets\\appendicitis_constraints_0.05_contradictory.json'

# %% Load dataset
df = pd.read_csv(dataset)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
n_clusters = len(np.unique(y))

# %% Load constraints
with open(constraint_set, 'r') as f:
    constraints = json.load(f)

ml = constraints['ml']
sml = constraints['sml']
cl = constraints['cl']
scl = constraints['scl']
sml_weights = constraints['sml_proba'] or None
scl_weights = constraints['scl_proba'] or None

# %% Apply PCCC algorithm
output = pccc(
    X,
    n_clusters,
    ml=ml,
    cl=cl,
    sml=sml,
    scl=scl,
    sml_weights=sml_weights,
    scl_weights=scl_weights,
    cluster_repositioning='none',
    dynamic_n_neighbors='none',
    random_state=24,
    log_flag=True,
)

# %% Evaluate assignment (with graceful infeasibility check)
if not output or 'labels' not in output or output['labels'] is None:
    print('\n' + '=' * 70)
    print('[!] SOLVER STATUS: INFEASIBLE (No valid cluster assignment exists)')
    print('    Contradictory pairwise constraints prevented a feasible partition.')
    print('    Program exited gracefully without KeyError.')
    print('=' * 70)
    sys.exit(0)

print('ARI: ', adjusted_rand_score(y, output['labels']))