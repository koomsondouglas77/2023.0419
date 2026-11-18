import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def generate_noise_free_constraints(
    dataset_csv="my work/DATASET 3/sociotechnical_pccc_ready.csv",
    output_dir="my work/DATASET 3/paper_constraints",
    percentages=[0.05, 0.10, 0.15, 0.20],
    seed=24,
):
    print("=" * 75)
    print(
        "GENERATING NOISE-FREE CONSTRAINTS (APPENDIX C / GONZALEZ-ALMAGRO ET AL.)"
    )
    print("=" * 75)

    data_path = Path(dataset_csv)
    if not data_path.exists():
        candidates = list(Path(".").rglob("*sociotechnical_pccc_ready*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate '{dataset_csv}'.")
        data_path = candidates[0]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Ground truth labels are in the final column
    y = df.iloc[:, -1].values
    n = len(y)
    n_classes = len(np.unique(y))

    print(f"[*] Total Objects (n): {n}")
    print(f"[*] Total Classes (k): {n_classes}")
    print(
        f"[*] Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
    )

    # Generate all possible pairs (i, j) with i < j
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))

    # Set reproducible random seed
    rng = np.random.default_rng(seed)

    for pc in percentages:
        # Sizing formula from Appendix C of Baumann & Hochbaum (2024)
        m = math.ceil(pc * n)
        n_constraints = m * (m - 1) // 2

        # Draw pairs uniformly at random without replacement
        if n_constraints > 0:
            selected_indices = rng.choice(
                len(all_pairs), size=n_constraints, replace=False
            )
            selected_pairs = [all_pairs[idx] for idx in selected_indices]
        else:
            selected_pairs = []

        ml = []
        cl = []
        for i, j in selected_pairs:
            if y[i] == y[j]:
                ml.append([int(i), int(j)])
            else:
                cl.append([int(i), int(j)])

        payload = {
            "ml": ml,
            "cl": cl,
            "sml": [],
            "scl": [],
            "sml_proba": [],
            "scl_proba": [],
        }

        tag = f"{int(pc * 100)}%"
        json_filename = out_path / f"constraints_{pc:.2f}.json"
        with open(json_filename, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\n[+] Generated {tag:>3s} CS (p_c = {pc:.2f}):")
        print(
            f"    - Formula: ceil({pc} * {n}) = {m} vertices -> {n_constraints} total pairs"
        )
        print(f"    - Must-Links (ML):   {len(ml)}")
        print(f"    - Cannot-Links (CL): {len(cl)}")
        print(f"    - Output: {json_filename}")

    print("\n" + "=" * 75)
    print("ALL CONSTRAINT SETS SUCCESSFULLY GENERATED")
    print("=" * 75)


if __name__ == "__main__":
    generate_noise_free_constraints()