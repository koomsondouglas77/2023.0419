"""
generate_generation_noisy_constraints.py
=============================================================================
Reproduces Appendix F of Baumann & Hochbaum (2024) for Dataset 1 (Generation).
Generates 24 noisy, confidence-weighted constraint sets across:
  - Percentages (p_c): 0.05, 0.10, 0.15, 0.20
  - Noise Lower Bounds (l): 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Naming format matching the paper benchmark:
  generation_noisy_constraints_{l:.2f}_{str(pc)}.json
=============================================================================
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def generate_noisy_constraints_suite(
    dataset_csv="generation_pccc_ready.csv",
    output_dir="my work/DATASET 1/paper_noisy_constraints",
    percentages=[0.05, 0.10, 0.15, 0.20],
    l_values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    seed=24,
):
    print("=" * 80)
    print("GENERATING NOISY SOFT CONSTRAINTS (APPENDIX F - BAUMANN & HOCHBAUM 2024)")
    print("=" * 80)

    # 1. Locate dataset
    data_path = Path(dataset_csv)
    if not data_path.exists():
        candidates = list(Path(".").rglob("*generation_pccc_ready*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate '{dataset_csv}'.")
        data_path = candidates[0]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading dataset: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    # Ground truth labels are in the final column
    y = df.iloc[:, -1].values
    n = len(y)
    n_classes = len(np.unique(y))

    print(f"[*] Total Objects (n): {n:,}")
    print(f"[*] Total Classes (k): {n_classes}")
    print(f"[*] Target Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[*] Target Directory:    {out_path.resolve()}\n")

    # Generate all candidate pairs (i, j) with i < j
    print("[*] Indexing all candidate pairwise pairs...")
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    total_possible_pairs = len(all_pairs)
    print(f"[*] Total pair pool size: {total_possible_pairs:,}")

    # Set master random seed for reproducibility
    rng = np.random.default_rng(seed)

    summary_records = []

    # 2. Iterate through each percentage level
    for pc in percentages:
        # Sizing formula from Appendix C & F
        m = math.ceil(pc * n)
        n_constraints = m * (m - 1) // 2
        pc_str = str(pc)  # Formats as '0.05', '0.1', '0.15', '0.2'

        print("-" * 80)
        print(f"[*] PROCESSING PERCENTAGE p_c = {pc:.2f} ({int(pc*100)}% CS)")
        print(f"    - Subgraph size m = ceil({pc} * {n}) = {m}")
        print(f"    - Number of constraints = {n_constraints:,}")
        print("-" * 80)

        # Draw the fixed set of pairs for this p_c (consistent across l levels)
        selected_pair_indices = rng.choice(
            total_possible_pairs, size=n_constraints, replace=False
        )
        selected_pairs = [all_pairs[idx] for idx in selected_pair_indices]

        # 3. Iterate through each noise lower bound l
        for l_val in l_values:
            l_str = f"{l_val:.2f}"
            json_filename = (
                out_path / f"generation_noisy_constraints_{l_str}_{pc_str}.json"
            )

            sml = []
            scl = []
            sml_proba = []
            scl_proba = []

            n_sml_wrong = 0
            n_scl_wrong = 0

            # Draw per-constraint noise and confidence
            for i, j in selected_pairs:
                y_i = y[i]
                y_j = y[j]
                is_ground_truth_same = bool(y_i == y_j)

                # Step A: Sample probability of correctness o_ij
                if math.isclose(l_val, 1.0):
                    o_ij = 1.0
                else:
                    o_ij = float(rng.uniform(l_val, 1.0))

                # Step B: Sample outcome test r_ij
                r_ij = float(rng.uniform(0.0, 1.0))

                # Step C: Determine constraint type (correct vs wrong)
                is_correct = r_ij <= o_ij

                if is_ground_truth_same:
                    if is_correct:
                        constraint_type = "sml"
                    else:
                        constraint_type = "scl"  # Inverted by noise
                        n_scl_wrong += 1
                else:
                    if is_correct:
                        constraint_type = "scl"
                    else:
                        constraint_type = "sml"  # Inverted by noise
                        n_sml_wrong += 1

                # Step D: Confidence value w_ij = 2 * (o_ij - 0.5)
                w_ij = float(2.0 * (o_ij - 0.5))
                w_ij = max(0.0, min(1.0, w_ij))  # Bound strictly to [0, 1]

                # Append to corresponding structure
                if constraint_type == "sml":
                    sml.append([int(i), int(j)])
                    sml_proba.append(w_ij)
                else:
                    scl.append([int(i), int(j)])
                    scl_proba.append(w_ij)

            # Build payload
            payload = {
                "ml": [],
                "cl": [],
                "sml": sml,
                "scl": scl,
                "sml_proba": sml_proba,
                "scl_proba": scl_proba,
            }

            with open(json_filename, "w") as f_out:
                json.dump(payload, f_out)

            print(
                f"  [+] Saved: {json_filename.name:<48} | "
                f"SML: {len(sml):5d} (Wrong: {n_sml_wrong:3d}) | "
                f"SCL: {len(scl):5d} (Wrong: {n_scl_wrong:3d})"
            )

            summary_records.append({
                "Filename": json_filename.name,
                "p_c": pc,
                "lower_bound_l": l_val,
                "Total_Constraints": n_constraints,
                "SML_Count": len(sml),
                "SML_Violated_In_Ground_Truth": n_sml_wrong,
                "SCL_Count": len(scl),
                "SCL_Violated_In_Ground_Truth": n_scl_wrong,
            })

    # Export audit summary
    summary_df = pd.DataFrame(summary_records)
    summary_csv = out_path / "generation_noisy_constraints_manifest.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n" + "=" * 80)
    print(f"[+] All 24 noisy constraint files generated successfully!")
    print(f"[+] Manifest summary saved -> '{summary_csv.resolve()}'")
    print("=" * 80)


if __name__ == "__main__":
    generate_noisy_constraints_suite()