#!/usr/bin/env python3
"""
Generate all valid parameter combinations for test_inference_hypers.py
and write them to a CSV file for use with a SLURM array job.
"""
import csv
import itertools
from tests.test_inference_hypers import (
    HIDDEN_IRREP_COMBOS, EDGE_IRREP_COMBOS, SIZES, DTYPES, ENABLE_CUEQS,
    NUM_INTERACTIONS_LIST, MAX_ELLS, CORRELATIONS, INTERACTION_CLS_LIST
)

rows = []
for max_ell, hidden_irreps, edge_irreps in itertools.product(
    MAX_ELLS, HIDDEN_IRREP_COMBOS, EDGE_IRREP_COMBOS
):
    rows.append({
        "max_ell": max_ell,
        "hidden_irreps": hidden_irreps,
        "edge_irreps": edge_irreps,
    })

with open("benchmarks/inference_hypers_param_grid.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} parameter combinations to inference_hypers_param_grid.csv")
