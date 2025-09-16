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
for size, dtype, enable_cueq, num_interactions, max_ell, hidden_irreps, edge_irreps, correlation, interaction_cls, interaction_cls_first in itertools.product(
    SIZES, DTYPES, ENABLE_CUEQS, NUM_INTERACTIONS_LIST, MAX_ELLS, HIDDEN_IRREP_COMBOS, EDGE_IRREP_COMBOS, CORRELATIONS, INTERACTION_CLS_LIST, INTERACTION_CLS_LIST
):
    # Apply skip logic from test_inference_hypers.py
    if (num_interactions > 1) and ((max_ell > 3) or correlation > 5):
        continue
    rows.append({
        "size": size,
        "dtype": dtype,
        "enable_cueq": int(enable_cueq),
        "num_interactions": num_interactions,
        "max_ell": max_ell,
        "hidden_irreps": hidden_irreps,
        "edge_irreps": edge_irreps,
        "correlation": correlation,
        "interaction_cls": interaction_cls,
        "interaction_cls_first": interaction_cls_first,
    })

with open("benchmarks/inference_hypers_param_grid.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} parameter combinations to inference_hypers_param_grid.csv")
