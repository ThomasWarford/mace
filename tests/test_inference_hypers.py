"""
Benchmark inference performance of the MACE medium model on GPU.

Run benchmarks:
    pytest tests/test_benchmark.py --benchmark-save=<some name>

To also include torch.compile benchmarks:
    MACE_FULL_BENCH=1 pytest tests/test_benchmark.py --benchmark-save=<some name>

Convert results to CSV:
    python tests/test_benchmark.py > results.csv
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest
import torch
from ase import build

from mace import data as mace_data
from mace.calculators.foundations_models import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace import modules

from e3nn import o3

SIZES = [7]
DTYPES = ["float32", "float64"]
ENABLE_CUEQS = [True]
NUM_INTERACTIONS_LIST = [1, 2]
MAX_ELLS = [2, 3, 4]
CORRELATIONS = [3, 4, 5, 6]
INTERACTION_CLS_LIST = ["RealAgnosticResidualInteractionBlock", "RealAgnosticResidualNonLinearInteractionBlock"]

HIDDEN_IRREP_COMBOS = []
for irreps in [32, 64, 128, 258]:
        HIDDEN_IRREP_COMBOS.append(f"{irreps}x0e+{irreps}x1o")

EDGE_IRREP_COMBOS = []
for scalar_irreps in [64, 128, 258]:
    for vector_irreps in [16, 32, 64, 128]:
        EDGE_IRREP_COMBOS.append(f"{scalar_irreps}x0e+{vector_irreps}x1o")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.benchmark(warmup=True, warmup_iterations=4, min_rounds=8)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("enable_cueq", ENABLE_CUEQS)

@pytest.mark.parametrize("num_interactions", NUM_INTERACTIONS_LIST)
@pytest.mark.parametrize("max_ell", MAX_ELLS) # 4 only for single layer
@pytest.mark.parametrize("hidden_irreps", HIDDEN_IRREP_COMBOS) # node irreps
@pytest.mark.parametrize("edge_irreps", EDGE_IRREP_COMBOS)
@pytest.mark.parametrize("correlation", CORRELATIONS) # 6 only for single layer
@pytest.mark.parametrize("interaction_cls", INTERACTION_CLS_LIST)
@pytest.mark.parametrize("interaction_cls_first", INTERACTION_CLS_LIST)
def test_inference(
    benchmark, 
    size: int, 
    dtype: str, 
    enable_cueq: bool,

    num_interactions: int,
    max_ell: int,
    hidden_irreps: o3.Irreps,
    edge_irreps: o3.Irreps,
    correlation: int,
    interaction_cls: str,
    interaction_cls_first: str,

    device: str = "cuda",
        ):
    
    if (num_interactions > 1) and ((max_ell > 3) or correlation > 5):
        pytest.skip(f"Skipping; num_interactions:{num_interactions}, max_ell:{max_ell}, correlation:{correlation}")

    with torch_tools.default_dtype(dtype):
        model = create_mace(
            device,
            enable_cueq,

            num_interactions=num_interactions,
            max_ell=max_ell,
            hidden_irreps=hidden_irreps,
            edge_irreps=edge_irreps,
            correlation=correlation,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            )
        batch = create_batch(size, model, device)
        log_bench_info(benchmark, dtype, enable_cueq, 
                       num_interactions=num_interactions,
                       max_ell=max_ell,
                       hidden_irreps=hidden_irreps,
                       edge_irreps=edge_irreps,
                       correlation=correlation,
                       interaction_cls=interaction_cls,
                       interaction_cls_first=interaction_cls_first,
                       batch=batch)

        def func():
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            model(batch, training=False)
            torch.cuda.synchronize()
            max_mem_alloc = torch.cuda.max_memory_allocated()
            max_mem_reserved = torch.cuda.max_memory_reserved()
            benchmark.extra_info["max_memory_allocated"] = max_mem_alloc
            benchmark.extra_info["max_memory_reserved"] = max_mem_reserved

        torch.cuda.empty_cache()
        benchmark(func)


def create_mace(
        device,
        enable_cueq,

        num_interactions,
        max_ell,
        hidden_irreps,
        edge_irreps,
        correlation,
        interaction_cls,
        interaction_cls_first,
        ):

    z_table = AtomicNumberTable([6]) # TODO: generality

    model_config = {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "interaction_cls": modules.interaction_classes[interaction_cls],
        "interaction_cls_first": modules.interaction_classes[interaction_cls_first],
        "num_interactions": num_interactions,
        "num_elements": len(z_table),
        "hidden_irreps": o3.Irreps(hidden_irreps),
        "edge_irreps": o3.Irreps(edge_irreps),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": torch.ones(len(z_table)),
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": correlation,
        "radial_type": "bessel",
        "cueq_config": None,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }

    model = modules.ScaleShiftMACE(**model_config)
    if enable_cueq:
        model = run_e3nn_to_cueq(model)
    return model.to(device)


def create_batch(size: int, model: torch.nn.Module, device: str) -> dict:
    cutoff = model.r_max.item()
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    config = mace_data.config_from_atoms(atoms)
    dataset = [mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch.to(device)
    return batch.to_dict()


def log_bench_info(benchmark, dtype, enable_cueq, 
                   num_interactions,
                   max_ell,
                   hidden_irreps,
                   edge_irreps,
                   correlation,
                   interaction_cls,
                   interaction_cls_first,
                   batch):
    benchmark.extra_info["num_atoms"] = int(batch["positions"].shape[0])
    benchmark.extra_info["num_edges"] = int(batch["edge_index"].shape[1])
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["cueq_enabled"] = enable_cueq

    benchmark.extra_info["num_interactions"] = num_interactions
    benchmark.extra_info["max_ell"] = max_ell
    benchmark.extra_info["hidden_irreps"] = hidden_irreps
    benchmark.extra_info["edge_irreps"] = edge_irreps
    benchmark.extra_info["correlation"] = correlation
    benchmark.extra_info["interaction_cls"] = interaction_cls
    benchmark.extra_info["interaction_cls_first"] = interaction_cls_first


    benchmark.extra_info["device_name"] = torch.cuda.get_device_name()


def process_benchmark_file(bench_file: Path) -> pd.DataFrame:
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    records = []
    for bench in bench_data["benchmarks"]:
        record = {**bench["extra_info"], **bench["stats"]}
        records.append(record)

    result_df = pd.DataFrame(records)
    result_df["ns/day (1 fs/step)"] = 0.086400 / result_df["median"]
    result_df["Steps per day"] = result_df["ops"] * 86400
    columns = [
        "num_atoms",
        "num_edges",
        "dtype",
        "cueq_enabled",

        "num_interactions",
        "max_ell",
        "hidden_irreps",
        "edge_irreps",
        "correlation",
        "interaction_cls",
        "interaction_cls_first",

        "max_memory_allocated",
        "max_memory_reserved",

        "device_name",
        "median",
        "Steps per day",
        "ns/day (1 fs/step)",
    ]
    return result_df[columns]


def read_bench_results(result_files: List[str]) -> pd.DataFrame:
    return pd.concat([process_benchmark_file(Path(f)) for f in result_files])


if __name__ == "__main__":
    # Print to stdout a csv of the benchmark metrics
    import subprocess

    result = subprocess.run(
        ["pytest-benchmark", "list"], capture_output=True, text=True, check=True
    )

    print(result)

    bench_files = result.stdout.strip().split("\n")
    bench_results = read_bench_results(bench_files)
    print(bench_results.to_csv(index=False))
