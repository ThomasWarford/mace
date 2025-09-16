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



@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.benchmark(warmup=True, warmup_iterations=4, min_rounds=8)
@pytest.mark.parametrize("size", (3,))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("enable_cueq", [False, True])

@pytest.mark.parametrize("max_ell", [2, 3, 4])
@pytest.mark.parametrize("hidden_irreps", ['128x0e + 128x1o + 128x2e'])


def test_inference(
    benchmark, 
    size: int, 
    dtype: str, 
    enable_cueq: bool, 

    max_ell: int,
    hidden_irreps: o3.Irreps ,

    device: str = "cuda",
        ):

    with torch_tools.default_dtype(dtype):
        model = create_mace(
            device,
            enable_cueq,

            max_ell,
            hidden_irreps,
            )
        batch = create_batch(size, model, device)
        log_bench_info(benchmark, dtype, enable_cueq, 
                       max_ell=max_ell,
                       hidden_irreps=hidden_irreps,
                       batch=batch)

        def func():
            torch.cuda.synchronize()
            model(batch, training=False)

        torch.cuda.empty_cache()
        benchmark(func)


def create_mace(
        device,
        enable_cueq,

        max_ell,
        hidden_irreps,
        ):

    z_table = AtomicNumberTable([6]) # TODO: generality

    model_config = {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "num_interactions": 2,
        "num_elements": len(z_table),
        "hidden_irreps": o3.Irreps(hidden_irreps),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": torch.ones(len(z_table)),
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": 3,
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
                   max_ell,
                   hidden_irreps,
                   batch):
    benchmark.extra_info["num_atoms"] = int(batch["positions"].shape[0])
    benchmark.extra_info["num_edges"] = int(batch["edge_index"].shape[1])
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["cueq_enabled"] = enable_cueq

    benchmark.extra_info["max_ell"] = max_ell
    benchmark.extra_info["hidden_irreps"] = hidden_irreps

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
        "is_compiled",
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

    bench_files = result.stdout.strip().split("\n")
    bench_results = read_bench_results(bench_files)
    print(bench_results.to_csv(index=False))
