###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
from npy_append_array import NpyAppendArray
import torch
from e3nn import o3

from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules.utils import extract_invariant
from itertools import islice
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output_dir", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_descriptors",
        help="model outputs MACE descriptors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--descriptor_num_layers",
        help="number of layers to take descriptors from",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--descriptor_aggregation_method",
        help="method for aggregating node features",
        type=str,
        choices=["mean", "per_element_mean"],
        default="mean",
    )
    parser.add_argument(
        "--descriptor_invariants_only",
        help="save invariant (l=0) descriptors only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--index_key",
        help="Key identifying configuration corresponding with descriptor",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)

def next_n_items(iterator, n):
    return items if (items := list(islice(iterator, n))) else None

def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_iter = ase.io.iread(args.configs, index=":")
    zs = []
    output_dir = Path(args.output_dir); output_dir.mkdir(exist_ok=False)
    (output_dir/'all').mkdir(exist_ok=False)
    for z in model.atomic_numbers:
        z = int(z)
        zs.append(z)
        (output_dir/ase.data.chemical_symbols[z]).mkdir(exist_ok=False)
    z_table = utils.AtomicNumberTable(zs)

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    # Process data in batches to avoid memory issues
    start_idx = 0
    batches_per_iteration = 100
    atoms_count = 1
    while batch_atoms := next_n_items(atoms_iter, batches_per_iteration * args.batch_size):
        if args.head is not None:
            for atoms in batch_configs:
                atoms.info["head"] = args.head
        caught_up = any(atoms)
        batch_configs = [data.config_from_atoms(atoms) for atoms in batch_atoms]
        batch_data = [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in batch_configs
        ]
        # Create a dataloader for just the batches_per_iteration batches
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=batch_data,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        descriptors_list = []
        
        # Process this batch
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch.to_dict(), compute_stress=args.compute_stress)
            
            num_layers = args.descriptor_num_layers
            if num_layers == -1: num_layers = model.num_interactions
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
            per_layer_features = [irreps_out.dim for _ in range(int(model.num_interactions))]
            per_layer_features[-1] = (
                num_invariant_features  # Equivariant features not created for the last layer
            )

            descriptors = output['node_feats']
            if args.descriptor_invariants_only:
                descriptors = extract_invariant(
                    descriptors,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                    )

            to_keep = np.sum(per_layer_features[:num_layers])
            descriptors = descriptors[:, :to_keep].detach().cpu().numpy()
            print(descriptors.shape)
            descriptors = np.split(
                descriptors,
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            descriptors_list.extend(descriptors[:-1]) # drop last as its empty
            
            # Store results in atoms objects
        config_ids = []
        mean_descriptors = []
        for atoms, descriptor in zip(batch_atoms, descriptors_list):
            atoms_count += 1
            mean_descriptors.append(np.mean(descriptor, axis=0))
            element_to_descriptor = {
                element: np.mean(descriptor[np.array(atoms.get_chemical_symbols()) == element], axis=0, keepdims=True)
                for element in np.unique(atoms.get_chemical_symbols())
            }
            config_id = atoms.info.get(args.index_key, 'NaN')
            config_ids.append(config_id)
            for element, descriptor in element_to_descriptor.items():
                if (output_dir/element).exists():
                    with NpyAppendArray(output_dir/element/'index.npy') as npaa:
                        npaa.append(np.array(config_id, ndmin=1))
                    with NpyAppendArray(output_dir/element/'descriptor.npy') as npaa:
                        npaa.append(descriptor)
        print(f'Done {atoms_count} atoms')

        with NpyAppendArray(output_dir/'all/index.npy') as npaa:
            npaa.append(np.array(config_ids,))
        with NpyAppendArray(output_dir/'all/descriptor.npy') as npaa:
            npaa.append(np.stack(mean_descriptors))
        
    for dir in output_dir.rglob('*'):
        if dir.is_dir() and not any(dir.iterdir()): dir.rmdir()
        # Write this batch to file with append=True
        # # Free memory
        # del batch, output
        # if 'descriptors' in locals():
        #     del descriptors, processed_descriptors
        # torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
