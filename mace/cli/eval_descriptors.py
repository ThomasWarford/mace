###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import json
import os
import signal

import ase.data
import ase.io
import numpy as np
from npy_append_array import NpyAppendArray
import torch
from e3nn import o3

from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules.utils import extract_invariant
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from pathlib import Path


# Global Boolean variable that indicates that a signal has been received
INTERRUPTED = False

# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True

# Register the signal handler
signal.signal(signal.SIGTERM, signal_handler)


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
    # Continuation arguments
    parser.add_argument(
        "--checkpoint_interval",
        help="Save checkpoint after processing this many batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--resume",
        help="Resume from previous run",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--total_configs",
        help="Total number of configurations in XYZ file (optional, for progress reporting)",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def get_checkpoint_path(output_dir):
    return os.path.join(output_dir, "checkpoint.json")


def setup_model_and_directories(args):
    """Set up the model and output directories."""
    # Set up device and dtype
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = run_e3nn_to_cueq(model, device=device).to(device)
    # model=model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    # Set up output directories
    output_dir = Path(args.output_dir)
    if args.resume and output_dir.exists():
        print(f"Resuming from existing output directory: {output_dir}")
    else:
        output_dir.mkdir(exist_ok=args.resume)
        (output_dir/'all').mkdir(exist_ok=args.resume)

    # Create element directories
    zs = []
    for z in model.atomic_numbers:
        z = int(z)
        zs.append(z)
        element = ase.data.chemical_symbols[z]
        (output_dir/element).mkdir(exist_ok=args.resume)
        
        # Create files if they don't exist (for appending)
        if not (output_dir/element/'index.npy').exists() and args.resume:
            with open(output_dir/element/'index.npy', 'wb') as f:
                pass
        if not (output_dir/element/'descriptor.npy').exists() and args.resume:
            with open(output_dir/element/'descriptor.npy', 'wb') as f:
                pass
    
    # Create all/index.npy and all/descriptor.npy if they don't exist
    if args.resume:
        if not (output_dir/'all'/'index.npy').exists():
            with open(output_dir/'all'/'index.npy', 'wb') as f:
                pass
        if not (output_dir/'all'/'descriptor.npy').exists():
            with open(output_dir/'all'/'descriptor.npy', 'wb') as f:
                pass
                
    z_table = utils.AtomicNumberTable(zs)
    
    try:
        heads = model.heads
    except AttributeError:
        heads = None
        
    return model, output_dir, z_table, heads, device


def get_descriptor_parameters(model, args):
    """Get parameters for extracting descriptors."""
    num_layers = args.descriptor_num_layers
    if num_layers == -1:
        num_layers = model.num_interactions
        
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
    per_layer_features = [irreps_out.dim for _ in range(int(model.num_interactions))]
    per_layer_features[-1] = num_invariant_features  # Equivariant features not created for the last layer
    
    to_keep = np.sum(per_layer_features[:num_layers])
    
    return num_layers, l_max, num_invariant_features, to_keep


def process_batch(batch, model, descriptor_params, args, device):
    """Process a single batch and return descriptors."""
    batch = batch.to(device)
    output = model(batch.to_dict(), compute_stress=args.compute_stress)
    
    num_layers, l_max, num_invariant_features, to_keep = descriptor_params
    
    descriptors = output['node_feats']
    if args.descriptor_invariants_only:
        descriptors = extract_invariant(
            descriptors,
            num_layers=num_layers,
            num_features=num_invariant_features,
            l_max=l_max,
        )

    descriptors = descriptors[:, :to_keep].detach().cpu().numpy()
    
    # Split by graph
    descriptors_per_graph = np.split(
        descriptors,
        indices_or_sections=batch.ptr[1:],
        axis=0,
    )
    def print_nested_keys(dictionary, prefix=""):
        """
        Recursively prints all keys in a nested dictionary.
        
        Args:
            dictionary (dict): The dictionary to traverse
            prefix (str): Used to track the current path in the recursion
        """
        for key, value in dictionary.items():
            current_key = f"{prefix}.{key}" if prefix else key
            print(current_key)
            
            if isinstance(value, dict):
                print_nested_keys(value, current_key)
    # Get atom info for each graph
    atoms_info = []
    for i, ptr in enumerate(range(len(batch.ptr) - 1)):
        config_id = batch.config_id[i] if hasattr(batch, 'config_id') else None
        atoms_per_graph = {
            'ptr': ptr,
            'num_atoms': int(batch.ptr[i+1] - batch.ptr[i]),
            'atomic_numbers': model.atomic_numbers[torch.argmax(batch.node_attrs[batch.ptr[i]:batch.ptr[i+1]], axis=1)].detach().cpu().numpy(),
            'batch_idx': i,
            'config_id': (config_id.cpu() if isinstance(config_id, torch.Tensor) else config_id),
        }
        atoms_info.append(atoms_per_graph)
    
    return descriptors_per_graph[:-1], atoms_info  # Last element is empty


def save_descriptors(descriptors, atoms_info, z_table, config_indices, output_dir, args):
    """Save descriptors to output files."""
    # Prepare data for all directory
    all_config_ids = []
    all_mean_descriptors = []
    
    # Process each configuration
    for i, (descriptor, atom_info) in enumerate(zip(descriptors, atoms_info)):
        # Get config ID from batch or use index
        config_id = atom_info['config_id'] #or str(config_indices[i])
        all_config_ids.append(config_id)
        
        # Calculate mean descriptor for the entire configuration
        mean_descriptor = np.mean(descriptor, axis=0)
        all_mean_descriptors.append(mean_descriptor)
        
        # Get atomic numbers and chemical symbols
        atomic_numbers = atom_info['atomic_numbers']
        chemical_symbols = [ase.data.chemical_symbols[int(z)] for z in atomic_numbers]
        
        # Save per-element descriptors
        for element in set(chemical_symbols):
            # Skip if element directory doesn't exist
            if not (output_dir/element).exists():
                continue
                
            # Get indices of atoms of this element
            element_indices = [j for j, symbol in enumerate(chemical_symbols) if symbol == element]
            if not element_indices:
                continue
                
            # Calculate mean descriptor for this element
            element_descriptor = np.mean(descriptor[element_indices], axis=0, keepdims=True)
            
            # Save to element directory
            with NpyAppendArray(output_dir/element/'index.npy') as npaa:
                npaa.append(np.array(config_id, ndmin=1))
            with NpyAppendArray(output_dir/element/'descriptor.npy') as npaa:
                npaa.append(element_descriptor)
    
    # Save to all directory
    if all_config_ids:
        with NpyAppendArray(output_dir/'all/index.npy') as npaa:
            npaa.append(np.array(all_config_ids))
        with NpyAppendArray(output_dir/'all/descriptor.npy') as npaa:
            npaa.append(np.stack(all_mean_descriptors))


def get_dataset_length(xyz_file):
    """Get the number of configurations in the XYZ file."""
    try:
        # Fast counting by only reading headers
        with open(xyz_file, 'r') as f:
            # Count number of lines that have just a number (atom count lines)
            count = 0
            for line in f:
                if line.strip().isdigit():
                    count += 1
        return count
    except Exception as e:
        print(f"Error counting configurations in XYZ file: {e}")
        print("Will continue without total count")
        return None


def run(args: argparse.Namespace) -> None:
    # Set up model, directories, and parameters
    model, output_dir, z_table, heads, device = setup_model_and_directories(args)
    descriptor_params = get_descriptor_parameters(model, args)
    
    # Load checkpoint if resuming
    checkpoint_path = get_checkpoint_path(output_dir)
    processed_count = 0
    start_idx = 0
    
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('next_start_idx', 0)
            processed_count = checkpoint.get('processed_count', 0)
            print(f"Resuming from checkpoint, starting at index {start_idx}, processed count {processed_count}")
    
    # Get dataset size if not provided
    total_configs = args.total_configs
    if total_configs is None:
        print("Counting total configurations in XYZ file...")
        total_configs = get_dataset_length(args.configs)
        if total_configs:
            print(f"Found {total_configs} configurations in XYZ file")
    
    # Create dataset from XYZ file with on-demand loading
    class LazyXYZDataset(torch.utils.data.IterableDataset):
        def __init__(self, xyz_file, z_table, r_max, heads=None, start_idx=0, head=None):
            self.xyz_file = xyz_file
            self.z_table = z_table
            self.r_max = r_max
            self.heads = heads
            self.head = head
            self.start_idx = start_idx
            self.current_idx = start_idx
            
        def __iter__(self):
            # Open the XYZ file and skip to the start_idx
            self.atoms_iter = ase.io.iread(self.xyz_file, index=f"{self.start_idx}:")
            self.current_idx = self.start_idx
            return self
            
        def __next__(self):
            try:
                atoms = next(self.atoms_iter)
                if self.head is not None:
                    atoms.info["head"] = self.head
                
                config = data.config_from_atoms(atoms)
                atomic_data = data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
                
                # Add config index for tracking
                try:
                    atomic_data.config_id = atoms.info['mp_id'] + '-' + str(atoms.info['calc_id']) + '-' + str(atoms.info['ionic_step'])
                except:
                    atomic_data.config_id = self.current_idx
                self.current_idx += 1
                
                return atomic_data
            except StopIteration:
                raise StopIteration
    
    # Create dataset
    dataset = LazyXYZDataset(
        args.configs, 
        z_table, 
        float(model.r_max), 
        heads=heads, 
        start_idx=start_idx,
        head=args.head
    )
    
    # Custom collate function to handle batch processing
    def collate_fn(data_list):
        if not data_list:
            return None
        return torch_geometric.batch.Batch.from_data_list(data_list)
    
    # Create dataloader with custom collate function
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Must be 0 for IterableDataset with file reading
    )
    
    # Process batches with checkpointing
    batch_counter = 0
    current_idx = start_idx
    
    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
            
        # Process batch
        descriptors, atoms_info = process_batch(batch, model, descriptor_params, args, device)
        
        # Calculate indices for this batch
        batch_indices = list(range(current_idx, current_idx + len(descriptors)))
        current_idx += len(descriptors)
        
        # Save descriptors
        save_descriptors(descriptors, atoms_info, z_table, batch_indices, output_dir, args)
        
        # Update processed count
        processed_count += len(descriptors)
        
        # Progress reporting
        if total_configs:
            progress = (current_idx / total_configs) * 100
            print(f"Processed {processed_count} configurations ({progress:.2f}% complete)")
        else:
            print(f"Processed {processed_count} configurations (current index: {current_idx})")
        
        # Checkpoint after interval
        batch_counter += 1
        if batch_counter >= args.checkpoint_interval:
            # Save checkpoint
            checkpoint = {
                'next_start_idx': current_idx,
                'processed_count': processed_count,
                'timestamp': np.datetime_as_string(np.datetime64('now'))
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
                
            print(f'Saved checkpoint at index {current_idx}.')
            batch_counter = 0
            
            # Free memory
            torch.cuda.empty_cache()

            if INTERRUPTED:
                break
                

    
    # Final checkpoint
    checkpoint = {
        'next_start_idx': current_idx,
        'processed_count': processed_count,
        'timestamp': np.datetime_as_string(np.datetime64('now')),
        'completed': not INTERRUPTED
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)
    
    # Clean up empty directories at the end
    for dir in output_dir.rglob('*'):
        if dir.is_dir() and not any(dir.iterdir()):
            dir.rmdir()
    
    print(f"Processing {("interrupted" if INTERRUPTED else "complete")}. Total configurations processed: {processed_count}")


if __name__ == "__main__":
    main()
