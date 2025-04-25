import torch
from torch.utils.data import DataLoader
from pathlib import Path
from mace.data import AtomicData
from mace.data import LMDBDataset
from mace.tools import torch_geometric, utils, torch_tools
from mace.calculators import mace_mp
from e3nn import o3
from mace.modules.utils import extract_invariant
import numpy as np
import ase
from npy_append_array import NpyAppendArray
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

def main():
    device = 'cuda'
    path = '/lustre/fsn1/projects/rech/gax/ums98bp/salex/val'
    model = mace_mp('/lustre/fswork/projects/rech/gax/ums98bp/.models/medium-omat-0.model', device=device, return_raw_model=True)
    model = run_e3nn_to_cueq(model)
    model.to(device)
    ds = LMDBDataset(
        path,
        float(model.r_max),
        utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    )

    dataloader = torch_geometric.dataloader.DataLoader(
        dataset=ds,
        batch_size=8,
        shuffle=False,
        drop_last=False
    )

    save_descriptors(
        model,
        dataloader,
        Path('/lustre/fswork/projects/rech/gax/ums98bp/eval_dataset/omat/val'),
        device,
    )

def save_descriptors(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: str,
    save_config_info:bool=True,
    ):
    torch_tools.set_default_dtype("float64")

    output_dir.mkdir(exist_ok=True)
    (output_dir/'all').mkdir(exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        
        batch.to(device)
        output = model(batch.to_dict(),)
        descriptors_list = get_descriptors(batch, output, model)        
        atomic_numbers = torch.matmul(batch.node_attrs, torch.atleast_2d(model.atomic_numbers.double()).T)

                
        atomic_numbers_list = np.split(
                    torch_tools.to_numpy(atomic_numbers),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                    )[:-1] # drop last as its empty
        
        residuals = torch_tools.to_numpy(output["energy"]) - torch_tools.to_numpy(batch.energy)

        for idx, (identifier, residual, atomic_numbers, descriptors) in enumerate(zip(batch.identifier, residuals, atomic_numbers_list, descriptors_list)):
            with NpyAppendArray(output_dir/'all/descriptors.npy') as npaa:
                npaa.append(descriptors.astype(np.float16))
            
            if save_config_info:
                N = len(atomic_numbers)
                dtype = np.dtype([
                        ('identifier', '<U30'),
                        ('dataset_index', np.uint32),
                        ('chemical_formula', '<U30'),
                        ('energy_residual', np.float16),
                        ('atomic_number', np.uint8),
                        ('indices', np.uint8),
                        ])
                config_info = np.empty(N, dtype=dtype)
                config_info['identifier'] = identifier
                config_info['dataset_index'] = batch_idx*dataloader.batch_size+idx
                config_info['chemical_formula'] = ase.formula.Formula.from_list([ase.data.chemical_symbols[int(z)] for z in atomic_numbers]).format('hill') 
                config_info['energy_residual'] = residual
                config_info['atomic_number'] = atomic_numbers.flatten()
                config_info['indices'] = np.arange(N)

                with NpyAppendArray(output_dir/'all/index.npy') as npaa:
                    npaa.append(config_info)
            





def get_descriptors(batch, output, model, num_layers=-1, invariants_only=True):

        if num_layers == -1:
            num_layers = int(model.num_interactions)
        irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [
            irreps_out.dim for _ in range(int(model.num_interactions))
        ]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        descriptors = output["node_feats"]

        if invariants_only:
            descriptors = extract_invariant(
                descriptors,
                num_layers=num_layers,
                num_features=num_invariant_features,
                l_max=l_max,
            )

        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = descriptors[:, :to_keep].detach().cpu().numpy()

        descriptors = np.split(
            descriptors,
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        return descriptors[:-1]  # (drop last as its empty)

main()