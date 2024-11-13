###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.utils.data
from scipy.constants import c, e
from torch.nn import Parameter
from torch_scatter import scatter_min

from mace.tools import to_numpy
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric.batch import Batch
from .blocks import AtomicEnergiesBlock


def compute_forces(
        energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
        energy: torch.Tensor,
        positions: torch.Tensor,
        displacement: torch.Tensor,
        cell: torch.Tensor,
        training: bool = True,
        compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_symmetric_displacement(
        positions: torch.Tensor,
        unit_shifts: torch.Tensor,
        cell: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_graphs: int,
        batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
            displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


@torch.jit.unused
def compute_hessians_vmap(
        forces: torch.Tensor,
        positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
        forces: torch.Tensor,
        positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def get_outputs(
        energy: torch.Tensor,
        positions: torch.Tensor,
        displacement: Optional[torch.Tensor],
        cell: torch.Tensor,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = True,
        compute_stress: bool = True,
        compute_hessian: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        # forces come for free
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian),
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    return forces, virials, stress, hessian


def get_edge_vectors_and_lengths(
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
        normalize: bool = False,
        eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def _check_non_zero(std):
    if std == 0.0:
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std = 1.0
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    for i in range(num_layers - 1):
        out.append(
            x[
            :,
            i
            * (l_max + 1) ** 2
            * num_features: (i * (l_max + 1) ** 2 + 1)
                            * num_features,
            ]
        )
    out.append(x[:, -num_features:])
    return torch.cat(out, dim=-1)


def compute_mean_std_atomic_inter_energy(
        data_loader: torch.utils.data.DataLoader,
        atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()
    std = _check_non_zero(std)

    return mean, std


def _compute_mean_std_atomic_inter_energy(
        batch: Batch,
        atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes
    return atom_energies


def compute_mean_rms_energy_forces(
        data_loader: torch.utils.data.DataLoader,
        atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    rms = _check_non_zero(rms)

    return mean, rms


def _compute_mean_rms_energy_forces(
        batch: Batch,
        atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes  # {[n_graphs], }
    forces = batch.forces  # {[n_graphs*n_atoms,3], }

    return atom_energies, forces


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_statistics(
        data_loader: torch.utils.data.DataLoader,
        atomic_energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


def compute_rms_dipoles(
        data_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    dipoles_list = []
    for batch in data_loader:
        dipoles_list.append(batch.dipole)  # {[n_graphs,3], }

    dipoles = torch.cat(dipoles_list, dim=0)  # {[total_n_graphs,3], }
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(dipoles)))).item()
    rms = _check_non_zero(rms)
    return rms


def compute_fixed_charge_dipole(
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]


# %% initial reference parameters
def initial_ref_params(
        num_types: int,
        default_dtype: str,
        Q_tot: float,
        eps: float,
        A_ref: Union[float, list],
        B_ref: Union[float, list],
        C_ref: Union[float, list],
        D_ref: Union[float, list],
        mu_ref: Union[float, list],
        sigma_ref: Union[float, list],
        eta_ref: Union[float, list],
):
    # atom types
    dtype = getattr(torch, default_dtype)
    torch.set_default_dtype(dtype)
    # parameters from reference
    A_ref = torch.tensor(A_ref, dtype=dtype)
    B_ref = torch.tensor(B_ref, dtype=dtype)
    C_ref = torch.tensor(C_ref, dtype=dtype)
    D_ref = torch.tensor(D_ref, dtype=dtype)
    mu_ref = torch.tensor(mu_ref, dtype=dtype)
    sigma_ref = torch.tensor(sigma_ref, dtype=dtype)
    eta_ref = torch.tensor(eta_ref, dtype=dtype)
    if A_ref.dim() == 0:
        # random implementation of 2b parameters
        rand_A = torch.rand(num_types, num_types, dtype=dtype) * A_ref
        rand_B = torch.rand(num_types, num_types, dtype=dtype) * B_ref
        rand_C = torch.rand(num_types, num_types, dtype=dtype) * C_ref
        rand_D = torch.rand(num_types, num_types, dtype=dtype) * D_ref
        rand_mu = torch.rand(num_types, num_types, dtype=dtype) * mu_ref
        ref_A = Parameter((rand_A + rand_A.T) / 2, requires_grad=True)
        ref_B = Parameter((rand_B + rand_B.T) / 2, requires_grad=True)
        ref_C = Parameter((rand_C + rand_C.T) / 2, requires_grad=True)
        ref_D = Parameter((rand_D + rand_D.T) / 2, requires_grad=True)
        ref_mu = Parameter((rand_mu + rand_mu.T) / 2, requires_grad=True)
        # random implementation of Gaussian charge density parameters
        # ensure sigma is always positive
        ref_log_sigma = Parameter(torch.rand(num_types, dtype=dtype) * torch.log(sigma_ref), requires_grad=True)
        ref_eta = Parameter(torch.rand(num_types, dtype=dtype) * eta_ref, requires_grad=True)
    elif A_ref.dim() == 2:
        ref_A = Parameter(A_ref, requires_grad=True)
        ref_B = Parameter(B_ref, requires_grad=True)
        ref_C = Parameter(C_ref, requires_grad=True)
        ref_D = Parameter(D_ref, requires_grad=True)
        ref_mu = Parameter(mu_ref, requires_grad=True)
        # ensure sigma is always positive
        ref_log_sigma = Parameter(torch.log(sigma_ref), requires_grad=True)
        ref_eta = Parameter(torch.tensor(eta_ref, dtype=dtype), requires_grad=True)
    else:
        raise ValueError("Reference parameters must be provided in config!")

    # ee4G-HDNNP configs
    ee4GHDNNP_config = {
        # reference parameters
        "Q_tot": Q_tot,
        "eps": eps,
        "ref_A": ref_A,
        "ref_B": ref_B,
        "ref_C": ref_C,
        "ref_D": ref_D,
        "ref_mu": ref_mu,
        "ref_eta": ref_eta,
        "ref_log_sigma": ref_log_sigma,
    }

    return ee4GHDNNP_config


# %% atom-type parameters to atom-wise parameters
@torch.jit.script
def make_parameters(ref: torch.Tensor, indices: List[torch.Tensor]) -> torch.Tensor:
    """
    Fabricate two body energy and electric parameters from reference data.
    """
    num_types = len(indices)
    num_atoms = torch.max(torch.cat(indices), dim=0)[0].item() + 1
    param = torch.zeros(num_atoms, num_atoms).type_as(ref)
    for i in range(num_types):
        for j in range(num_types):
            param[indices[i][:, None], indices[j]] = ref[i, j]

    return param


# %% calculation of distance matrix R
@torch.jit.script
def create_distance_matrix(edge_index: torch.Tensor, edge_length: torch.Tensor, eps: float) -> torch.Tensor:
    # Assume edge_index is of shape [2, num_edges] and edge_length is of shape [num_edges]
    src = edge_index[0]
    dst = edge_index[1]
    # Get the number of nodes (assuming gamma is a square matrix of shape [num_atoms, num_atoms])
    num_atoms = int(edge_index.max().item() + 1)
    # Create undirected edge indices by sorting the node pairs
    min_nodes = torch.min(src, dst)
    max_nodes = torch.max(src, dst)
    # Map undirected edge pairs to unique edge keys
    edge_keys = min_nodes * num_atoms + max_nodes
    # Get unique undirected edges and inverse indices
    unique_edge_keys, inverse_indices = torch.unique(edge_keys, return_inverse=True)
    # Use scatter_min to get minimal edge length over undirected edges
    min_edge_length, _ = scatter_min(edge_length, inverse_indices, dim=0)
    # Assign minimal edge lengths back to edges
    edge_length = min_edge_length[inverse_indices]
    # Create the distance matrix R
    R = torch.full((num_atoms, num_atoms), 1 / eps).type_as(edge_length)
    R[edge_index[0], edge_index[1]] = edge_length

    return R
