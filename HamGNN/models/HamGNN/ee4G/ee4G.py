
""" 
added by Xiwen Li

including long range interaction by using ee4GHDNNP block to predict net charge on atoms
""" 
import ast
import logging
import math
from torch import Tensor
from torch_runstats.scatter import scatter

from .ee4G_utils import make_parameters, create_distance_matrix


####
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

from torch_scatter import scatter
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum
from .ee4G_block import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    KANReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
    RealAgnosticResidualInteractionBlock
)
from .ee4G_utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    initial_ref_params,
)

# %% ee4G-HDNNP cut off function
@torch.jit.script
def f_cut(r: Tensor, r_max: float, r_in: float) -> Tensor:
    """
    ee4G-HDNNP cut off function
    """
    result = torch.zeros_like(r)
    condition_1 = (r > 0.) & (r < r_in)
    condition_2 = (r >= r_in) & (r <= r_max)

    result[condition_1] = torch.tanh(torch.tensor(1)) ** 3
    result[condition_2] = torch.tanh(1 - (r[condition_2] - r_in) / (r_max - r_in)) ** 3

    return result

# %% electric equilibrium model
@compile_mode("script")
class electric_equilibrium(torch.nn.Module):
    def __init__(
            self,
            ee4GHDNNP_config: Dict[str, Union[torch.Tensor, float]],
            r_max: float = 6.,
            r_in: float = 1.,
    ):
        super().__init__()
        # electric energy parameters
        self.Q_tot = ee4GHDNNP_config["Q_tot"]
        self.ref_eta = ee4GHDNNP_config["ref_eta"]
        self.ref_log_sigma = ee4GHDNNP_config["ref_log_sigma"]

        # cutoff parameters
        self.r_max = r_max
        self.r_in = r_in

        self.eps = ee4GHDNNP_config["eps"]

    def make_params(self, data: Dict[str, torch.Tensor]):
        # deal with batch size & device
        batch = data["batch"]
        batch_size = torch.unique(batch).size(0)

        # direct sum of parameters with respect to batches
        # ensure sigmma is always positive
        ref_sigma = torch.exp(self.ref_log_sigma)
        # ensure the relationship between sigma and gamma
        ref_gamma = torch.sqrt(ref_sigma[:, None] ** 2 + ref_sigma[None, :] ** 2)

        batch_num_nodes = batch.size(0)
        gamma = torch.ones(batch_num_nodes, batch_num_nodes).type_as(ref_gamma) * self.eps
        eta = torch.zeros(batch_num_nodes).type_as(self.ref_eta)
        sigma = torch.zeros(batch_num_nodes).type_as(ref_sigma)
        # initial index for slice
        index = 0
        for batch_i in range(batch_size):
            # atom type indices
            batch_node_attrs = data["node_attrs"][data["batch"] == batch_i]
            num_types = batch_node_attrs.size(1)
            num_atoms = batch_node_attrs.size(0)
            indices_list = [torch.nonzero(batch_node_attrs[:, i]).squeeze(-1) for i in range(num_types)]
            # atom-type parameters to per-atom parameters
            # Gaussian charge density parameters
            gamma_i = make_parameters(ref_gamma, indices_list)
            eta_sigma = torch.zeros(num_atoms, 2).type_as(self.ref_eta)
            for i in range(num_types):
                eta_sigma[indices_list[i], 0] = self.ref_eta[i]
                eta_sigma[indices_list[i], 1] = ref_sigma[i]
            eta_i, sigma_i = eta_sigma[:, 0], eta_sigma[:, 1]

            gamma[index: index + num_atoms, index: index + num_atoms] = gamma_i
            eta[index: index + num_atoms] = eta_i
            sigma[index: index + num_atoms] = sigma_i
            index += num_atoms

        return sigma, gamma, eta, data

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = data["batch"]
        batch_size = torch.unique(batch).size(0)

        sigma, gamma, eta, data = self.make_params(data)
        kappa = data["kappa"]
        edge_index = data["edge_index"]
        edge_length = data["edge_length"].squeeze(-1)
        node_feats = data["node_feats"]

        # distance matrix
        R = create_distance_matrix(edge_index, edge_length, self.eps)
        F_cut = f_cut(R, self.r_max, self.r_in)
        # write into data
        data["R"] = R
        data["F_cut"] = F_cut

        # electric energy
        # charge equilibrium
        factor = torch.erf(R / (math.sqrt(2) * gamma)) / R
        A_off_diagonal_part = factor
        A_diagonal_part = torch.diag(eta + 1 / (sigma * math.sqrt(math.pi)))
        A_ij = A_diagonal_part + A_off_diagonal_part

        # deal with permutation equivalence
        if "node_perm" in data.keys():
            # TODO: When running node permutation equivariance test, must add key "node_perm" to data(dict type).
            # permed data
            node_perm = data["node_perm"]
            inv_node_perm = data["inv_node_perm"]
            A_ij_orig = A_ij[:, inv_node_perm][inv_node_perm, :]
            A_ij_orig = torch.tril(A_ij_orig, diagonal=0)
            A_ij = A_ij_orig[:, node_perm][node_perm, :]

            del data["node_perm"]
            del data["inv_node_perm"]
        else:
            # original data
            A_ij = torch.tril(A_ij, diagonal=0)

        # deal with total charge
        batch_num_nodes = batch.size(0)
        # append total charge limitation equation
        A_ij_ext = torch.zeros(batch_num_nodes + batch_size, batch_num_nodes + batch_size).type_as(A_ij)
        kappa_ext = torch.zeros(batch_num_nodes + batch_size).type_as(kappa)
        # append a False mask at the end of each batch, in order to rule out total charge limitation solve
        batch_ext = torch.zeros(batch_num_nodes + batch_size)
        # initial index for slice
        index = 0
        for batch_i in range(batch_size):
            batch_mask = data["batch"] == batch_i
            batch_ext = batch_ext.type_as(batch_mask)
            # A matrix
            # A_ij of every batch is a lower triangle matrix.
            # Add [1,1,...,0] to the last column and the last row of A_ij for every batch
            # and stack A_ij from every batch to a big block diagonal tensor.
            A_batch = A_ij[batch_mask, :][:, batch_mask]
            num_atoms = A_batch.size(0)
            # append cols and rows
            new_row = torch.ones(1, num_atoms).type_as(A_batch)
            new_col = torch.ones(num_atoms + 1, 1).type_as(A_batch)
            A_batch = torch.cat((A_batch, new_row), dim=0)
            A_batch = torch.cat((A_batch, new_col), dim=1)
            A_batch[-1, -1] = 0
            # kappa
            # Add Q_tot to the end of kappa from each batch.
            kappa_batch = kappa[batch_mask]
            new_item = torch.tensor([self.Q_tot]).type_as(kappa_batch)
            kappa_batch = torch.cat((kappa_batch, new_item), dim=0)
            # batch
            # Add False mask to the end of each batch_mask, in order to unexpected solutions.
            true_mask = batch_mask[batch_mask]
            false_element = torch.tensor([False]).type_as(batch_mask)
            batch = torch.cat((true_mask, false_element), dim=0)

            A_ij_ext[index: index + num_atoms + 1, index: index + num_atoms + 1] = A_batch
            kappa_ext[index: index + num_atoms + 1] = kappa_batch
            batch_ext[index: index + num_atoms + 1] = batch
            index += num_atoms + 1

        # electric equilibrium with total charge limitation
        atomic_charges = torch.linalg.solve(A_ij_ext, -kappa_ext)
        atomic_charges = atomic_charges[batch_ext]

        # short range charge potential descriptor
        atomic_potentials_factor = factor * F_cut
        atomic_potentials = atomic_potentials_factor @ atomic_charges
        # energy calculation
        E_factor_diagonal = torch.diag(0.5 / (sigma * torch.sqrt(torch.tensor(torch.pi))))
        E_factor = E_factor_diagonal + A_off_diagonal_part
        V = E_factor @ atomic_charges
        atomic_E_electric = atomic_charges * V
        E_electric = scatter(atomic_E_electric, data["batch"], dim=0, reduce="sum")
        # write into data
        data["electric_energy"] = E_electric
        data["atomic_electric_energy"] = atomic_E_electric

        # concatenate kappa, Q and V
        data["node_feats"] = torch.cat([node_feats, atomic_charges.unsqueeze(-1), atomic_potentials.unsqueeze(-1)], dim=1)
        data["atomic_charges"] = atomic_charges
        return data

# %% sum total energy
@compile_mode("script")
class total_energy_sum(torch.nn.Module):
    def __init__(self, ee4GHDNNP_config: Dict[str, torch.Tensor]):
        super().__init__()
        # parameters for 2b energy
        self.ref_A = ee4GHDNNP_config["ref_A"]
        self.ref_B = ee4GHDNNP_config["ref_B"]
        self.ref_C = ee4GHDNNP_config["ref_C"]
        self.ref_D = ee4GHDNNP_config["ref_D"]
        self.ref_mu = ee4GHDNNP_config["ref_mu"]

    def make_params(self, data: Dict[str, torch.Tensor]):
        # deal with batch size & device
        batch = data["batch"]
        batch_size = torch.unique(batch).size(0)

        batch_num_nodes = batch.size(0)
        A = torch.zeros(batch_num_nodes, batch_num_nodes).type_as(self.ref_A)
        B = torch.zeros(batch_num_nodes, batch_num_nodes).type_as(self.ref_B)
        C = torch.zeros(batch_num_nodes, batch_num_nodes).type_as(self.ref_C)
        D = torch.zeros(batch_num_nodes, batch_num_nodes).type_as(self.ref_D)
        mu = torch.zeros(batch_num_nodes, batch_num_nodes).type_as(self.ref_mu)
        # initial index for slice
        index = 0
        for batch_i in range(batch_size):
            # atom type indices
            # atom type indices
            batch_node_attrs = data["node_attrs"][data["batch"] == batch_i]
            num_types = batch_node_attrs.size(1)
            num_atoms = batch_node_attrs.size(0)
            indices_list = [torch.nonzero(batch_node_attrs[:, i]).squeeze(-1) for i in range(num_types)]
            # atom-type parameters to per-atom parameters
            # 2b parameters
            # A, B, C, D, mu are block diagonal tensors.
            A_i = make_parameters(self.ref_A, indices_list)
            B_i = make_parameters(self.ref_B, indices_list)
            C_i = make_parameters(self.ref_C, indices_list)
            D_i = make_parameters(self.ref_D, indices_list)
            mu_i = make_parameters(self.ref_mu, indices_list)

            A[index: index + num_atoms, index: index + num_atoms] = A_i
            B[index: index + num_atoms, index: index + num_atoms] = B_i
            C[index: index + num_atoms, index: index + num_atoms] = C_i
            D[index: index + num_atoms, index: index + num_atoms] = D_i
            mu[index: index + num_atoms, index: index + num_atoms] = mu_i
            index += num_atoms

        R = data["R"]
        F_cut = data["F_cut"]
        del data["R"]
        del data["F_cut"]

        return A, B, C, D, mu, R, F_cut

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        A, B, C, D, mu, R, F_cut = self.make_params(data)
        # two-body energy
        E_2b_ij = (A * torch.exp(B * (mu - R)) - C / R ** 6 - D / R ** 8) * F_cut
        E_2b_ij.fill_diagonal_(0)
        atomic_E_2b = E_2b_ij.sum(dim=-1) / 2
        atomic_E_2b = atomic_E_2b.unsqueeze(-1)
        E_2b = scatter(atomic_E_2b, data["batch"], dim=0, reduce="sum")

        E_electric = data["electric_energy"]
        atomic_E_electric = data["atomic_electric_energy"]
        E_short = data["short_energy"]
        atomic_E_short = data["atomic_short_energy"]

        # total energy
        E_tot = E_electric + E_2b + E_short
        atomic_E = atomic_E_electric + atomic_E_2b + atomic_E_short
        data["node_energy"] = atomic_E
        data["energy"] = E_tot

        del data["atomic_electric_energy"]
        del data["electric_energy"]

        return data

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
            max_ell: int,
            interaction_cls: Type[InteractionBlock],
            interaction_cls_first: Type[InteractionBlock],
            num_interactions: int,
            num_elements: int,
            hidden_irreps: o3.Irreps,
            MLP_irreps: o3.Irreps,
            atomic_energies: np.ndarray,
            avg_num_neighbors: float,
            atomic_numbers: List[int],
            correlation: Union[int, List[int]],
            gate: Optional[Callable],
            pair_repulsion: bool = False,
            distance_transform: str = "None",
            radial_MLP: Optional[List[int]] = None,
            radial_type: Optional[str] = "bessel",
            KAN_readout: bool = False,
            default_dtype: str = "float64",
            # use long range part
            r_in: float = 1.0,
            long_range: bool = False,
            eps: float = 1e-12,
            Q_tot: float = 0.,
            A_ref: Optional[Union[float, list]] = 46.613,
            B_ref: Optional[Union[float, list]] = 3.980,
            C_ref: Optional[Union[float, list]] = 274.432,
            D_ref: Optional[Union[float, list]] = 0.,
            mu_ref: Optional[Union[float, list]] = 1.918,
            sigma_ref: Optional[Union[float, list]] = 0.25,
            eta_ref: Optional[Union[float, list]] = 0.0107,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        self.long_range = long_range

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # in long range, node_feats should concatenate Q and V
        if self.long_range:
            node_feats_irreps = node_feats_irreps + o3.Irreps("1x0e") + o3.Irreps("1x0e")
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.KAN_readout = KAN_readout
        if KAN_readout:
            self.readouts.append(KANReadoutBlock(hidden_irreps, MLP_irreps))
        else:
            self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                if KAN_readout:
                    self.readouts.append(KANReadoutBlock(hidden_irreps_out, MLP_irreps))
                else:
                    self.readouts.append(
                        NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                    )
            else:
                if KAN_readout:
                    self.readouts.append(KANReadoutBlock(hidden_irreps, MLP_irreps))
                else:
                    self.readouts.append(LinearReadoutBlock(hidden_irreps))

        if self.long_range:
            # electric equilibrium kappa calculation
            self.kappa_readout = KANReadoutBlock(irreps_in=edge_feats_irreps, MLP_irreps=MLP_irreps)
            ee4GHDNNP_config = initial_ref_params(
                num_types=num_elements,
                default_dtype=default_dtype,
                Q_tot=Q_tot,
                eps=eps,
                A_ref=A_ref,
                B_ref=B_ref,
                C_ref=C_ref,
                D_ref=D_ref,
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                eta_ref=eta_ref,
            )
            self.electric_equilibrium = electric_equilibrium(ee4GHDNNP_config, r_max, r_in)
            self.total_energy_sum = total_energy_sum(ee4GHDNNP_config)

    def forward(
            self,
            data: Dict[str, torch.Tensor],
            training: bool = False,
            compute_force: bool = True,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
            compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
                self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        '''
        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }
        '''
        return total_energy

@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
            self,
            atomic_inter_scale: float,
            atomic_inter_shift: float,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
            self,
            data: Dict[str, torch.Tensor],
            training: bool = False,
            compute_force: bool = True,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
            compute_hessian: bool = False,            
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # electric equilibrium
        if self.long_range:
            # kappa calculation
            edge_kappa = self.kappa_readout(edge_feats).squeeze(-1)
            # TODO: compute kappa with KANReadoutBlock(edge_feats), needs to be checked.
            num_nodes = data["edge_index"].max().item() + 1
            node_kappa = scatter(edge_kappa, data["edge_index"][0], dim=0, dim_size=num_nodes)
            # electric energy calculation
            data["kappa"] = node_kappa
            data["edge_length"] = lengths
            data["node_feats"] = node_feats
            data = self.electric_equilibrium(data)
            node_feats = data["node_feats"]

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
                self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # print("node_es_list", node_es_list)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        if self.long_range:
            data["short_energy"] = inter_e
            data["atomic_short_energy"] = node_inter_es
            data = self.total_energy_sum(data)
            inter_e = data["short_energy"]
            node_inter_es = data["atomic_short_energy"]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        '''
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output
        '''
        
        return total_energy


def cal_atomic_charge(data):
    """ 
    initialize ee-4HDNNP network
    """
    #z_table = np.array([8,30])
    atomic_energies = np.array([-16.036709155245,-222.745277507140])    
    # basic parameters setting for ee-4HDNNP
    model_config = dict(
        r_max=5.5,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        num_interactions=2,
        num_elements=len(data.z),
        hidden_irreps=o3.Irreps(None),
        atomic_energies=atomic_energies,
        avg_num_neighbors=1,
        atomic_numbers=data.z,
        r_in=1.0,
        long_range=True,
        Q_tot=0,
        eps=1e-12,
        A_ref=46.613,
        B_ref=3.980,
        C_ref=274.432,
        D_ref=0.0,
        mu_ref=1.918,
        sigma_ref=0.25,
        eta_ref=0.0107,
    )            
    model = ScaleShiftMACE(
        **model_config,
        pair_repulsion=False,
        KAN_readout=False,
        distance_transform=None,
        correlation=3,
        gate=torch.nn.functional.silu,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        MLP_irreps=o3.Irreps("16x0e"),
        radial_MLP=ast.literal_eval('[64, 256, 512]'),
        radial_type='bessel',
    )   
    return model(data)    