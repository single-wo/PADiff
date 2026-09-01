import torch
from torch.nn import functional as F

_DISTANCE_BOND_THRESHOLDS = {
    1: {1: 0.84, 5: 1.29, 6: 1.19, 7: 1.11, 8: 1.06, 9: 1.02,
        15: 1.54, 16: 1.44, 17: 1.37, 35: 1.51, 53: 1.71},
    5: {1: 1.29, 17: 1.85},
    6: {1: 1.19, 6: 1.64, 7: 1.57, 8: 1.53, 9: 1.45,
        15: 1.94, 16: 1.92, 17: 1.87, 35: 2.04, 53: 2.24},
    7: {1: 1.11, 6: 1.57, 7: 1.55, 8: 1.50, 9: 1.46,
        15: 1.87, 16: 1.78, 17: 1.85},
    8: {1: 1.06, 6: 1.53, 7: 1.50, 8: 1.58, 9: 1.52,
        15: 1.73, 16: 1.61, 17: 1.74},
    9: {1: 1.02, 6: 1.45, 7: 1.46, 8: 1.52, 9: 1.52,
        15: 1.66, 16: 1.68, 17: 1.76, 53: 2.01},
    15: {1: 1.54, 6: 1.94, 7: 1.87, 8: 1.73, 9: 1.66,
         15: 2.31, 16: 2.20, 17: 2.13},
    16: {1: 1.44, 6: 1.92, 7: 1.78, 8: 1.61, 9: 1.68,
         15: 2.20, 16: 2.14, 17: 2.17},
    17: {1: 1.37, 5: 1.85, 6: 1.87, 7: 1.85, 8: 1.74, 9: 1.76,
         15: 2.13, 16: 2.17, 17: 2.09, 53: 2.42},
    35: {1: 1.51, 6: 2.04, 35: 2.38},
    53: {1: 1.71, 6: 2.24, 9: 2.01, 17: 2.42, 53: 2.77},
}

_DISTANCE_BOND_THRESHOLD_TABLE = torch.zeros((128, 128), dtype=torch.float32)
for _left, _neighbors in _DISTANCE_BOND_THRESHOLDS.items():
    for _right, _threshold in _neighbors.items():
        _DISTANCE_BOND_THRESHOLD_TABLE[_left, _right] = _threshold


def _distance_bond_thresholds(elements, edge_index, dtype):

    left = elements.long()[edge_index[0]]
    right = elements.long()[edge_index[1]]
    valid = (
        (left >= 0) & (left < _DISTANCE_BOND_THRESHOLD_TABLE.size(0))
        & (right >= 0) & (right < _DISTANCE_BOND_THRESHOLD_TABLE.size(1))
    )
    thresholds = torch.zeros(left.shape, dtype=dtype, device=elements.device)
    if valid.any():
        table = _DISTANCE_BOND_THRESHOLD_TABLE.to(
            device=elements.device, dtype=dtype
        )
        thresholds[valid] = table[left[valid], right[valid]]
    return thresholds


def _zero_like(position):
    return position.new_zeros((), dtype=torch.float32)


def geometry_regularization_losses(
    pred_ligand_pos,
    target_ligand_pos,
    halfedge_index,
    halfedge_type,
    ligand_batch,
    protein_pos,
    protein_batch,
    ligand_element=None,
    bond_smooth_l1_beta=0.10,
    internal_clash_distance=1.20,
    distance_valence_margin=0.05,
    pocket_clash_distance=1.50,
    compute_bond_length=True,
    compute_internal_clash=True,
    compute_distance_valence=False,
    compute_pocket_clash=True,
):

    pred = pred_ligand_pos.float()
    target = target_ligand_pos.float()
    edge_index = halfedge_index.long()
    edge_type = halfedge_type.long()
    zero = _zero_like(pred)
    losses = {
        "ligand_bond_length": zero,
        "ligand_internal_clash": zero,
        "ligand_distance_valence": zero,
        "pocket_ligand_clash": zero,
    }

    if edge_index.numel() > 0 and (
        compute_bond_length or compute_internal_clash or compute_distance_valence
    ):
        pred_distance = torch.linalg.vector_norm(
            pred[edge_index[0]] - pred[edge_index[1]], dim=-1
        )
        if compute_bond_length:
            true_bond = (edge_type > 0) & (edge_type < 5)
            if true_bond.any():
                target_distance = torch.linalg.vector_norm(
                    target[edge_index[0, true_bond]]
                    - target[edge_index[1, true_bond]],
                    dim=-1,
                )
                losses["ligand_bond_length"] = F.smooth_l1_loss(
                    pred_distance[true_bond],
                    target_distance,
                    beta=float(bond_smooth_l1_beta),
                )
        if compute_internal_clash:
            nonbond = edge_type == 0
            if nonbond.any():
                penetration = F.relu(
                    float(internal_clash_distance) - pred_distance[nonbond]
                )
                losses["ligand_internal_clash"] = (
                    penetration.square().sum() / max(int(pred.size(0)), 1)
                )
        if compute_distance_valence and ligand_element is not None:
            nonbond = edge_type == 0
            if nonbond.any():
                thresholds = _distance_bond_thresholds(
                    ligand_element, edge_index[:, nonbond], pred_distance.dtype
                )
                supported = thresholds > 0.0
                if supported.any():
                    penetration = F.relu(
                        thresholds[supported]
                        + float(distance_valence_margin)
                        - pred_distance[nonbond][supported]
                    )
                    losses["ligand_distance_valence"] = (
                        penetration.square().sum() / max(int(pred.size(0)), 1)
                    )

    if compute_pocket_clash and pred.numel() > 0 and protein_pos.numel() > 0:
        protein = protein_pos.float()
        ligand_penalties = []
        num_graphs = int(ligand_batch.max().item()) + 1 if ligand_batch.numel() else 0
        for graph_id in range(num_graphs):
            ligand_mask = ligand_batch == graph_id
            protein_mask = protein_batch == graph_id
            if not ligand_mask.any() or not protein_mask.any():
                continue
            nearest = torch.cdist(
                pred[ligand_mask], protein[protein_mask]
            ).amin(dim=1)
            ligand_penalties.append(
                F.relu(float(pocket_clash_distance) - nearest).square()
            )
        if ligand_penalties:
            losses["pocket_ligand_clash"] = torch.cat(ligand_penalties).mean()

    return losses


@torch.no_grad()
def geometry_validation_statistics(
    pred_ligand_pos,
    target_ligand_pos,
    halfedge_index,
    halfedge_type,
    ligand_batch,
    protein_pos,
    protein_batch,
    ligand_element=None,
    internal_clash_distance=1.20,
    distance_valence_margin=0.05,
    pocket_clash_distance=1.50,
    bond_error_tolerance=0.20,
):

    pred = pred_ligand_pos.detach().float()
    target = target_ligand_pos.detach().float()
    edge_index = halfedge_index.long()
    edge_type = halfedge_type.long()
    ligand_batch = ligand_batch.long()
    protein_batch = protein_batch.long()
    num_graphs = int(ligand_batch.max().item()) + 1 if ligand_batch.numel() else 0

    bond_error_sum = 0.0
    bond_count = 0
    bond_within_tolerance = 0
    internal_clash_atoms = 0
    distance_valence_atoms = 0
    pocket_clash_atoms = 0
    ligand_atoms = int(pred.size(0))
    stable_graphs = 0

    pred_distance = None
    true_bond = None
    nonbond = None
    if edge_index.numel() > 0:
        pred_distance = torch.linalg.vector_norm(
            pred[edge_index[0]] - pred[edge_index[1]], dim=-1
        )
        true_bond = (edge_type > 0) & (edge_type < 5)
        nonbond = edge_type == 0
        if true_bond.any():
            target_distance = torch.linalg.vector_norm(
                target[edge_index[0, true_bond]]
                - target[edge_index[1, true_bond]],
                dim=-1,
            )
            error = (pred_distance[true_bond] - target_distance).abs()
            bond_error_sum = float(error.sum().item())
            bond_count = int(error.numel())
            bond_within_tolerance = int(
                (error <= float(bond_error_tolerance)).sum().item()
            )

    protein = protein_pos.detach().float()
    for graph_id in range(num_graphs):
        atom_ids = torch.nonzero(
            ligand_batch == graph_id, as_tuple=False
        ).flatten()
        if atom_ids.numel() == 0:
            continue
        graph_internal_clash = torch.zeros(
            atom_ids.numel(), dtype=torch.bool, device=pred.device
        )
        graph_distance_valence = torch.zeros(
            atom_ids.numel(), dtype=torch.bool, device=pred.device
        )
        graph_bond_errors = []
        if edge_index.numel() > 0:
            edge_graph_mask = ligand_batch[edge_index[0]] == graph_id
            edge_ids = torch.nonzero(edge_graph_mask, as_tuple=False).flatten()
            if edge_ids.numel() > 0:
                global_to_local = torch.full(
                    (pred.size(0),), -1, dtype=torch.long, device=pred.device
                )
                global_to_local[atom_ids] = torch.arange(
                    atom_ids.numel(), device=pred.device
                )
                graph_nonbond = edge_ids[nonbond[edge_ids]]
                if graph_nonbond.numel() > 0:
                    clashing = graph_nonbond[
                        pred_distance[graph_nonbond]
                        < float(internal_clash_distance)
                    ]
                    if clashing.numel() > 0:
                        graph_internal_clash[
                            global_to_local[edge_index[:, clashing]].reshape(-1)
                        ] = True
                    if ligand_element is not None:
                        thresholds = _distance_bond_thresholds(
                            ligand_element,
                            edge_index[:, graph_nonbond],
                            pred_distance.dtype,
                        )
                        inferred = graph_nonbond[
                            (thresholds > 0.0)
                            & (pred_distance[graph_nonbond]
                               < thresholds + float(distance_valence_margin))
                        ]
                        if inferred.numel() > 0:
                            graph_distance_valence[
                                global_to_local[
                                    edge_index[:, inferred]
                                ].reshape(-1)
                            ] = True
                graph_true_bond = edge_ids[true_bond[edge_ids]]
                if graph_true_bond.numel() > 0:
                    target_distance = torch.linalg.vector_norm(
                        target[edge_index[0, graph_true_bond]]
                        - target[edge_index[1, graph_true_bond]],
                        dim=-1,
                    )
                    graph_bond_errors = (
                        pred_distance[graph_true_bond] - target_distance
                    ).abs()
        internal_clash_atoms += int(graph_internal_clash.sum().item())
        distance_valence_atoms += int(graph_distance_valence.sum().item())

        protein_ids = torch.nonzero(
            protein_batch == graph_id, as_tuple=False
        ).flatten()
        graph_pocket_clash = torch.zeros(
            atom_ids.numel(), dtype=torch.bool, device=pred.device
        )
        if protein_ids.numel() > 0:
            nearest = torch.cdist(
                pred[atom_ids], protein[protein_ids]
            ).amin(dim=1)
            graph_pocket_clash = nearest < float(pocket_clash_distance)
        pocket_clash_atoms += int(graph_pocket_clash.sum().item())

        graph_bond_ok = (
            not torch.is_tensor(graph_bond_errors)
            or graph_bond_errors.numel() == 0
            or bool((graph_bond_errors <= float(bond_error_tolerance)).all().item())
        )
        stable_graphs += int(
            graph_bond_ok
            and not graph_internal_clash.any().item()
            and not graph_distance_valence.any().item()
            and not graph_pocket_clash.any().item()
        )

    return {
        "bond_error_sum": bond_error_sum,
        "bond_count": bond_count,
        "bond_within_tolerance": bond_within_tolerance,
        "internal_clash_atoms": internal_clash_atoms,
        "distance_valence_atoms": distance_valence_atoms,
        "pocket_clash_atoms": pocket_clash_atoms,
        "ligand_atoms": ligand_atoms,
        "stable_graphs": stable_graphs,
        "graphs": num_graphs,
    }
