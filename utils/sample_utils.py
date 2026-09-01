import torch

DEFAULT_FOLLOW_BATCH = [
    "protein_element", "ligand_element", "phore_type", "protein_anchor_type"
]


def _as_tensor(value):
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def separate_outputs(
    outputs,
    n_graphs,
    batch_node,
    halfedge_index,
    batch_halfedge,
    batch_phore=None,
):

    batch_node = _as_tensor(batch_node)
    halfedge_index = _as_tensor(halfedge_index)
    batch_halfedge = _as_tensor(batch_halfedge)
    batch_phore = None if batch_phore is None else _as_tensor(batch_phore)
    outputs_pred = outputs["pred"]
    outputs_traj = outputs["traj"]
    phore_pred = outputs.get("phore_pred")
    phore_traj = outputs.get("phore_traj")
    phore_atom_assignment = outputs.get("phore_atom_assignment")
    terminal_ligand_node_state = outputs.get("terminal_ligand_node_state")
    terminal_ligand_halfedge_state = outputs.get("terminal_ligand_halfedge_state")
    terminal_phore_node_state = outputs.get("terminal_phore_node_state")

    new_outputs = []
    for graph_id in range(n_graphs):
        node_mask = batch_node == graph_id
        halfedge_mask = batch_halfedge == graph_id
        n_nodes = int(node_mask.sum())
        assert n_nodes * (n_nodes - 1) == int(halfedge_mask.sum()) * 2
        pred = [
            outputs_pred[0][node_mask],
            outputs_pred[1][node_mask],
            outputs_pred[2][halfedge_mask],
        ]
        traj = [
            outputs_traj[0][:, node_mask],
            outputs_traj[1][:, node_mask],
            outputs_traj[2][:, halfedge_mask],
        ]
        edge_index = halfedge_index[:, halfedge_mask]
        node_ids = torch.nonzero(node_mask, as_tuple=False).flatten()
        if edge_index.numel() > 0:
            edge_index = edge_index - node_ids.min()

        item = {"pred": pred, "traj": traj, "halfedge_index": edge_index}
        if terminal_ligand_node_state is not None:
            item["terminal_ligand_node_state"] = terminal_ligand_node_state[node_mask]
        if terminal_ligand_halfedge_state is not None:
            item["terminal_ligand_halfedge_state"] = terminal_ligand_halfedge_state[halfedge_mask]
        if batch_phore is not None and phore_pred is not None:
            phore_mask = batch_phore == graph_id
            item["phore_pred"] = [
                phore_pred[0][phore_mask],
                phore_pred[1][phore_mask],
                phore_pred[2][phore_mask],
            ]
            if terminal_phore_node_state is not None:
                item["terminal_phore_node_state"] = terminal_phore_node_state[phore_mask]
            if phore_traj is not None:
                item["phore_traj"] = [
                    phore_traj[0][:, phore_mask],
                    phore_traj[1][:, phore_mask],
                    phore_traj[2][:, phore_mask],
                ]
            if phore_atom_assignment is not None:
                assignment_index = phore_atom_assignment["index"]
                assignment_logits = phore_atom_assignment["logits"]
                pair_mask = phore_mask[assignment_index[0]] & node_mask[assignment_index[1]]
                local_index = assignment_index[:, pair_mask].clone()
                phore_ids = torch.nonzero(phore_mask, as_tuple=False).flatten()
                node_ids_for_assignment = torch.nonzero(node_mask, as_tuple=False).flatten()
                if local_index.numel() > 0:
                    local_index[0] -= phore_ids.min()
                    local_index[1] -= node_ids_for_assignment.min()
                item["phore_atom_assignment"] = {
                    "index": local_index,
                    "logits": assignment_logits[pair_mask],
                    "atom_type_logits": phore_atom_assignment["atom_type_logits"][node_mask],
                }
        new_outputs.append(item)
    return new_outputs
