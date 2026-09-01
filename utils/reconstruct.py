"""Build RDKit molecules from PADiff's explicitly predicted molecular graph.

This module retains attribution to liGAN's GPL-2.0 reconstruction work:
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""

import numpy as np
from rdkit import Chem, Geometry


class MolReconsError(Exception):
    pass


def _is_kekulize_error(error):
    return "kekul" in (type(error).__name__ + " " + str(error)).lower()


def _build_predicted_rdkit_mol(xyz, atomic_nums, unique_edges, aromatic_flags):

    n_atoms = len(atomic_nums)
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)
    aromatic_incident = set()
    for node_i, node_j, predicted_type in unique_edges:
        if int(predicted_type) == 4:
            aromatic_incident.update((int(node_i), int(node_j)))

    for atom_idx, (atomic_num, position) in enumerate(zip(atomic_nums, xyz)):
        atom = Chem.Atom(int(atomic_num))
        atom.SetIsAromatic(
            bool(aromatic_flags[atom_idx] and atom_idx in aromatic_incident)
        )
        rd_mol.AddAtom(atom)
        rd_conf.SetAtomPosition(
            atom_idx, Geometry.Point3D(*(float(value) for value in position))
        )
    rd_mol.AddConformer(rd_conf)

    rdkit_bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for node_i, node_j, predicted_type in unique_edges:
        rd_mol.AddBond(
            int(node_i), int(node_j), rdkit_bond_types[int(predicted_type)]
        )
        if int(predicted_type) == 4:
            rd_mol.GetBondBetweenAtoms(
                int(node_i), int(node_j)
            ).SetIsAromatic(True)
    return rd_mol.GetMol()


def reconstruct_from_generated_with_edges(
    mol_info,
    check_validity=True,
    atom_is_aromatic=None,
    aromatic_repair="single",
):

    required = ("atom_pos", "element", "bond_index", "bond_type")
    missing = [key for key in required if key not in mol_info]
    if missing:
        raise MolReconsError(
            "predicted_graph_missing_fields: %s" % ", ".join(missing)
        )

    xyz = np.asarray(mol_info["atom_pos"])
    atomic_nums = np.asarray(mol_info["element"])
    bond_index = np.asarray(mol_info["bond_index"])
    bond_type = np.asarray(mol_info["bond_type"])
    n_atoms = len(atomic_nums)

    if atom_is_aromatic is None:
        atom_is_aromatic = mol_info.get("atom_is_aromatic")
    if atom_is_aromatic is None:
        aromatic_flags = np.zeros(n_atoms, dtype=bool)
        for edge_idx, predicted_type in enumerate(bond_type):
            if int(predicted_type) == 4:
                aromatic_flags[int(bond_index[0, edge_idx])] = True
                aromatic_flags[int(bond_index[1, edge_idx])] = True
        aromatic_source = "inferred_from_bonds"
    else:
        aromatic_flags = np.asarray(atom_is_aromatic, dtype=bool)
        if aromatic_flags.shape != (n_atoms,):
            raise MolReconsError(
                "predicted_atom_aromatic_shape_mismatch: expected (%d,), got %r"
                % (n_atoms, aromatic_flags.shape)
            )
        aromatic_source = "predicted_atom_classes"

    if xyz.shape != (n_atoms, 3):
        raise MolReconsError(
            "predicted_atom_position_shape_mismatch: expected (%d, 3), got %r"
            % (n_atoms, xyz.shape)
        )
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise MolReconsError(
            "predicted_bond_index_shape_mismatch: expected (2, E), got %r"
            % (bond_index.shape,)
        )
    if bond_index.shape[1] != len(bond_type):
        raise MolReconsError(
            "predicted_bond_count_mismatch: index=%d type=%d"
            % (bond_index.shape[1], len(bond_type))
        )

    unique_edges = []
    seen_edges = set()
    for edge_idx, predicted_type in enumerate(bond_type):
        node_i = int(bond_index[0, edge_idx])
        node_j = int(bond_index[1, edge_idx])
        if node_i == node_j:
            raise MolReconsError("predicted_self_bond: atom %d" % node_i)
        if not (0 <= node_i < n_atoms and 0 <= node_j < n_atoms):
            raise MolReconsError(
                "predicted_bond_index_out_of_range: (%d, %d) for %d atoms"
                % (node_i, node_j, n_atoms)
            )
        edge = (min(node_i, node_j), max(node_i, node_j))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        predicted_type = int(predicted_type)
        if predicted_type not in (1, 2, 3, 4):
            raise MolReconsError(
                "unknown_predicted_bond_order: %d" % predicted_type
            )
        unique_edges.append((edge[0], edge[1], predicted_type))

    stats = {
        "aromatic_flag_source": aromatic_source,
        "predicted_aromatic_atoms": int(aromatic_flags.sum()),
        "predicted_aromatic_bonds": int(
            sum(edge[2] == 4 for edge in unique_edges)
        ),
        "whole_graph_sanitize_attempted": bool(check_validity),
        "initial_sanitize_success": None,
        "kekulize_repair_attempted": False,
        "demoted_aromatic_bonds": 0,
        "final_sanitize_success": None,
        "edge_count_preserved": True,
    }
    mol = _build_predicted_rdkit_mol(
        xyz, atomic_nums, unique_edges, aromatic_flags
    )
    if check_validity:
        try:
            Chem.SanitizeMol(mol)
            stats["initial_sanitize_success"] = True
            stats["final_sanitize_success"] = True
        except Exception as error:
            stats["initial_sanitize_success"] = False
            repair_policy = str(aromatic_repair or "none").lower()
            can_repair = (
                _is_kekulize_error(error)
                and any(edge[2] == 4 for edge in unique_edges)
                and repair_policy in {"single", "demote"}
            )
            if not can_repair:
                mol_info["reconstruction_stats"] = stats
                raise MolReconsError(
                    "predicted_bond_sanitize_failed: %s: %s"
                    % (type(error).__name__, error)
                ) from error

            stats["kekulize_repair_attempted"] = True
            repaired_edges = [
                (node_i, node_j, 1 if predicted_type == 4 else predicted_type)
                for node_i, node_j, predicted_type in unique_edges
            ]
            stats["demoted_aromatic_bonds"] = int(
                sum(predicted_type == 4 for _, _, predicted_type in unique_edges)
            )
            repaired_flags = np.zeros(n_atoms, dtype=bool)
            mol = _build_predicted_rdkit_mol(
                xyz, atomic_nums, repaired_edges, repaired_flags
            )
            try:
                Chem.SanitizeMol(mol)
                stats["final_sanitize_success"] = True
            except Exception as repaired_error:
                stats["final_sanitize_success"] = False
                mol_info["reconstruction_stats"] = stats
                raise MolReconsError(
                    "predicted_bond_sanitize_failed_after_aromatic_demotion: %s: %s"
                    % (type(repaired_error).__name__, repaired_error)
                ) from repaired_error
    mol_info["reconstruction_stats"] = stats
    return mol
