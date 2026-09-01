from __future__ import annotations
import numpy as np
import torch
from utils.data import load_ligand_molecule
from utils.phore_realization import extract_molecule_pharmacophores


def load_heavy_ligand(path):
    """Return a sanitized heavy-atom molecule and aligned coordinates."""
    mol = load_ligand_molecule(path)
    positions = np.asarray(
        mol.GetConformer().GetPositions(), dtype=np.float32
    )
    return mol, positions


def _empty_edge():
    return torch.empty((2, 0), dtype=torch.long)


def build_training_pharmacophore_targets(mol):
    """Extract six-class features, directions, and realizing atom indices."""
    features = extract_molecule_pharmacophores(mol)
    phore_types = torch.as_tensor(
        [feature["type"] for feature in features], dtype=torch.long
    )
    if features:
        phore_positions = torch.as_tensor(
            np.stack([feature["pos"] for feature in features]),
            dtype=torch.float32,
        )
        phore_vectors = torch.as_tensor(
            np.stack([
                feature["vec"]
                if feature["vec"] is not None else np.zeros(3, dtype=np.float32)
                for feature in features
            ]),
            dtype=torch.float32,
        )
        phore_ids, atom_ids = [], []
        for phore_id, feature in enumerate(features):
            realizing_atoms = tuple(int(index) for index in feature["atom_ids"])
            phore_ids.extend([phore_id] * len(realizing_atoms))
            atom_ids.extend(realizing_atoms)
        phore_atom_index = torch.as_tensor(
            [phore_ids, atom_ids], dtype=torch.long
        )
    else:
        phore_positions = torch.empty((0, 3), dtype=torch.float32)
        phore_vectors = torch.empty((0, 3), dtype=torch.float32)
        phore_atom_index = _empty_edge()

    return {
        "phore_pos": phore_positions,
        "phore_vec": phore_vectors,
        "phore_type": phore_types,
        "ph2atom_idx": phore_atom_index,
        "protein_anchor_pos": torch.empty((0, 3), dtype=torch.float32),
        "protein_anchor_vec": torch.empty((0, 3), dtype=torch.float32),
        "protein_anchor_type": torch.empty(0, dtype=torch.long),
        "protein_anchor_confidence": torch.empty(0, dtype=torch.float32),
        "protein_anchor_source_atom": torch.empty(0, dtype=torch.long),
        "ph2anchor_edge_index": _empty_edge(),
        "ph2anchor_contact_label": torch.empty(0, dtype=torch.long),
        "ph2anchor_contact_type": torch.empty(0, dtype=torch.long),
        "ph2anchor_contact_confidence": torch.empty(0, dtype=torch.float32),
    }
